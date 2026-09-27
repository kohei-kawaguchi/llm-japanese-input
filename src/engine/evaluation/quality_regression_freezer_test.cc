// Copyright 2026 LLM Japanese Input Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//     * Neither the name of LLM Japanese Input Authors nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "engine/evaluation/quality_regression_freezer.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "converter/candidate.h"
#include "converter/converter_mock.h"
#include "converter/segments.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/evaluation_freezer.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"
#include "testing/gmock.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

using ::testing::_;
using ::testing::InSequence;

constexpr char kHex64[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kRevision40[] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kEvaluationClockUtcRfc3339[] = "2000-01-01T00:00:00Z";

void FillSourceIdentity(EvaluationSourceIdentity* source) {
  source->set_benchmark_name("Synthetic quality regression");
  source->set_source_revision(kRevision40);
  source->set_source_relative_path("synthetic/quality.tsv");
  source->set_source_sha256(kHex64);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic separated corpus.");
}

QualityRegressionInputCorpus MakeInputCorpus() {
  QualityRegressionInputCorpus input;
  input.set_schema_version(kQualityRegressionCorpusSchemaVersion);
  QualityRegressionCorpusIdentity* identity = input.mutable_identity();
  identity->set_role(DEVELOPMENT);
  FillSourceIdentity(identity->mutable_source());
  identity->set_parser_definition_version(
      kQualityRegressionParserDefinitionVersion);
  identity->set_normalization_definition_version(
      kQualityRegressionNormalizationDefinitionVersion);
  identity->set_import_config_sha256(kHex64);
  QualityRegressionInputCase* first = input.add_cases();
  first->set_source_line(2);
  first->set_reading("ｍ");
  QualityRegressionInputCase* second = input.add_cases();
  second->set_source_line(7);
  second->set_reading("かな");
  return input;
}

std::string MessageSha256(const protobuf::Message& message) {
  absl::StatusOr<std::string> hash = DeterministicMessageSha256(message);
  EXPECT_TRUE(hash.ok()) << hash.status();
  return hash.ok() ? *hash : std::string();
}

QualityRegressionFreezerConfig MakeFreezerConfig(
    const QualityRegressionInputCorpus& input) {
  QualityRegressionFreezerConfig config;
  config.set_schema_version(kQualityRegressionFreezerConfigSchemaVersion);
  config.set_input_corpus_schema_version(
      kQualityRegressionCorpusSchemaVersion);
  config.set_frozen_corpus_schema_version(
      kQualityRegressionFrozenCorpusSchemaVersion);
  *config.mutable_expected_corpus_identity() = input.identity();
  config.set_input_corpus_sha256(MessageSha256(input));
  config.set_expected_case_count(input.cases_size());
  EvaluationMozcIdentity* mozc = config.mutable_mozc();
  mozc->set_source_revision(kRevision40);
  mozc->set_data_type("oss");
  mozc->set_data_sha256(kHex64);
  mozc->set_default_desktop_request_sha256(
      MessageSha256(MakeQualityRegressionDefaultDesktopRequest()));
  mozc->set_default_desktop_config_sha256(
      MessageSha256(MakeQualityRegressionDefaultDesktopConfig()));
  mozc->set_evaluation_clock_utc_rfc3339(kEvaluationClockUtcRfc3339);
  return config;
}

void AddCandidate(Segment* segment, std::string key, std::string value,
                  int32_t cost, uint32_t attributes,
                  uint32_t consumed_key_size,
                  converter::Candidate::Command command =
                      converter::Candidate::DEFAULT_COMMAND) {
  converter::Candidate* candidate = segment->push_back_candidate();
  candidate->key = std::move(key);
  candidate->value = std::move(value);
  candidate->cost = cost;
  candidate->attributes = attributes;
  candidate->consumed_key_size = consumed_key_size;
  candidate->command = command;
}

absl::StatusOr<std::string> ReadConfig(absl::string_view filename) {
  return FileUtil::GetContents(
      testing::GetSourceFileOrDie({"engine", "evaluation", filename}));
}

TEST(QualityRegressionFreezerTest, DefaultDesktopIdentityUsesTypingCorrection) {
  const commands::Request request =
      MakeQualityRegressionDefaultDesktopRequest();
  const config::Config config = MakeQualityRegressionDefaultDesktopConfig();
  EXPECT_TRUE(config.use_typing_correction());
  EXPECT_FALSE(config.has_candidate_ranking_config());
  const std::string request_sha256 = MessageSha256(request);
  const std::string config_sha256 = MessageSha256(config);
  EXPECT_EQ(request_sha256,
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(config_sha256,
            "747675393bfacbb3542de2a1b80b9273"
            "be46c7987b01956811b5783298958f31");
}

TEST(QualityRegressionFreezerTest,
     FreezesConversionQueryWithEmptyContextAndSeparatedIdentity) {
  const QualityRegressionInputCorpus input = MakeInputCorpus();
  const QualityRegressionFreezerConfig config = MakeFreezerConfig(input);
  const std::string freezer_config_sha256 =
      Sha256Bytes("synthetic raw freezer config");
  StrictMockConverter converter;
  {
    InSequence sequence;
    EXPECT_CALL(converter, ResetConversion(_))
        .WillOnce([](Segments* segments) { segments->Clear(); });
    EXPECT_CALL(converter, StartConversion(_, _))
        .WillOnce([](const ConversionRequest& request, Segments* segments) {
          EXPECT_TRUE(request.config().use_typing_correction());
          EXPECT_FALSE(request.config().has_candidate_ranking_config());
          EXPECT_EQ(request.key(), "m");
          EXPECT_TRUE(request.context().has_preceding_text());
          EXPECT_EQ(request.context().preceding_text(), "");
          EXPECT_TRUE(request.context().has_following_text());
          EXPECT_EQ(request.context().following_text(), "");
          Segment* first = segments->add_segment();
          first->set_key("m");
          AddCandidate(first, "m", "第一", 101, 0x10, 3);
          AddCandidate(first, "m", "第壱", -202, 0x20, 6,
                       converter::Candidate::ENABLE_INCOGNITO_MODE);
          Segment* second = segments->add_segment();
          second->set_key("next");
          AddCandidate(second, "next", "第二", 303, 0x40, 9);
          return true;
        });
    EXPECT_CALL(converter, ResetConversion(_))
        .WillOnce([](Segments* segments) { segments->Clear(); });
    EXPECT_CALL(converter, StartConversion(_, _))
        .WillOnce([](const ConversionRequest& request, Segments* segments) {
          EXPECT_TRUE(request.config().use_typing_correction());
          EXPECT_EQ(request.key(), "かな");
          Segment* segment = segments->add_segment();
          segment->set_key("かな");
          AddCandidate(segment, "かな", "仮名", 7, 0, 6);
          return true;
        });
  }
  EXPECT_CALL(converter, ReconstructHistory(_, _)).Times(0);
  EXPECT_CALL(converter, FinishConversion(_, _)).Times(0);
  EXPECT_CALL(converter, CommitSegmentValue(_, _, _)).Times(0);
  EXPECT_CALL(converter, CommitSegments(_, _)).Times(0);
  EXPECT_CALL(converter, CommitContext(_)).Times(0);

  absl::StatusOr<QualityRegressionFrozenCorpus> frozen =
      FreezeQualityRegressionCorpus(input, config.input_corpus_sha256(),
                                    config, freezer_config_sha256, converter);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cases_size(), 2);
  EXPECT_EQ(frozen->identity().role(), DEVELOPMENT);
  EXPECT_EQ(frozen->input_corpus_sha256(), config.input_corpus_sha256());
  EXPECT_EQ(frozen->freezer_config_sha256(), freezer_config_sha256);
  EXPECT_EQ(frozen->mozc().default_desktop_config_sha256(),
            config.mozc().default_desktop_config_sha256());

  const QualityRegressionFrozenCase& first = frozen->cases(0);
  EXPECT_EQ(first.source_line(), 2);
  EXPECT_EQ(first.conversion_reading(), "m");
  EXPECT_EQ(first.mozc_baseline_output(), "第一第二");
  EXPECT_EQ(first.request().reading(), "m");
  EXPECT_EQ(first.request().preceding_text(), "");
  EXPECT_EQ(first.request().following_text(), "");
  EXPECT_EQ(first.request().token().request_sequence(), 1);
  EXPECT_EQ(first.request().mode(),
            FrozenCandidateRankerRequest::MODE_CONVERSION);
  ASSERT_EQ(first.request().segments_size(), 2);
  EXPECT_EQ(first.request().segments(0).candidates(1).id(), 1);
  EXPECT_TRUE(first.request().segments(0).candidates(1).is_protected());
  EXPECT_EQ(frozen->cases(1).source_line(), 7);
  EXPECT_EQ(frozen->cases(1).conversion_reading(), "かな");
  EXPECT_EQ(frozen->cases(1).mozc_baseline_output(), "仮名");
  EXPECT_EQ(frozen->cases(1).request().token().request_sequence(), 2);
  EXPECT_TRUE(ValidateQualityRegressionFrozenCorpus(
                  *frozen, config, freezer_config_sha256)
                  .ok());

  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*frozen);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*frozen);
  ASSERT_TRUE(first_binary.ok()) << first_binary.status();
  ASSERT_TRUE(second_binary.ok()) << second_binary.status();
  EXPECT_EQ(*first_binary, *second_binary);
  EXPECT_EQ(Sha256Bytes(*first_binary),
            "7ed00857dcb3b4ba4acc1e90acf02439"
            "1147d62ec7471a42dc3008ecd39d14f8");
}

TEST(QualityRegressionFreezerTest, RejectsInputAndFrozenIdentityPoison) {
  const QualityRegressionInputCorpus input = MakeInputCorpus();
  const QualityRegressionFreezerConfig config = MakeFreezerConfig(input);
  EXPECT_EQ(ValidateQualityRegressionFreezerInput(input, kHex64, config).code(),
            absl::StatusCode::kFailedPrecondition);

  QualityRegressionFreezerConfig unknown_config = config;
  std::string bytes;
  ASSERT_TRUE(unknown_config.SerializeToString(&bytes));
  bytes.append("\xa0\x06\x01", 3);
  ASSERT_TRUE(unknown_config.ParseFromString(bytes));
  EXPECT_EQ(ValidateQualityRegressionFreezerConfig(unknown_config).code(),
            absl::StatusCode::kInvalidArgument);

  QualityRegressionFrozenCorpus incomplete;
  EXPECT_EQ(ValidateQualityRegressionFrozenCorpus(
                incomplete, config, Sha256Bytes("synthetic config"))
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionFreezerTest, CheckedConfigsPinBothInputPartitions) {
  absl::StatusOr<std::string> development_bytes =
      ReadConfig("quality_regression_development_freezer_config.textproto");
  absl::StatusOr<std::string> holdout_bytes =
      ReadConfig("quality_regression_holdout_freezer_config.textproto");
  ASSERT_TRUE(development_bytes.ok()) << development_bytes.status();
  ASSERT_TRUE(holdout_bytes.ok()) << holdout_bytes.status();
  EXPECT_EQ(development_bytes->size(), 1779);
  EXPECT_EQ(holdout_bytes->size(), 1724);
  EXPECT_EQ(Sha256Bytes(*development_bytes),
            "b9818bc4fb205b76c2e7555e2a37556c"
            "f35084e367b08d2e0b7a88914c463ce3");
  EXPECT_EQ(Sha256Bytes(*holdout_bytes),
            "570fd92630dfb07d54c6ed5af456ea7b"
            "d2bbebc4cb331a74490c3fda2746fd12");
  QualityRegressionFreezerConfig development;
  QualityRegressionFreezerConfig holdout;
  ASSERT_TRUE(ParseTextproto(*development_bytes, &development).ok());
  ASSERT_TRUE(ParseTextproto(*holdout_bytes, &holdout).ok());
  EXPECT_TRUE(ValidateQualityRegressionFreezerConfig(development).ok());
  EXPECT_TRUE(ValidateQualityRegressionFreezerConfig(holdout).ok());
  EXPECT_EQ(development.expected_corpus_identity().role(), DEVELOPMENT);
  EXPECT_EQ(holdout.expected_corpus_identity().role(), HOLDOUT);
  EXPECT_EQ(development.expected_case_count(), 219);
  EXPECT_EQ(holdout.expected_case_count(), 72);
  EXPECT_EQ(development.input_corpus_sha256(),
            "7f06025eacc5d43d7a56aa8b92f79a98"
            "2ed54699960a73b764fb322c0d84272f");
  EXPECT_EQ(holdout.input_corpus_sha256(),
            "97e5b7eb6a6c7f831f5306725a552eee"
            "6d398def178fb5c38601e962b5084087");
  EXPECT_EQ(development.mozc().data_sha256(),
            "fbb9a23419d677a33968145cfe1979da"
            "c6e7d3c28712d469854b4707a5965ed1");
  EXPECT_EQ(development.mozc().default_desktop_config_sha256(),
            "747675393bfacbb3542de2a1b80b9273"
            "be46c7987b01956811b5783298958f31");
  EXPECT_EQ(development.mozc().SerializeAsString(),
            holdout.mozc().SerializeAsString());
}

}  // namespace
}  // namespace mozc::engine::evaluation
