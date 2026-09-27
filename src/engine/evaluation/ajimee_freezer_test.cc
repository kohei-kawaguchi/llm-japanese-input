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

#include "engine/evaluation/ajimee_freezer.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "base/clock.h"
#include "base/clock_mock.h"
#include "base/file_util.h"
#include "converter/candidate.h"
#include "converter/converter_mock.h"
#include "converter/segments.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
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
using ::testing::Return;

constexpr char kHex64[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kRevision40[] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kEvaluationClockUtcRfc3339[] = "2000-01-01T00:00:00Z";

void FillSourceIdentity(EvaluationSourceIdentity* source) {
  source->set_benchmark_name("AJIMEE-Bench synthetic");
  source->set_source_revision(kRevision40);
  source->set_source_relative_path("synthetic/evaluation_items.json");
  source->set_source_sha256(kHex64);
  source->set_creator("Synthetic test creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("CC-BY-SA-3.0");
  source->set_license_url("https://creativecommons.org/licenses/by-sa/3.0/");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic transformed corpus.");
}

AjimeeInputCorpus MakeInputCorpus() {
  AjimeeInputCorpus corpus;
  corpus.set_schema_version(kAjimeeInputCorpusSchemaVersion);
  FillSourceIdentity(corpus.mutable_source());

  AjimeeInputCase* no_context = corpus.add_cases();
  no_context->set_source_index(1);
  no_context->set_complete_katakana_reading("アメ");
  no_context->set_published_preceding_context("");
  no_context->set_context_slice(AjimeeInputCase::CONTEXT_SLICE_NO_CONTEXT);
  no_context->set_has_source_split_data(false);

  AjimeeInputCase* with_context = corpus.add_cases();
  with_context->set_source_index(3);
  with_context->set_complete_katakana_reading("コウショウ");
  with_context->set_published_preceding_context("契約ABC");
  with_context->set_context_slice(AjimeeInputCase::CONTEXT_SLICE_HAS_CONTEXT);
  with_context->set_has_source_split_data(true);
  return corpus;
}

std::string MessageSha256(const protobuf::Message& message) {
  absl::StatusOr<std::string> bytes = SerializeDeterministically(message);
  EXPECT_TRUE(bytes.ok()) << bytes.status();
  return bytes.ok() ? Sha256Bytes(*bytes) : std::string();
}

AjimeeFreezerConfig MakeFreezerConfig(const AjimeeInputCorpus& input,
                                      const commands::Request& request,
                                      const config::Config& config) {
  AjimeeFreezerConfig freezer_config;
  freezer_config.set_schema_version(kAjimeeFreezerConfigSchemaVersion);
  freezer_config.set_input_corpus_schema_version(
      kAjimeeInputCorpusSchemaVersion);
  freezer_config.set_frozen_corpus_schema_version(
      kAjimeeFrozenCorpusSchemaVersion);
  *freezer_config.mutable_expected_source() = input.source();
  freezer_config.set_input_corpus_sha256(MessageSha256(input));
  freezer_config.set_expected_case_count(input.cases_size());
  freezer_config.set_mozc_source_revision(kRevision40);
  freezer_config.set_mozc_data_type("oss");
  freezer_config.set_mozc_data_sha256(kHex64);
  freezer_config.set_default_desktop_request_sha256(MessageSha256(request));
  freezer_config.set_default_desktop_config_sha256(MessageSha256(config));
  freezer_config.set_evaluation_clock_utc_rfc3339(
      kEvaluationClockUtcRfc3339);
  return freezer_config;
}

void AddCandidate(Segment* segment, std::string key, std::string value,
                  int32_t cost, uint32_t attributes, uint32_t consumed_key_size,
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

TEST(AjimeeFreezerTest, FreezesFullPostRewriteSnapshotWithoutLearning) {
  const AjimeeInputCorpus input = MakeInputCorpus();
  const commands::Request request = MakeAjimeeDefaultDesktopRequest();
  const config::Config config = MakeAjimeeDefaultDesktopConfig();
  const AjimeeFreezerConfig freezer_config =
      MakeFreezerConfig(input, request, config);
  StrictMockConverter converter;
  {
    InSequence sequence;
    EXPECT_CALL(converter, ResetConversion(_)).WillOnce([](Segments* segments) {
      segments->Clear();
    });
    EXPECT_CALL(converter, StartConversion(_, _))
        .WillOnce([](const ConversionRequest& conversion_request,
                     Segments* segments) {
          EXPECT_EQ(conversion_request.request_type(),
                    ConversionRequest::CONVERSION);
          EXPECT_TRUE(conversion_request.options().defer_candidate_limits);
          EXPECT_EQ(conversion_request.key(), "あめ");
          EXPECT_TRUE(conversion_request.context().has_preceding_text());
          EXPECT_EQ(conversion_request.context().preceding_text(), "");
          EXPECT_TRUE(conversion_request.context().has_following_text());
          EXPECT_EQ(conversion_request.context().following_text(), "");
          EXPECT_EQ(segments->history_segments_size(), 0);

          Segment* first = segments->add_segment();
          first->set_key("あめ");
          AddCandidate(first, "あめ", "雨", 101, 0x20, 3);
          AddCandidate(first, "あめ", "飴", -202, 0x20, 6,
                       converter::Candidate::ENABLE_INCOGNITO_MODE);
          Segment* second = segments->add_segment();
          second->set_key("です");
          AddCandidate(second, "です", "です", 303, 0, 9);
          AddCandidate(second, "です", "デス", 404, 0, 12,
                       converter::Candidate::ENABLE_INCOGNITO_MODE);
          return true;
        });
    EXPECT_CALL(converter, ReconstructHistory(_, "契約ABC"))
        .WillOnce([](Segments* segments, absl::string_view) {
          segments->Clear();
          Segment* history = segments->add_segment();
          history->set_segment_type(Segment::HISTORY);
          history->set_key("ABC");
          AddCandidate(history, "ABC", "ABC", 1, 0, 0);
          return true;
        });
    EXPECT_CALL(converter, StartConversion(_, _))
        .WillOnce([](const ConversionRequest& conversion_request,
                     Segments* segments) {
          EXPECT_TRUE(conversion_request.options().defer_candidate_limits);
          EXPECT_EQ(conversion_request.key(), "こうしょう");
          EXPECT_EQ(conversion_request.context().preceding_text(), "契約ABC");
          EXPECT_EQ(segments->history_segments_size(), 1);
          Segment* segment = segments->add_segment();
          segment->set_key("こうしょう");
          AddCandidate(segment, "こうしょう", "交渉", 7, 0x40, 15);
          AddCandidate(segment, "こうしょう", "校章", 8, 0, 18);
          return true;
        });
  }
  EXPECT_CALL(converter, FinishConversion(_, _)).Times(0);
  EXPECT_CALL(converter, CommitSegmentValue(_, _, _)).Times(0);
  EXPECT_CALL(converter, CommitSegments(_, _)).Times(0);
  EXPECT_CALL(converter, CommitContext(_)).Times(0);

  absl::StatusOr<AjimeeFrozenCorpus> frozen =
      FreezeAjimeeCorpus(input, freezer_config.input_corpus_sha256(),
                         freezer_config, request, config, converter);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cases_size(), 2);
  EXPECT_EQ(frozen->source().source_revision(), kRevision40);
  EXPECT_EQ(frozen->input_corpus_sha256(),
            freezer_config.input_corpus_sha256());
  EXPECT_EQ(frozen->mozc().data_sha256(), kHex64);
  EXPECT_EQ(frozen->mozc().evaluation_clock_utc_rfc3339(),
            kEvaluationClockUtcRfc3339);

  const AjimeeFrozenCase& first = frozen->cases(0);
  EXPECT_EQ(first.source_index(), 1);
  EXPECT_EQ(first.normalized_hiragana_reading(), "あめ");
  EXPECT_FALSE(first.history_reconstructed());
  EXPECT_EQ(first.baseline_output(), "雨です");
  EXPECT_EQ(first.request().token().session_generation(), 0);
  EXPECT_EQ(first.request().token().state_revision(), 0);
  EXPECT_EQ(first.request().token().request_sequence(), 1);
  EXPECT_EQ(first.request().mode(),
            FrozenCandidateRankerRequest::MODE_CONVERSION);
  EXPECT_EQ(first.request().preceding_text(), "");
  EXPECT_EQ(first.request().following_text(), "");
  EXPECT_EQ(first.request().reading(), "あめ");
  EXPECT_EQ(first.request().focused_segment_id(), 0);
  ASSERT_EQ(first.request().segments_size(), 2);
  const FrozenCandidateRankerCandidate& protected_candidate =
      first.request().segments(0).candidates(1);
  EXPECT_EQ(protected_candidate.id(), 1);
  EXPECT_EQ(protected_candidate.key(), "あめ");
  EXPECT_EQ(protected_candidate.value(), "飴");
  EXPECT_EQ(protected_candidate.cost(), -202);
  EXPECT_EQ(protected_candidate.attributes(), 0x20);
  EXPECT_EQ(protected_candidate.consumed_key_size(), 6);
  EXPECT_TRUE(protected_candidate.is_protected());
  EXPECT_TRUE(first.request().segments(1).candidates(1).is_protected());

  const AjimeeFrozenCase& second = frozen->cases(1);
  EXPECT_EQ(second.normalized_hiragana_reading(), "こうしょう");
  EXPECT_TRUE(second.history_reconstructed());
  EXPECT_EQ(second.baseline_output(), "交渉");
  EXPECT_EQ(second.request().token().request_sequence(), 2);
  EXPECT_EQ(second.request().preceding_text(), "契約ABC");
  EXPECT_TRUE(ValidateAjimeeFrozenCorpus(*frozen, freezer_config).ok());
  AjimeeFreezerConfig mismatched_identity = freezer_config;
  mismatched_identity.set_mozc_data_type("mismatched");
  EXPECT_EQ(
      ValidateAjimeeFrozenCorpus(*frozen, mismatched_identity).code(),
      absl::StatusCode::kFailedPrecondition);

  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*frozen);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*frozen);
  ASSERT_TRUE(first_binary.ok());
  ASSERT_TRUE(second_binary.ok());
  EXPECT_EQ(*first_binary, *second_binary);
  EXPECT_EQ(MessageSha256(*frozen),
            "f4a4b3e1ba657e9a33e78d58359c8ab0"
            "9e3555adb9cc5dcb180a14f445319837");
  AjimeeFrozenCorpus round_trip;
  ASSERT_TRUE(round_trip.ParseFromString(*first_binary));
  EXPECT_EQ(MessageSha256(round_trip), MessageSha256(*frozen));
  absl::StatusOr<std::string> review = ReviewTextproto(*frozen);
  ASSERT_TRUE(review.ok());
  EXPECT_NE(review->find("交渉"), std::string::npos);
}

TEST(AjimeeFreezerTest, RecordsFailedHistoryReconstructionWithoutFallback) {
  AjimeeInputCorpus input = MakeInputCorpus();
  input.mutable_cases()->DeleteSubrange(0, 1);
  const commands::Request request = MakeAjimeeDefaultDesktopRequest();
  const config::Config config = MakeAjimeeDefaultDesktopConfig();
  const AjimeeFreezerConfig freezer_config =
      MakeFreezerConfig(input, request, config);
  StrictMockConverter converter;
  EXPECT_CALL(converter, ReconstructHistory(_, "契約ABC"))
      .WillOnce(Return(false));
  EXPECT_CALL(converter, StartConversion(_, _))
      .WillOnce([](const ConversionRequest&, Segments* segments) {
        Segment* segment = segments->add_segment();
        segment->set_key("こうしょう");
        AddCandidate(segment, "こうしょう", "交渉", 1, 0, 0);
        return true;
      });

  absl::StatusOr<AjimeeFrozenCorpus> frozen =
      FreezeAjimeeCorpus(input, freezer_config.input_corpus_sha256(),
                         freezer_config, request, config, converter);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cases_size(), 1);
  EXPECT_FALSE(frozen->cases(0).history_reconstructed());
}

TEST(AjimeeFreezerTest,
     PinsUtcClockForTimeSensitiveImaAcrossDifferentLiveTimes) {
  AjimeeInputCorpus input = MakeInputCorpus();
  input.clear_cases();
  AjimeeInputCase* input_case = input.add_cases();
  input_case->set_source_index(1699);
  input_case->set_complete_katakana_reading("イマ");
  input_case->set_published_preceding_context("");
  input_case->set_context_slice(AjimeeInputCase::CONTEXT_SLICE_NO_CONTEXT);
  input_case->set_has_source_split_data(false);
  const commands::Request request = MakeAjimeeDefaultDesktopRequest();
  const config::Config config = MakeAjimeeDefaultDesktopConfig();
  const AjimeeFreezerConfig freezer_config =
      MakeFreezerConfig(input, request, config);

  const auto freeze_at_live_time = [&](absl::Time live_time) {
    ScopedClockMock live_clock(live_time);
    StrictMockConverter converter;
    EXPECT_CALL(converter, ResetConversion(_))
        .WillOnce([](Segments* segments) { segments->Clear(); });
    EXPECT_CALL(converter, StartConversion(_, _))
        .WillOnce([](const ConversionRequest& conversion_request,
                     Segments* segments) {
          EXPECT_EQ(conversion_request.key(), "いま");
          Segment* segment = segments->add_segment();
          segment->set_key("いま");
          AddCandidate(segment, "いま", "今", 1, 0, 0);
          AddCandidate(segment, "いま",
                       absl::FormatTime("%H:%M", Clock::GetAbslTime(),
                                        Clock::GetTimeZone()),
                       2, 0, 0);
          return true;
        });
    absl::StatusOr<AjimeeFrozenCorpus> frozen = FreezeAjimeeCorpus(
        input, freezer_config.input_corpus_sha256(), freezer_config, request,
        config, converter);
    EXPECT_EQ(Clock::GetAbslTime(), live_time);
    return frozen;
  };

  absl::StatusOr<AjimeeFrozenCorpus> first = freeze_at_live_time(
      absl::FromCivil(absl::CivilSecond(2026, 8, 19, 10, 38, 0),
                      absl::UTCTimeZone()));
  absl::StatusOr<AjimeeFrozenCorpus> second = freeze_at_live_time(
      absl::FromCivil(absl::CivilSecond(2026, 8, 19, 10, 43, 0),
                      absl::UTCTimeZone()));
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(first->cases_size(), 1);
  ASSERT_EQ(first->cases(0).request().segments_size(), 1);
  ASSERT_EQ(first->cases(0).request().segments(0).candidates_size(), 2);
  EXPECT_EQ(first->cases(0).request().segments(0).candidates(1).value(),
            "00:00");
  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*first);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*second);
  ASSERT_TRUE(first_binary.ok()) << first_binary.status();
  ASSERT_TRUE(second_binary.ok()) << second_binary.status();
  EXPECT_EQ(*first_binary, *second_binary);
}

TEST(AjimeeFreezerTest, RejectsInvalidInputIdentityAndOrdering) {
  const commands::Request request = MakeAjimeeDefaultDesktopRequest();
  const config::Config config = MakeAjimeeDefaultDesktopConfig();

  AjimeeInputCorpus incomplete = MakeInputCorpus();
  AjimeeFreezerConfig incomplete_config =
      MakeFreezerConfig(incomplete, request, config);
  incomplete.mutable_cases(0)->clear_complete_katakana_reading();
  StrictMockConverter incomplete_converter;
  absl::StatusOr<AjimeeFrozenCorpus> incomplete_result = FreezeAjimeeCorpus(
      incomplete, incomplete_config.input_corpus_sha256(), incomplete_config,
      request, config, incomplete_converter);
  EXPECT_EQ(incomplete_result.status().code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeInputCorpus duplicate = MakeInputCorpus();
  duplicate.mutable_cases(1)->set_source_index(1);
  AjimeeFreezerConfig duplicate_config =
      MakeFreezerConfig(duplicate, request, config);
  StrictMockConverter duplicate_converter;
  absl::StatusOr<AjimeeFrozenCorpus> duplicate_result = FreezeAjimeeCorpus(
      duplicate, duplicate_config.input_corpus_sha256(), duplicate_config,
      request, config, duplicate_converter);
  EXPECT_EQ(duplicate_result.status().code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeInputCorpus schema = MakeInputCorpus();
  schema.set_schema_version(2);
  AjimeeFreezerConfig schema_config =
      MakeFreezerConfig(schema, request, config);
  StrictMockConverter schema_converter;
  absl::StatusOr<AjimeeFrozenCorpus> schema_result =
      FreezeAjimeeCorpus(schema, schema_config.input_corpus_sha256(),
                         schema_config, request, config, schema_converter);
  EXPECT_EQ(schema_result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeFreezerTest, RejectsInvalidConfigAndCandidateSnapshot) {
  AjimeeInputCorpus input = MakeInputCorpus();
  input.mutable_cases()->DeleteSubrange(1, 1);
  const commands::Request request = MakeAjimeeDefaultDesktopRequest();
  const config::Config config = MakeAjimeeDefaultDesktopConfig();
  AjimeeFreezerConfig freezer_config =
      MakeFreezerConfig(input, request, config);

  AjimeeFreezerConfig missing_hash = freezer_config;
  missing_hash.clear_input_corpus_sha256();
  EXPECT_EQ(ValidateAjimeeFreezerConfig(missing_hash).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeFreezerConfig missing_clock = freezer_config;
  missing_clock.clear_evaluation_clock_utc_rfc3339();
  EXPECT_EQ(ValidateAjimeeFreezerConfig(missing_clock).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeFreezerConfig noncanonical_clock = freezer_config;
  noncanonical_clock.set_evaluation_clock_utc_rfc3339(
      "2000-01-01T00:00:00+00:00");
  EXPECT_EQ(ValidateAjimeeFreezerConfig(noncanonical_clock).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeFreezerConfig invalid_clock = freezer_config;
  invalid_clock.set_evaluation_clock_utc_rfc3339("invalid");
  EXPECT_EQ(ValidateAjimeeFreezerConfig(invalid_clock).code(),
            absl::StatusCode::kInvalidArgument);

  config::Config ranking_config = config;
  ranking_config.mutable_candidate_ranking_config()->set_enabled(false);
  AjimeeFreezerConfig ranking_freezer_config =
      MakeFreezerConfig(input, request, ranking_config);
  StrictMockConverter ranking_converter;
  absl::StatusOr<AjimeeFrozenCorpus> ranking_result = FreezeAjimeeCorpus(
      input, ranking_freezer_config.input_corpus_sha256(),
      ranking_freezer_config, request, ranking_config, ranking_converter);
  EXPECT_EQ(ranking_result.status().code(), absl::StatusCode::kInvalidArgument);

  StrictMockConverter empty_candidates_converter;
  EXPECT_CALL(empty_candidates_converter, ResetConversion(_));
  EXPECT_CALL(empty_candidates_converter, StartConversion(_, _))
      .WillOnce([](const ConversionRequest&, Segments* segments) {
        segments->add_segment()->set_key("あめ");
        return true;
      });
  absl::StatusOr<AjimeeFrozenCorpus> empty_candidates = FreezeAjimeeCorpus(
      input, freezer_config.input_corpus_sha256(), freezer_config, request,
      config, empty_candidates_converter);
  EXPECT_EQ(empty_candidates.status().code(),
            absl::StatusCode::kFailedPrecondition);

  StrictMockConverter failed_conversion_converter;
  EXPECT_CALL(failed_conversion_converter, ResetConversion(_));
  EXPECT_CALL(failed_conversion_converter, StartConversion(_, _))
      .WillOnce(Return(false));
  absl::StatusOr<AjimeeFrozenCorpus> failed_conversion = FreezeAjimeeCorpus(
      input, freezer_config.input_corpus_sha256(), freezer_config, request,
      config, failed_conversion_converter);
  EXPECT_EQ(failed_conversion.status().code(), absl::StatusCode::kUnknown);
}

TEST(AjimeeArtifactUtilTest, UsesExactSha256AndRejectsPartialMessages) {
  EXPECT_EQ(Sha256Bytes(""),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  AjimeeFrozenCorpus incomplete;
  EXPECT_EQ(SerializeDeterministically(incomplete).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ReviewTextproto(incomplete).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeFreezerTest, DefaultDesktopIdentityIsStable) {
  EXPECT_EQ(MessageSha256(MakeAjimeeDefaultDesktopRequest()),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(MessageSha256(MakeAjimeeDefaultDesktopConfig()),
            "376212704866775205cd89f831ca3504"
            "300eb5a2762609fe07081db84a5a6e7b");
}

TEST(AjimeeFreezerTest, CheckedProductionConfigPinsAllAvailableIdentity) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation", "ajimee_freezer_config.textproto"});
  absl::StatusOr<std::string> contents = FileUtil::GetContents(path);
  ASSERT_TRUE(contents.ok()) << contents.status();
  AjimeeFreezerConfig freezer_config;
  ASSERT_TRUE(ParseTextproto(*contents, &freezer_config).ok());
  EXPECT_TRUE(ValidateAjimeeFreezerConfig(freezer_config).ok());
  EXPECT_EQ(freezer_config.schema_version(), 2);
  EXPECT_EQ(freezer_config.frozen_corpus_schema_version(), 2);
  EXPECT_EQ(freezer_config.input_corpus_sha256(),
            "55fc26b4d86c5d30c2eae5f40fcf1cd"
            "9f9b4822ad5a8bae424cdc9ed10e82848");
  EXPECT_EQ(freezer_config.expected_case_count(), 200);
  EXPECT_EQ(freezer_config.expected_source().source_revision(),
            "401666cd56d1a570c2021798b64b6da4396bfd45");
  EXPECT_EQ(freezer_config.expected_source().source_sha256(),
            "e9eb668fd6aa14b1e26436f429b55501"
            "08af0a1dfd443b8cea0bcb3ab3028fca");
  EXPECT_EQ(freezer_config.mozc_source_revision(),
            "851c3fe33060d2a6090363e4d7ec44fafde2c03d");
  EXPECT_EQ(freezer_config.mozc_data_type(), "oss");
  EXPECT_EQ(freezer_config.mozc_data_sha256(),
            "fbb9a23419d677a33968145cfe1979da"
            "c6e7d3c28712d469854b4707a5965ed1");
  EXPECT_EQ(freezer_config.default_desktop_request_sha256(),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(freezer_config.default_desktop_config_sha256(),
            "376212704866775205cd89f831ca3504"
            "300eb5a2762609fe07081db84a5a6e7b");
  EXPECT_EQ(freezer_config.evaluation_clock_utc_rfc3339(),
            "2000-01-01T00:00:00Z");
}

}  // namespace
}  // namespace mozc::engine::evaluation
