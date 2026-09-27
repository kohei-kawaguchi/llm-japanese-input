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

#include "engine/evaluation/quality_regression_baseline_report.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/file_util.h"
#include "base/protobuf/descriptor.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/exact_quality_metrics.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/quality_regression_baseline_report.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kFrozenArtifactSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kAnswerArtifactSha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kReportConfigSha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kInputArtifactSha256[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr char kFreezerConfigSha256[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr char kSourceRevision[] =
    "ffffffffffffffffffffffffffffffffffffffff";
constexpr char kPrivateReading[] =
    "PRIVATE_BASELINE_READING_MUST_NOT_ENTER_THE_REPORT";
constexpr char kPrivateCorrectOutput[] =
    "PRIVATE_CORRECT_OUTPUT_MUST_NOT_ENTER_THE_REPORT";
constexpr char kPrivateBaselineOutput[] =
    "PRIVATE_BASELINE_OUTPUT_MUST_NOT_ENTER_THE_REPORT";
constexpr char kPrivateAnswerOutput[] =
    "PRIVATE_ANSWER_OUTPUT_MUST_NOT_ENTER_THE_REPORT";

template <typename Message>
void AddUnknownField(Message* message) {
  std::string bytes;
  ASSERT_TRUE(message->SerializeToString(&bytes));
  bytes.append("\xa0\x06\x01", 3);
  ASSERT_TRUE(message->ParseFromString(bytes));
}

void FillCorpusIdentity(QualityRegressionCorpusIdentity* identity) {
  identity->set_role(DEVELOPMENT);
  EvaluationSourceIdentity* source = identity->mutable_source();
  source->set_benchmark_name("Synthetic quality regression development");
  source->set_source_revision(kSourceRevision);
  source->set_source_relative_path("synthetic/development.tsv");
  source->set_source_sha256(kFrozenArtifactSha256);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic filtered development corpus.");
  identity->set_parser_definition_version(1);
  identity->set_normalization_definition_version(1);
  identity->set_import_config_sha256(kAnswerArtifactSha256);
}

void FillMozcIdentity(EvaluationMozcIdentity* identity) {
  identity->set_source_revision(kSourceRevision);
  identity->set_data_type("oss");
  identity->set_data_sha256(kFrozenArtifactSha256);
  identity->set_default_desktop_request_sha256(kAnswerArtifactSha256);
  identity->set_default_desktop_config_sha256(kReportConfigSha256);
  identity->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");
}

void AddFrozenCase(uint64_t source_line, uint64_t request_sequence,
                   const std::string& baseline,
                   QualityRegressionFrozenCorpus* corpus) {
  QualityRegressionFrozenCase* frozen_case = corpus->add_cases();
  frozen_case->set_source_line(source_line);
  frozen_case->set_conversion_reading(kPrivateReading);
  frozen_case->set_mozc_baseline_output(baseline);
  FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
  request->mutable_token()->set_session_generation(0);
  request->mutable_token()->set_state_revision(0);
  request->mutable_token()->set_request_sequence(request_sequence);
  request->set_mode(FrozenCandidateRankerRequest::MODE_CONVERSION);
  request->set_preceding_text("");
  request->set_following_text("");
  request->set_reading(kPrivateReading);
  request->set_focused_segment_id(0);
  FrozenCandidateRankerSegment* segment = request->add_segments();
  segment->set_id(0);
  segment->set_key("synthetic-key");
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(0);
  candidate->set_key("synthetic-key");
  candidate->set_value(baseline);
  candidate->set_cost(123);
  candidate->set_attributes(0);
  candidate->set_consumed_key_size(3);
  candidate->set_is_protected(false);
}

QualityRegressionFrozenCorpus MakeFrozenCorpus() {
  QualityRegressionFrozenCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  corpus.set_input_corpus_sha256(kInputArtifactSha256);
  FillMozcIdentity(corpus.mutable_mozc());
  AddFrozenCase(2, 1, kPrivateCorrectOutput, &corpus);
  AddFrozenCase(7, 2, kPrivateBaselineOutput, &corpus);
  corpus.set_freezer_config_sha256(kFreezerConfigSha256);
  return corpus;
}

QualityRegressionAnswerCorpus MakeAnswerCorpus() {
  QualityRegressionAnswerCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  QualityRegressionAnswerCase* first = corpus.add_cases();
  first->set_source_line(2);
  first->set_normalized_whole_output(kPrivateCorrectOutput);
  QualityRegressionAnswerCase* second = corpus.add_cases();
  second->set_source_line(7);
  second->set_normalized_whole_output(kPrivateAnswerOutput);
  return corpus;
}

QualityRegressionDevelopmentBaselineReportConfig MakeConfig() {
  QualityRegressionDevelopmentBaselineReportConfig config;
  config.set_schema_version(1);
  config.set_frozen_corpus_schema_version(1);
  config.set_answer_corpus_schema_version(1);
  config.set_report_schema_version(1);
  FillCorpusIdentity(config.mutable_expected_corpus_identity());
  config.set_frozen_corpus_sha256(kFrozenArtifactSha256);
  config.set_answer_corpus_sha256(kAnswerArtifactSha256);
  config.set_expected_case_count(2);
  config.set_expected_correct_count(1);
  return config;
}

absl::Status ValidateReport(
    const QualityRegressionDevelopmentBaselineReport& report,
    const QualityRegressionFrozenCorpus& frozen,
    const QualityRegressionAnswerCorpus& answers,
    const QualityRegressionDevelopmentBaselineReportConfig& config) {
  return ValidateQualityRegressionDevelopmentBaselineReport(
      report, frozen, kFrozenArtifactSha256, answers, kAnswerArtifactSha256,
      config, kReportConfigSha256);
}

TEST(QualityRegressionBaselineReportTest,
     BuildsDeterministicAggregateOnlyReport) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  const QualityRegressionDevelopmentBaselineReportConfig config = MakeConfig();
  absl::StatusOr<QualityRegressionDevelopmentBaselineReport> report =
      BuildQualityRegressionDevelopmentBaselineReport(
          frozen, kFrozenArtifactSha256, answers, kAnswerArtifactSha256,
          config, kReportConfigSha256);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_EQ(report->schema_version(), 1);
  EXPECT_EQ(report->case_count(), 2);
  EXPECT_EQ(report->correct_count(), 1);
  EXPECT_EQ(report->top_output_accuracy().numerator(), 1);
  EXPECT_EQ(report->top_output_accuracy().denominator(), 2);
  EXPECT_EQ(report->GetDescriptor()->field_count(), 5);
  EXPECT_EQ(report->identity().GetDescriptor()->field_count(), 4);
  EXPECT_TRUE(ValidateReport(*report, frozen, answers, config).ok());

  absl::StatusOr<std::string> first = SerializeDeterministically(*report);
  absl::StatusOr<std::string> second = SerializeDeterministically(*report);
  absl::StatusOr<std::string> review = ReviewTextproto(*report);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(review.ok()) << review.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(Sha256Bytes(*first),
            "66a8ef6392324438e7d82fe54b58d5fe"
            "192513776c1b00a69110af35d8f757f1");
  for (const std::string private_value :
       {kPrivateReading, kPrivateCorrectOutput, kPrivateBaselineOutput,
        kPrivateAnswerOutput}) {
    EXPECT_EQ(first->find(private_value), std::string::npos);
    EXPECT_EQ(review->find(private_value), std::string::npos);
  }
}

TEST(QualityRegressionBaselineReportTest,
     RejectsHashIdentityOrderAndExactCountMismatches) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  QualityRegressionDevelopmentBaselineReportConfig config = MakeConfig();
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kAnswerArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kFailedPrecondition);

  answers.mutable_cases(1)->set_source_line(8);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  answers = MakeAnswerCorpus();
  config.set_expected_correct_count(2);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kFailedPrecondition);

  config = MakeConfig();
  config.mutable_expected_corpus_identity()->mutable_source()->set_creator(
      "Different creator");
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(QualityRegressionBaselineReportTest, ComparesOutputsByExactUtf8Bytes) {
  QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  QualityRegressionDevelopmentBaselineReportConfig config = MakeConfig();
  frozen.mutable_cases(0)->set_mozc_baseline_output("\xc3\xa9");
  frozen.mutable_cases(0)
      ->mutable_request()
      ->mutable_segments(0)
      ->mutable_candidates(0)
      ->set_value("\xc3\xa9");
  answers.mutable_cases(0)->set_normalized_whole_output("e\xcc\x81");
  config.set_expected_correct_count(0);
  absl::StatusOr<QualityRegressionDevelopmentBaselineReport> report =
      BuildQualityRegressionDevelopmentBaselineReport(
          frozen, kFrozenArtifactSha256, answers, kAnswerArtifactSha256,
          config, kReportConfigSha256);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_EQ(report->correct_count(), 0);
}

TEST(QualityRegressionBaselineReportTest,
     RejectsRecursiveUnknownFieldsAndNonDevelopmentConfig) {
  QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  QualityRegressionDevelopmentBaselineReportConfig config = MakeConfig();

  QualityRegressionDevelopmentBaselineReportConfig unknown_config = config;
  AddUnknownField(&unknown_config);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineReportConfig(
                unknown_config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  unknown_config = config;
  AddUnknownField(unknown_config.mutable_expected_corpus_identity());
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineReportConfig(
                unknown_config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  config.mutable_expected_corpus_identity()->set_role(HOLDOUT);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineReportConfig(config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  config = MakeConfig();
  AddUnknownField(&frozen);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  frozen = MakeFrozenCorpus();
  AddUnknownField(&answers);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  answers = MakeAnswerCorpus();
  AddUnknownField(answers.mutable_cases(0));
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kInvalidArgument);

  answers = MakeAnswerCorpus();
  AddUnknownField(frozen.mutable_cases(0)
                      ->mutable_request()
                      ->mutable_segments(0)
                      ->mutable_candidates(0));
  EXPECT_EQ(ValidateQualityRegressionDevelopmentBaselineInputs(
                frozen, kFrozenArtifactSha256, answers,
                kAnswerArtifactSha256, config)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionBaselineReportTest,
     RejectsUnknownOrInconsistentAggregateReport) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  const QualityRegressionDevelopmentBaselineReportConfig config = MakeConfig();
  absl::StatusOr<QualityRegressionDevelopmentBaselineReport> built =
      BuildQualityRegressionDevelopmentBaselineReport(
          frozen, kFrozenArtifactSha256, answers, kAnswerArtifactSha256,
          config, kReportConfigSha256);
  ASSERT_TRUE(built.ok()) << built.status();

  QualityRegressionDevelopmentBaselineReport report = *built;
  report.set_correct_count(0);
  EXPECT_EQ(ValidateReport(report, frozen, answers, config).code(),
            absl::StatusCode::kInvalidArgument);

  report = *built;
  AddUnknownField(&report);
  EXPECT_EQ(ValidateReport(report, frozen, answers, config).code(),
            absl::StatusCode::kInvalidArgument);

  report = *built;
  AddUnknownField(report.mutable_identity());
  EXPECT_EQ(ValidateReport(report, frozen, answers, config).code(),
            absl::StatusCode::kInvalidArgument);

  report = *built;
  AddUnknownField(report.mutable_top_output_accuracy());
  EXPECT_EQ(ValidateReport(report, frozen, answers, config).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionBaselineReportTest, CheckedDevelopmentConfigIsPinned) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "quality_regression_development_baseline_report_config.textproto"});
  absl::StatusOr<std::string> bytes = FileUtil::GetContents(path);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->size(), 1495);
  EXPECT_EQ(Sha256Bytes(*bytes),
            "9a5058100568bf357f56e9f75b87292f"
            "65155ac2d3faab92dc61985711ee944f");
  QualityRegressionDevelopmentBaselineReportConfig config;
  ASSERT_TRUE(ParseTextproto(*bytes, &config).ok());
  EXPECT_TRUE(
      ValidateQualityRegressionDevelopmentBaselineReportConfig(config).ok());
  EXPECT_EQ(config.expected_corpus_identity().role(), DEVELOPMENT);
  EXPECT_EQ(config.frozen_corpus_sha256(),
            "44c9a08d748f2cef5c8428b1101f490c"
            "2719db34e533ba1cda3cee9b12b41c59");
  EXPECT_EQ(config.answer_corpus_sha256(),
            "8d76595da3ccaa609903fc48a84e1ef47"
            "ed179c3936ebad06e6beaf242bd0d79");
  EXPECT_EQ(config.expected_case_count(), 219);
  EXPECT_EQ(config.expected_correct_count(), 164);
}

}  // namespace
}  // namespace mozc::engine::evaluation
