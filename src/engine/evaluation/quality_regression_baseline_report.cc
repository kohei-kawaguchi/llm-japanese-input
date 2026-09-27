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
#include "absl/strings/string_view.h"
#include "base/protobuf/message.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/exact_quality_metrics.pb.h"
#include "engine/evaluation/quality_regression_baseline_report.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"

namespace mozc::engine::evaluation {
namespace {

absl::Status RequireSameMessage(const protobuf::Message& actual,
                                const protobuf::Message& expected,
                                absl::string_view description) {
  absl::StatusOr<std::string> actual_bytes = SerializeDeterministically(actual);
  if (!actual_bytes.ok()) {
    return actual_bytes.status();
  }
  absl::StatusOr<std::string> expected_bytes =
      SerializeDeterministically(expected);
  if (!expected_bytes.ok()) {
    return expected_bytes.status();
  }
  if (*actual_bytes != *expected_bytes) {
    return absl::FailedPreconditionError(description);
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> ValidateInputsAndCountCorrect(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const QualityRegressionAnswerCorpus& answer_corpus,
    absl::string_view answer_corpus_sha256,
    const QualityRegressionDevelopmentBaselineReportConfig& config) {
  absl::Status status =
      ValidateQualityRegressionDevelopmentBaselineReportConfig(config);
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(frozen_corpus_sha256, 64) ||
      frozen_corpus_sha256 != config.frozen_corpus_sha256() ||
      !IsQualityRegressionLowercaseHex(answer_corpus_sha256, 64) ||
      answer_corpus_sha256 != config.answer_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression baseline input SHA256 mismatch");
  }

  status = ValidateQualityRegressionFrozenCorpusStructure(frozen_corpus);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionAnswerCorpus(answer_corpus);
  if (!status.ok()) {
    return status;
  }
  if (frozen_corpus.schema_version() !=
          config.frozen_corpus_schema_version() ||
      answer_corpus.schema_version() != config.answer_corpus_schema_version()) {
    return absl::InvalidArgumentError(
        "quality-regression baseline input schema mismatch");
  }
  status = RequireSameMessage(
      frozen_corpus.identity(), config.expected_corpus_identity(),
      "quality-regression baseline frozen identity mismatch");
  if (!status.ok()) {
    return status;
  }
  status = RequireSameMessage(
      answer_corpus.identity(), config.expected_corpus_identity(),
      "quality-regression baseline answer identity mismatch");
  if (!status.ok()) {
    return status;
  }
  if (frozen_corpus.cases_size() != config.expected_case_count() ||
      answer_corpus.cases_size() != config.expected_case_count()) {
    return absl::InvalidArgumentError(
        "quality-regression baseline case count mismatch");
  }

  uint64_t correct_count = 0;
  for (int case_index = 0; case_index < frozen_corpus.cases_size();
       ++case_index) {
    const QualityRegressionFrozenCase& frozen_case =
        frozen_corpus.cases(case_index);
    const QualityRegressionAnswerCase& answer_case =
        answer_corpus.cases(case_index);
    if (frozen_case.source_line() != answer_case.source_line()) {
      return absl::InvalidArgumentError(
          "quality-regression baseline source-line order mismatch");
    }
    if (frozen_case.mozc_baseline_output() ==
        answer_case.normalized_whole_output()) {
      ++correct_count;
    }
  }
  if (correct_count != config.expected_correct_count()) {
    return absl::FailedPreconditionError(
        "quality-regression baseline correct count mismatch");
  }
  return correct_count;
}

}  // namespace

absl::Status ValidateQualityRegressionDevelopmentBaselineReportConfig(
    const QualityRegressionDevelopmentBaselineReportConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "quality-regression baseline config is incomplete or has unknown "
        "fields");
  }
  if (config.schema_version() !=
          kQualityRegressionDevelopmentBaselineReportConfigSchemaVersion ||
      config.frozen_corpus_schema_version() !=
          kQualityRegressionFrozenCorpusSchemaVersion ||
      config.answer_corpus_schema_version() !=
          kQualityRegressionCorpusSchemaVersion ||
      config.report_schema_version() !=
          kQualityRegressionDevelopmentBaselineReportSchemaVersion) {
    return absl::InvalidArgumentError(
        "quality-regression baseline config schema is invalid");
  }
  absl::Status status = ValidateQualityRegressionCorpusIdentity(
      config.expected_corpus_identity());
  if (!status.ok()) {
    return status;
  }
  if (config.expected_corpus_identity().role() != DEVELOPMENT) {
    return absl::InvalidArgumentError(
        "quality-regression baseline config must be development-only");
  }
  if (!IsQualityRegressionLowercaseHex(config.frozen_corpus_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(config.answer_corpus_sha256(), 64) ||
      config.expected_case_count() == 0 ||
      config.expected_correct_count() > config.expected_case_count()) {
    return absl::InvalidArgumentError(
        "quality-regression baseline config identity or count is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionDevelopmentBaselineInputs(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const QualityRegressionAnswerCorpus& answer_corpus,
    absl::string_view answer_corpus_sha256,
    const QualityRegressionDevelopmentBaselineReportConfig& config) {
  absl::StatusOr<uint64_t> correct_count = ValidateInputsAndCountCorrect(
      frozen_corpus, frozen_corpus_sha256, answer_corpus,
      answer_corpus_sha256, config);
  return correct_count.status();
}

absl::StatusOr<QualityRegressionDevelopmentBaselineReport>
BuildQualityRegressionDevelopmentBaselineReport(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const QualityRegressionAnswerCorpus& answer_corpus,
    absl::string_view answer_corpus_sha256,
    const QualityRegressionDevelopmentBaselineReportConfig& config,
    absl::string_view report_config_sha256) {
  absl::StatusOr<uint64_t> correct_count = ValidateInputsAndCountCorrect(
      frozen_corpus, frozen_corpus_sha256, answer_corpus,
      answer_corpus_sha256, config);
  if (!correct_count.ok()) {
    return correct_count.status();
  }
  if (!IsQualityRegressionLowercaseHex(report_config_sha256, 64)) {
    return absl::InvalidArgumentError(
        "quality-regression baseline config SHA256 is invalid");
  }

  QualityRegressionDevelopmentBaselineReport report;
  report.set_schema_version(
      kQualityRegressionDevelopmentBaselineReportSchemaVersion);
  QualityRegressionDevelopmentBaselineReportIdentity* identity =
      report.mutable_identity();
  identity->set_report_config_sha256(report_config_sha256);
  identity->set_frozen_corpus_sha256(frozen_corpus_sha256);
  identity->set_answer_corpus_sha256(answer_corpus_sha256);
  *identity->mutable_corpus_identity() = config.expected_corpus_identity();
  report.set_case_count(config.expected_case_count());
  report.set_correct_count(*correct_count);
  report.mutable_top_output_accuracy()->set_numerator(*correct_count);
  report.mutable_top_output_accuracy()->set_denominator(
      config.expected_case_count());

  absl::Status status = ValidateQualityRegressionDevelopmentBaselineReport(
      report, frozen_corpus, frozen_corpus_sha256, answer_corpus,
      answer_corpus_sha256, config, report_config_sha256);
  if (!status.ok()) {
    return status;
  }
  return report;
}

absl::Status ValidateQualityRegressionDevelopmentBaselineReport(
    const QualityRegressionDevelopmentBaselineReport& report,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const QualityRegressionAnswerCorpus& answer_corpus,
    absl::string_view answer_corpus_sha256,
    const QualityRegressionDevelopmentBaselineReportConfig& config,
    absl::string_view report_config_sha256) {
  absl::StatusOr<uint64_t> correct_count = ValidateInputsAndCountCorrect(
      frozen_corpus, frozen_corpus_sha256, answer_corpus,
      answer_corpus_sha256, config);
  if (!correct_count.ok()) {
    return correct_count.status();
  }
  if (!IsQualityRegressionLowercaseHex(report_config_sha256, 64) ||
      !report.IsInitialized() || HasUnknownFieldsRecursively(report)) {
    return absl::InvalidArgumentError(
        "quality-regression baseline report is incomplete or has unknown "
        "fields");
  }
  if (report.schema_version() !=
          kQualityRegressionDevelopmentBaselineReportSchemaVersion ||
      report.schema_version() != config.report_schema_version()) {
    return absl::InvalidArgumentError(
        "quality-regression baseline report schema is invalid");
  }
  const QualityRegressionDevelopmentBaselineReportIdentity& identity =
      report.identity();
  if (identity.report_config_sha256() != report_config_sha256 ||
      identity.frozen_corpus_sha256() != frozen_corpus_sha256 ||
      identity.answer_corpus_sha256() != answer_corpus_sha256) {
    return absl::FailedPreconditionError(
        "quality-regression baseline report hash identity mismatch");
  }
  absl::Status status = RequireSameMessage(
      identity.corpus_identity(), config.expected_corpus_identity(),
      "quality-regression baseline report corpus identity mismatch");
  if (!status.ok()) {
    return status;
  }
  if (report.case_count() != config.expected_case_count() ||
      report.correct_count() != *correct_count ||
      report.top_output_accuracy().numerator() !=
          static_cast<int64_t>(*correct_count) ||
      report.top_output_accuracy().denominator() != report.case_count()) {
    return absl::InvalidArgumentError(
        "quality-regression baseline report counts are inconsistent");
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
