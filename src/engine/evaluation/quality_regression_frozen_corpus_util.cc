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

#include "engine/evaluation/quality_regression_frozen_corpus_util.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/strings/unicode.h"
#include "engine/evaluation/evaluation_clock_util.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"

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

absl::Status ValidateMozcIdentity(const EvaluationMozcIdentity& mozc) {
  if (!mozc.IsInitialized() || HasUnknownFieldsRecursively(mozc) ||
      !IsQualityRegressionLowercaseHex(mozc.source_revision(), 40) ||
      mozc.data_type().empty() || !strings::IsValidUtf8(mozc.data_type()) ||
      !IsQualityRegressionLowercaseHex(mozc.data_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(
          mozc.default_desktop_request_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(
          mozc.default_desktop_config_sha256(), 64) ||
      !ParseCanonicalEvaluationClockUtc(
           mozc.evaluation_clock_utc_rfc3339())
           .ok()) {
    return absl::InvalidArgumentError(
        "quality-regression Mozc identity is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateFrozenRequest(
    const QualityRegressionFrozenCase& frozen_case, uint64_t case_ordinal) {
  const FrozenCandidateRankerRequest& request = frozen_case.request();
  absl::Status status =
      ValidateFrozenCandidateRankerRequestStructure(request);
  if (!status.ok()) {
    return status;
  }
  if (request.mode() != FrozenCandidateRankerRequest::MODE_CONVERSION ||
      !request.preceding_text().empty() || !request.following_text().empty() ||
      request.reading() != frozen_case.conversion_reading() ||
      request.focused_segment_id() != 0 ||
      request.token().session_generation() != 0 ||
      request.token().state_revision() != 0 ||
      request.token().request_sequence() != case_ordinal + 1) {
    return absl::InvalidArgumentError(
        "quality-regression frozen request identity is invalid");
  }

  std::string expected_baseline;
  for (const FrozenCandidateRankerSegment& segment : request.segments()) {
    if (segment.key().empty()) {
      return absl::InvalidArgumentError(
          "quality-regression frozen segment is invalid");
    }
    for (const FrozenCandidateRankerCandidate& candidate :
         segment.candidates()) {
      if (candidate.key().empty() || candidate.value().empty()) {
        return absl::InvalidArgumentError(
            "quality-regression frozen candidate is invalid");
      }
    }
    expected_baseline.append(segment.candidates(0).value());
  }
  if (frozen_case.mozc_baseline_output() != expected_baseline) {
    return absl::InvalidArgumentError(
        "quality-regression baseline does not match candidate zero values");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateQualityRegressionFreezerConfig(
    const QualityRegressionFreezerConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "quality-regression freezer config is incomplete or has unknown "
        "fields");
  }
  if (config.schema_version() !=
          kQualityRegressionFreezerConfigSchemaVersion ||
      config.input_corpus_schema_version() !=
          kQualityRegressionCorpusSchemaVersion ||
      config.frozen_corpus_schema_version() !=
          kQualityRegressionFrozenCorpusSchemaVersion) {
    return absl::InvalidArgumentError(
        "quality-regression freezer config schema is invalid");
  }
  absl::Status status = ValidateQualityRegressionCorpusIdentity(
      config.expected_corpus_identity());
  if (!status.ok()) {
    return status;
  }
  status = ValidateMozcIdentity(config.mozc());
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(config.input_corpus_sha256(), 64) ||
      config.expected_case_count() == 0) {
    return absl::InvalidArgumentError(
        "quality-regression freezer config input identity is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionFreezerInput(
    const QualityRegressionInputCorpus& input_corpus,
    absl::string_view input_corpus_sha256,
    const QualityRegressionFreezerConfig& config) {
  absl::Status status = ValidateQualityRegressionFreezerConfig(config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionInputCorpus(input_corpus);
  if (!status.ok()) {
    return status;
  }
  if (input_corpus.schema_version() != config.input_corpus_schema_version()) {
    return absl::InvalidArgumentError(
        "quality-regression input corpus schema does not match freezer config");
  }
  status = RequireSameMessage(
      input_corpus.identity(), config.expected_corpus_identity(),
      "quality-regression input identity does not match freezer config");
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(input_corpus_sha256, 64) ||
      input_corpus_sha256 != config.input_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression input SHA256 does not match freezer config");
  }
  if (input_corpus.cases_size() != config.expected_case_count()) {
    return absl::InvalidArgumentError(
        "quality-regression input case count does not match freezer config");
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionFrozenCorpusStructure(
    const QualityRegressionFrozenCorpus& corpus) {
  if (!corpus.IsInitialized() || HasUnknownFieldsRecursively(corpus)) {
    return absl::InvalidArgumentError(
        "quality-regression frozen corpus is incomplete or has unknown "
        "fields");
  }
  if (corpus.schema_version() != kQualityRegressionFrozenCorpusSchemaVersion) {
    return absl::InvalidArgumentError(
        "quality-regression frozen corpus schema is invalid");
  }
  absl::Status status =
      ValidateQualityRegressionCorpusIdentity(corpus.identity());
  if (!status.ok()) {
    return status;
  }
  status = ValidateMozcIdentity(corpus.mozc());
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(corpus.input_corpus_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(corpus.freezer_config_sha256(), 64) ||
      corpus.cases().empty()) {
    return absl::InvalidArgumentError(
        "quality-regression frozen artifact identity is invalid");
  }

  uint64_t previous_source_line = 0;
  for (int case_index = 0; case_index < corpus.cases_size(); ++case_index) {
    const QualityRegressionFrozenCase& frozen_case = corpus.cases(case_index);
    if (frozen_case.source_line() <= previous_source_line ||
        frozen_case.conversion_reading().empty() ||
        frozen_case.mozc_baseline_output().empty() ||
        !strings::IsValidUtf8(frozen_case.conversion_reading()) ||
        !strings::IsValidUtf8(frozen_case.mozc_baseline_output())) {
      return absl::InvalidArgumentError(
          "quality-regression frozen cases are invalid or unordered");
    }
    previous_source_line = frozen_case.source_line();
    status = ValidateFrozenRequest(frozen_case,
                                   static_cast<uint64_t>(case_index));
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionFrozenCorpus(
    const QualityRegressionFrozenCorpus& corpus,
    const QualityRegressionFreezerConfig& config,
    absl::string_view freezer_config_sha256) {
  absl::Status status = ValidateQualityRegressionFreezerConfig(config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  if (corpus.schema_version() != config.frozen_corpus_schema_version()) {
    return absl::InvalidArgumentError(
        "quality-regression frozen corpus schema does not match config");
  }
  status = RequireSameMessage(
      corpus.identity(), config.expected_corpus_identity(),
      "quality-regression frozen identity does not match freezer config");
  if (!status.ok()) {
    return status;
  }
  status = RequireSameMessage(
      corpus.mozc(), config.mozc(),
      "quality-regression Mozc identity does not match freezer config");
  if (!status.ok()) {
    return status;
  }
  if (corpus.input_corpus_sha256() != config.input_corpus_sha256() ||
      corpus.cases_size() != config.expected_case_count() ||
      !IsQualityRegressionLowercaseHex(freezer_config_sha256, 64) ||
      corpus.freezer_config_sha256() != freezer_config_sha256) {
    return absl::FailedPreconditionError(
        "quality-regression frozen artifact identity does not match config");
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
