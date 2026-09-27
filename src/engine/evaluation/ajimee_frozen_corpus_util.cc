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

#include "engine/evaluation/ajimee_frozen_corpus_util.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/evaluation_clock_util.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"

namespace mozc::engine::evaluation {

bool IsAjimeeLowercaseHex(absl::string_view value, size_t size) {
  if (value.size() != size) {
    return false;
  }
  for (const char ch : value) {
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
      return false;
    }
  }
  return true;
}

absl::Status ValidateAjimeeSourceIdentity(
    const EvaluationSourceIdentity& source) {
  if (!source.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE source identity is incomplete");
  }
  if (source.benchmark_name().empty() || source.source_revision().empty() ||
      source.source_relative_path().empty() || source.creator().empty() ||
      source.source_url().empty() || source.license_identifier().empty() ||
      source.license_url().empty() || source.upstream_dataset_url().empty() ||
      source.changes_notice().empty() ||
      !IsAjimeeLowercaseHex(source.source_sha256(), 64)) {
    return absl::InvalidArgumentError("AJIMEE source identity is invalid");
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Time> ParseAjimeeEvaluationClockUtc(
    absl::string_view value) {
  return ParseCanonicalEvaluationClockUtc(value);
}

absl::Status ValidateAjimeeFrozenCorpusStructure(
    const AjimeeFrozenCorpus& corpus) {
  if (!corpus.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE frozen corpus is incomplete");
  }
  if (corpus.schema_version() != kAjimeeFrozenCorpusSchemaVersion) {
    return absl::InvalidArgumentError("AJIMEE frozen corpus schema mismatch");
  }
  absl::Status status = ValidateAjimeeSourceIdentity(corpus.source());
  if (!status.ok()) {
    return status;
  }
  const EvaluationMozcIdentity& mozc = corpus.mozc();
  if (!IsAjimeeLowercaseHex(corpus.input_corpus_sha256(), 64) ||
      !IsAjimeeLowercaseHex(mozc.source_revision(), 40) ||
      mozc.data_type().empty() ||
      !IsAjimeeLowercaseHex(mozc.data_sha256(), 64) ||
      !IsAjimeeLowercaseHex(mozc.default_desktop_request_sha256(), 64) ||
      !IsAjimeeLowercaseHex(mozc.default_desktop_config_sha256(), 64)) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus identity is invalid");
  }
  if (!ParseAjimeeEvaluationClockUtc(mozc.evaluation_clock_utc_rfc3339())
           .ok()) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus evaluation clock is invalid");
  }
  if (corpus.cases().empty()) {
    return absl::InvalidArgumentError("AJIMEE frozen corpus has no cases");
  }

  uint64_t previous_source_index = 0;
  bool has_previous_source_index = false;
  for (int case_index = 0; case_index < corpus.cases_size(); ++case_index) {
    const AjimeeFrozenCase& frozen_case = corpus.cases(case_index);
    if (frozen_case.normalized_hiragana_reading().empty() ||
        (has_previous_source_index &&
         frozen_case.source_index() <= previous_source_index)) {
      return absl::InvalidArgumentError(
          "AJIMEE frozen cases are invalid or unordered");
    }
    has_previous_source_index = true;
    previous_source_index = frozen_case.source_index();
    const FrozenCandidateRankerRequest& request = frozen_case.request();
    status = ValidateFrozenCandidateRankerRequestStructure(request);
    if (!status.ok()) {
      return status;
    }
    if (request.mode() != FrozenCandidateRankerRequest::MODE_CONVERSION ||
        request.reading() != frozen_case.normalized_hiragana_reading() ||
        !request.following_text().empty() ||
        request.focused_segment_id() != 0 ||
        request.token().session_generation() != 0 ||
        request.token().state_revision() != 0 ||
        request.token().request_sequence() !=
            static_cast<uint64_t>(case_index) + 1) {
      return absl::InvalidArgumentError(
          "AJIMEE frozen ranker request is invalid");
    }

    std::string expected_baseline;
    for (int segment_index = 0; segment_index < request.segments_size();
         ++segment_index) {
      const FrozenCandidateRankerSegment& segment =
          request.segments(segment_index);
      expected_baseline.append(segment.candidates(0).value());
    }
    if (frozen_case.baseline_output() != expected_baseline) {
      return absl::InvalidArgumentError(
          "AJIMEE frozen baseline does not match candidate zero values");
    }
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
