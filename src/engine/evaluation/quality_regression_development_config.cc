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

#include "engine/evaluation/quality_regression_development_config.h"

#include <array>
#include <cstddef>
#include <set>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "base/strings/unicode.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"

namespace mozc::engine::evaluation {
namespace {

using RecordLayout =
    ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::RecordLayout;

constexpr std::array<RecordLayout, 3> kObjectives = {
    ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
        STRUCTURED_WITH_TARGET,
    ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
        STRUCTURED_WITHOUT_TARGET,
    ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
        NATURAL_TEXT_ONLY,
};

bool IsSimpleUtf8FileName(const std::string& value) {
  return !value.empty() && value != "." && value != ".." &&
         strings::IsValidUtf8(value) &&
         value.find('/') == std::string::npos &&
         value.find('\\') == std::string::npos &&
         value.find(':') == std::string::npos &&
         value.find('\0') == std::string::npos;
}

}  // namespace

absl::Status ValidateQualityRegressionDevelopmentExecutionConfig(
    const DevelopmentExecutionConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "quality-regression development config is incomplete or has unknown "
        "fields");
  }
  if (config.schema_version() !=
          kQualityRegressionDevelopmentExecutionConfigSchemaVersion ||
      config.frozen_corpus_schema_version() !=
          kQualityRegressionFrozenCorpusSchemaVersion ||
      config.native_coverage_schema_version() !=
          kQualityRegressionDevelopmentNativeCoverageSchemaVersion ||
      config.model_manifest_schema_version() !=
          kQualityRegressionDevelopmentModelManifestSchemaVersion ||
      config.coverage_definition_version() !=
          kQualityRegressionDevelopmentCoverageDefinitionVersion ||
      config.permutation_definition_version() !=
          kQualityRegressionDevelopmentPermutationDefinitionVersion) {
    return absl::InvalidArgumentError(
        "quality-regression development config schema or definition is "
        "invalid");
  }
  if (!IsQualityRegressionLowercaseHex(config.frozen_corpus_sha256(), 64) ||
      config.expected_case_count() == 0 ||
      config.objectives_size() != kObjectives.size()) {
    return absl::InvalidArgumentError(
        "quality-regression development config identity or objective count "
        "is invalid");
  }

  std::set<std::string> manifest_filenames;
  std::set<std::string> manifest_hashes;
  for (int objective_index = 0; objective_index < config.objectives_size();
       ++objective_index) {
    const DevelopmentObjectiveSpec& objective =
        config.objectives(objective_index);
    if (objective.objective() != kObjectives[objective_index] ||
        !IsSimpleUtf8FileName(objective.manifest_filename()) ||
        !IsQualityRegressionLowercaseHex(objective.manifest_sha256(), 64) ||
        !manifest_filenames.insert(objective.manifest_filename()).second ||
        !manifest_hashes.insert(objective.manifest_sha256()).second) {
      return absl::InvalidArgumentError(
          "quality-regression development objective identity or order is "
          "invalid");
    }
    const bool expected_eligibility =
        objective.objective() !=
        ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
            STRUCTURED_WITH_TARGET;
    if (objective.selection_eligible() != expected_eligibility) {
      return absl::InvalidArgumentError(
          "quality-regression development objective eligibility is invalid");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionDevelopmentManifests(
    const DevelopmentExecutionConfig& config,
    absl::Span<const ::mozc::engine::CandidateRankerModelManifest> manifests,
    absl::Span<const std::string> manifest_sha256s) {
  absl::Status status =
      ValidateQualityRegressionDevelopmentExecutionConfig(config);
  if (!status.ok()) {
    return status;
  }
  if (manifests.size() != config.objectives_size() ||
      manifest_sha256s.size() != manifests.size()) {
    return absl::InvalidArgumentError(
        "quality-regression development manifest count mismatch");
  }

  std::string reference_bytes;
  for (size_t manifest_index = 0; manifest_index < manifests.size();
       ++manifest_index) {
    const ::mozc::engine::CandidateRankerModelManifest& manifest =
        manifests[manifest_index];
    if (!manifest.IsInitialized() || HasUnknownFieldsRecursively(manifest)) {
      return absl::InvalidArgumentError(
          "quality-regression development manifest is incomplete or has "
          "unknown fields");
    }
    status = ::mozc::engine::ValidateCandidateRankerModelManifest(manifest);
    if (!status.ok()) {
      return status;
    }
    if (manifest_sha256s[manifest_index] !=
            config.objectives(manifest_index).manifest_sha256() ||
        manifest.schema_version() != config.model_manifest_schema_version() ||
        manifest.scoring_template().record_layout() !=
            config.objectives(manifest_index).objective()) {
      return absl::FailedPreconditionError(
          "quality-regression development manifest identity mismatch");
    }

    ::mozc::engine::CandidateRankerModelManifest canonical = manifest;
    canonical.mutable_scoring_template()->set_record_layout(
        ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
            STRUCTURED_WITH_TARGET);
    absl::StatusOr<std::string> canonical_bytes =
        SerializeDeterministically(canonical);
    if (!canonical_bytes.ok()) {
      return canonical_bytes.status();
    }
    if (manifest_index == 0) {
      reference_bytes = std::move(*canonical_bytes);
    } else if (*canonical_bytes != reference_bytes) {
      return absl::FailedPreconditionError(
          "quality-regression development manifests differ outside record "
          "layout");
    }
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
