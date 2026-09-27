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

#include "engine/evaluation/quality_regression_native_coverage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_development_config.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"

namespace mozc::engine::evaluation {
namespace {

using ::mozc::engine::CandidateRankerCapacityResult;
using ::mozc::engine::CandidateRankerSegmentCapacityDiagnostic;
using ::mozc::engine::CandidateRankerSegmentCapacityDisposition;

struct RankableSegment {
  size_t segment_index = 0;
  const converter::CandidateRankerSegment* segment = nullptr;
  std::vector<uint64_t> unprotected_candidate_ids;
};

std::vector<RankableSegment> GetRankableSegments(
    const converter::CandidateRankerRequest& request) {
  std::vector<RankableSegment> rankable_segments;
  for (size_t segment_index = 0; segment_index < request.segments.size();
       ++segment_index) {
    const converter::CandidateRankerSegment& segment =
        request.segments[segment_index];
    RankableSegment rankable{
        .segment_index = segment_index,
        .segment = &segment,
    };
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (!candidate.is_protected) {
        rankable.unprotected_candidate_ids.push_back(candidate.id);
      }
    }
    if (rankable.unprotected_candidate_ids.size() > 1) {
      rankable_segments.push_back(std::move(rankable));
    }
  }
  return rankable_segments;
}

absl::Status ValidateInputs(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentExecutionConfig& config,
    absl::string_view execution_config_sha256) {
  absl::Status status =
      ValidateQualityRegressionDevelopmentExecutionConfig(config);
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(frozen_corpus_sha256, 64) ||
      frozen_corpus_sha256 != config.frozen_corpus_sha256() ||
      !IsQualityRegressionLowercaseHex(execution_config_sha256, 64)) {
    return absl::FailedPreconditionError(
        "quality-regression native coverage input hash mismatch");
  }
  status = ValidateQualityRegressionFrozenCorpusStructure(frozen_corpus);
  if (!status.ok()) {
    return status;
  }
  if (frozen_corpus.schema_version() !=
          config.frozen_corpus_schema_version() ||
      frozen_corpus.identity().role() != DEVELOPMENT ||
      frozen_corpus.cases_size() != config.expected_case_count()) {
    return absl::FailedPreconditionError(
        "quality-regression native coverage frozen corpus mismatch");
  }
  return absl::OkStatus();
}

absl::Status CopyCapacityResult(
    const converter::CandidateRankerRequest& request,
    const CandidateRankerCapacityResult& capacity,
    NativeCoverageCase* coverage_case) {
  if (capacity.request_limit_violation.has_value()) {
    return absl::FailedPreconditionError(
        "quality-regression native coverage request exceeds a limit");
  }
  const std::vector<RankableSegment> rankable_segments =
      GetRankableSegments(request);
  if (capacity.segments.size() != rankable_segments.size()) {
    return absl::InvalidArgumentError(
        "quality-regression native coverage segment count mismatch");
  }

  for (size_t segment_index = 0; segment_index < rankable_segments.size();
       ++segment_index) {
    const RankableSegment& rankable = rankable_segments[segment_index];
    const CandidateRankerSegmentCapacityDiagnostic& diagnostic =
        capacity.segments[segment_index];
    const uint64_t unprotected_count =
        rankable.unprotected_candidate_ids.size();
    if (diagnostic.segment_id != rankable.segment->id ||
        diagnostic.unprotected_candidate_count != unprotected_count ||
        diagnostic.window_candidate_count < 2 ||
        diagnostic.window_candidate_count > unprotected_count ||
        diagnostic.omitted_by_window !=
            unprotected_count - diagnostic.window_candidate_count) {
      return absl::InvalidArgumentError(
          "quality-regression native coverage capacity diagnostic mismatch");
    }

    NativeCoverageSegment* coverage_segment = coverage_case->add_segments();
    coverage_segment->set_segment_id(rankable.segment->id);
    switch (diagnostic.disposition) {
      case CandidateRankerSegmentCapacityDisposition::kSelected:
        if (diagnostic.selected_candidate_count < 2 ||
            diagnostic.selected_candidate_count >
                diagnostic.window_candidate_count ||
            diagnostic.omitted_by_capacity !=
                diagnostic.window_candidate_count -
                    diagnostic.selected_candidate_count ||
            diagnostic.first_limiter.has_value() !=
                (diagnostic.omitted_by_capacity != 0)) {
          return absl::InvalidArgumentError(
              "quality-regression selected native coverage is invalid");
        }
        coverage_segment->set_disposition(
            DEVELOPMENT_NATIVE_COVERAGE_SELECTED);
        for (size_t candidate_index = 0;
             candidate_index < diagnostic.selected_candidate_count;
             ++candidate_index) {
          coverage_segment->add_selected_candidate_ids(
              rankable.unprotected_candidate_ids[candidate_index]);
        }
        break;
      case CandidateRankerSegmentCapacityDisposition::kOmittedCapacity:
        if (diagnostic.selected_candidate_count != 0 ||
            diagnostic.omitted_by_capacity !=
                diagnostic.window_candidate_count ||
            !diagnostic.first_limiter.has_value()) {
          return absl::InvalidArgumentError(
              "quality-regression omitted native coverage is invalid");
        }
        coverage_segment->set_disposition(
            DEVELOPMENT_NATIVE_COVERAGE_OMITTED_CAPACITY);
        break;
      default:
        return absl::InvalidArgumentError(
            "quality-regression native coverage disposition is invalid");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateCoverageMap(
    const NativeCoverageMap& map,
    const QualityRegressionFrozenCorpus& frozen_corpus) {
  if (map.cases_size() != frozen_corpus.cases_size()) {
    return absl::InvalidArgumentError(
        "quality-regression native coverage case count mismatch");
  }
  for (int case_index = 0; case_index < map.cases_size(); ++case_index) {
    const NativeCoverageCase& coverage_case = map.cases(case_index);
    const QualityRegressionFrozenCase& frozen_case =
        frozen_corpus.cases(case_index);
    if (coverage_case.source_line() != frozen_case.source_line()) {
      return absl::InvalidArgumentError(
          "quality-regression native coverage source-line order mismatch");
    }
    absl::StatusOr<converter::CandidateRankerRequest> request =
        ConvertFrozenCandidateRankerRequest(frozen_case.request());
    if (!request.ok()) {
      return request.status();
    }
    const std::vector<RankableSegment> rankable_segments =
        GetRankableSegments(*request);
    if (coverage_case.segments_size() != rankable_segments.size()) {
      return absl::InvalidArgumentError(
          "quality-regression native coverage segment count mismatch");
    }
    for (int segment_index = 0;
         segment_index < coverage_case.segments_size(); ++segment_index) {
      const NativeCoverageSegment& coverage_segment =
          coverage_case.segments(segment_index);
      const RankableSegment& rankable = rankable_segments[segment_index];
      if (coverage_segment.segment_id() != rankable.segment->id) {
        return absl::InvalidArgumentError(
            "quality-regression native coverage segment order mismatch");
      }
      switch (coverage_segment.disposition()) {
        case DEVELOPMENT_NATIVE_COVERAGE_SELECTED:
          if (coverage_segment.selected_candidate_ids_size() < 2 ||
              coverage_segment.selected_candidate_ids_size() >
                  rankable.unprotected_candidate_ids.size()) {
            return absl::InvalidArgumentError(
                "quality-regression selected coverage count is invalid");
          }
          for (int candidate_index = 0;
               candidate_index <
               coverage_segment.selected_candidate_ids_size();
               ++candidate_index) {
            if (coverage_segment.selected_candidate_ids(candidate_index) !=
                rankable.unprotected_candidate_ids[candidate_index]) {
              return absl::InvalidArgumentError(
                  "quality-regression selected coverage is not the original "
                  "unprotected prefix");
            }
          }
          break;
        case DEVELOPMENT_NATIVE_COVERAGE_OMITTED_CAPACITY:
          if (!coverage_segment.selected_candidate_ids().empty()) {
            return absl::InvalidArgumentError(
                "quality-regression omitted coverage has candidate IDs");
          }
          break;
        case DEVELOPMENT_NATIVE_COVERAGE_DISPOSITION_UNSPECIFIED:
        default:
          return absl::InvalidArgumentError(
              "quality-regression persisted coverage disposition is invalid");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RequireSameCoverageMap(const NativeCoverageMap& actual,
                                    const NativeCoverageMap& expected) {
  absl::StatusOr<std::string> actual_bytes =
      SerializeDeterministically(actual);
  if (!actual_bytes.ok()) {
    return actual_bytes.status();
  }
  absl::StatusOr<std::string> expected_bytes =
      SerializeDeterministically(expected);
  if (!expected_bytes.ok()) {
    return expected_bytes.status();
  }
  if (*actual_bytes != *expected_bytes) {
    return absl::FailedPreconditionError(
        "quality-regression native coverage maps differ by objective");
  }
  return absl::OkStatus();
}

bool CandidateEqual(const converter::CandidateRankerCandidate& lhs,
                    const converter::CandidateRankerCandidate& rhs) {
  return lhs.id == rhs.id && lhs.key == rhs.key && lhs.value == rhs.value &&
         lhs.cost == rhs.cost && lhs.attributes == rhs.attributes &&
         lhs.consumed_key_size == rhs.consumed_key_size &&
         lhs.is_protected == rhs.is_protected;
}

bool RequestEqual(const converter::CandidateRankerRequest& lhs,
                  const converter::CandidateRankerRequest& rhs) {
  if (lhs.token != rhs.token || lhs.mode != rhs.mode ||
      lhs.preceding_text != rhs.preceding_text ||
      lhs.following_text != rhs.following_text || lhs.reading != rhs.reading ||
      lhs.focused_segment_id != rhs.focused_segment_id ||
      lhs.segments.size() != rhs.segments.size()) {
    return false;
  }
  for (size_t segment_index = 0; segment_index < lhs.segments.size();
       ++segment_index) {
    const converter::CandidateRankerSegment& lhs_segment =
        lhs.segments[segment_index];
    const converter::CandidateRankerSegment& rhs_segment =
        rhs.segments[segment_index];
    if (lhs_segment.id != rhs_segment.id ||
        lhs_segment.key != rhs_segment.key ||
        lhs_segment.candidates.size() != rhs_segment.candidates.size()) {
      return false;
    }
    for (size_t candidate_index = 0;
         candidate_index < lhs_segment.candidates.size(); ++candidate_index) {
      if (!CandidateEqual(lhs_segment.candidates[candidate_index],
                          rhs_segment.candidates[candidate_index])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<NativeCoverageCase>
BuildQualityRegressionDevelopmentNativeCoverageCase(
    uint64_t source_line,
    const converter::CandidateRankerRequest& production_request,
    const CandidateRankerCapacityResult& capacity) {
  NativeCoverageCase coverage_case;
  coverage_case.set_source_line(source_line);
  absl::Status status =
      CopyCapacityResult(production_request, capacity, &coverage_case);
  if (!status.ok()) {
    return status;
  }
  return coverage_case;
}

absl::StatusOr<converter::CandidateRankerRequest>
BuildQualityRegressionFixedPermutationRequest(
    const NativeCoverageCase& accepted_coverage,
    const converter::CandidateRankerRequest& original_request) {
  if (!accepted_coverage.IsInitialized() ||
      HasUnknownFieldsRecursively(accepted_coverage)) {
    return absl::InvalidArgumentError(
        "quality-regression accepted coverage case is incomplete or has "
        "unknown fields");
  }
  const std::vector<RankableSegment> rankable_segments =
      GetRankableSegments(original_request);
  if (accepted_coverage.segments_size() != rankable_segments.size()) {
    return absl::InvalidArgumentError(
        "quality-regression accepted coverage segment count mismatch");
  }

  converter::CandidateRankerRequest permuted_request = original_request;
  for (int segment_index = 0;
       segment_index < accepted_coverage.segments_size(); ++segment_index) {
    const NativeCoverageSegment& accepted_segment =
        accepted_coverage.segments(segment_index);
    const RankableSegment& rankable = rankable_segments[segment_index];
    if (accepted_segment.segment_id() != rankable.segment->id) {
      return absl::InvalidArgumentError(
          "quality-regression accepted coverage segment order mismatch");
    }
    if (accepted_segment.disposition() ==
        DEVELOPMENT_NATIVE_COVERAGE_OMITTED_CAPACITY) {
      if (!accepted_segment.selected_candidate_ids().empty()) {
        return absl::InvalidArgumentError(
            "quality-regression omitted accepted coverage has candidate IDs");
      }
      continue;
    }
    if (accepted_segment.disposition() !=
            DEVELOPMENT_NATIVE_COVERAGE_SELECTED ||
        accepted_segment.selected_candidate_ids_size() < 2 ||
        accepted_segment.selected_candidate_ids_size() >
            rankable.unprotected_candidate_ids.size()) {
      return absl::InvalidArgumentError(
          "quality-regression selected accepted coverage is invalid");
    }
    for (int candidate_index = 0;
         candidate_index < accepted_segment.selected_candidate_ids_size();
         ++candidate_index) {
      if (accepted_segment.selected_candidate_ids(candidate_index) !=
          rankable.unprotected_candidate_ids[candidate_index]) {
        return absl::InvalidArgumentError(
            "quality-regression accepted coverage is not the original "
            "unprotected prefix");
      }
    }

    converter::CandidateRankerSegment& permuted_segment =
        permuted_request.segments[rankable.segment_index];
    std::vector<size_t> selected_slots;
    for (size_t candidate_index = 0;
         candidate_index < permuted_segment.candidates.size();
         ++candidate_index) {
      if (!permuted_segment.candidates[candidate_index].is_protected &&
          selected_slots.size() <
              accepted_segment.selected_candidate_ids().size()) {
        selected_slots.push_back(candidate_index);
      }
    }
    for (size_t selected_index = 0;
         selected_index < selected_slots.size() / 2; ++selected_index) {
      std::swap(permuted_segment.candidates[selected_slots[selected_index]],
                permuted_segment.candidates[
                    selected_slots[selected_slots.size() - selected_index -
                                   1]]);
    }
  }
  return permuted_request;
}

absl::Status
ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
    const NativeCoverageCase& accepted_coverage,
    const converter::CandidateRankerRequest& original_request,
    const converter::CandidateRankerRequest& permuted_request,
    const CandidateRankerCapacityResult& permuted_capacity) {
  absl::StatusOr<converter::CandidateRankerRequest> expected_permutation =
      BuildQualityRegressionFixedPermutationRequest(
          accepted_coverage, original_request);
  if (!expected_permutation.ok()) {
    return expected_permutation.status();
  }
  if (!RequestEqual(*expected_permutation, permuted_request)) {
    return absl::InvalidArgumentError(
        "quality-regression fixed permutation request mismatch");
  }
  absl::StatusOr<NativeCoverageCase> audited =
      BuildQualityRegressionDevelopmentNativeCoverageCase(
          accepted_coverage.source_line(), permuted_request,
          permuted_capacity);
  if (!audited.ok()) {
    return audited.status();
  }
  if (audited->segments_size() != accepted_coverage.segments_size()) {
    return absl::InvalidArgumentError(
        "quality-regression permuted coverage segment count mismatch");
  }
  for (int segment_index = 0;
       segment_index < accepted_coverage.segments_size(); ++segment_index) {
    const NativeCoverageSegment& accepted =
        accepted_coverage.segments(segment_index);
    const NativeCoverageSegment& actual = audited->segments(segment_index);
    if (actual.segment_id() != accepted.segment_id() ||
        actual.disposition() != accepted.disposition() ||
        actual.selected_candidate_ids_size() !=
            accepted.selected_candidate_ids_size()) {
      return absl::FailedPreconditionError(
          "quality-regression permuted native coverage identity mismatch");
    }
    for (int candidate_index = 0;
         candidate_index < actual.selected_candidate_ids_size();
         ++candidate_index) {
      const int reversed_index =
          actual.selected_candidate_ids_size() - candidate_index - 1;
      if (actual.selected_candidate_ids(candidate_index) !=
          accepted.selected_candidate_ids(reversed_index)) {
        return absl::FailedPreconditionError(
            "quality-regression permuted selected candidate order mismatch");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<DevelopmentNativeCoverageSuite>
BuildQualityRegressionDevelopmentNativeCoverageSuite(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentExecutionConfig& config,
    absl::string_view execution_config_sha256,
    QualityRegressionNativeCapacityCallback capacity_callback) {
  absl::Status status = ValidateInputs(frozen_corpus, frozen_corpus_sha256,
                                       config, execution_config_sha256);
  if (!status.ok()) {
    return status;
  }

  DevelopmentNativeCoverageSuite suite;
  suite.set_schema_version(
      kQualityRegressionDevelopmentNativeCoverageSchemaVersion);
  suite.set_execution_config_sha256(execution_config_sha256);
  suite.set_frozen_corpus_sha256(frozen_corpus_sha256);
  suite.set_coverage_definition_version(config.coverage_definition_version());
  for (const DevelopmentObjectiveSpec& objective : config.objectives()) {
    ObjectiveNativeCoverage* objective_coverage = suite.add_coverages();
    objective_coverage->set_objective(objective.objective());
    NativeCoverageMap* map = objective_coverage->mutable_map();
    for (const QualityRegressionFrozenCase& frozen_case :
         frozen_corpus.cases()) {
      absl::StatusOr<converter::CandidateRankerRequest> request =
          ConvertFrozenCandidateRankerRequest(frozen_case.request());
      if (!request.ok()) {
        return request.status();
      }
      absl::StatusOr<CandidateRankerCapacityResult> capacity =
          capacity_callback(objective, *request);
      if (!capacity.ok()) {
        return capacity.status();
      }
      absl::StatusOr<NativeCoverageCase> coverage_case =
          BuildQualityRegressionDevelopmentNativeCoverageCase(
              frozen_case.source_line(), *request, *capacity);
      if (!coverage_case.ok()) {
        return coverage_case.status();
      }
      *map->add_cases() = std::move(*coverage_case);
    }
  }

  status = ValidateQualityRegressionDevelopmentNativeCoverageSuite(
      suite, frozen_corpus, frozen_corpus_sha256, config,
      execution_config_sha256);
  if (!status.ok()) {
    return status;
  }
  return suite;
}

absl::Status ValidateQualityRegressionDevelopmentNativeCoverageSuite(
    const DevelopmentNativeCoverageSuite& suite,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentExecutionConfig& config,
    absl::string_view execution_config_sha256) {
  absl::Status status = ValidateInputs(frozen_corpus, frozen_corpus_sha256,
                                       config, execution_config_sha256);
  if (!status.ok()) {
    return status;
  }
  if (!suite.IsInitialized() || HasUnknownFieldsRecursively(suite)) {
    return absl::InvalidArgumentError(
        "quality-regression native coverage is incomplete or has unknown "
        "fields");
  }
  if (suite.schema_version() !=
          kQualityRegressionDevelopmentNativeCoverageSchemaVersion ||
      suite.schema_version() != config.native_coverage_schema_version() ||
      suite.execution_config_sha256() != execution_config_sha256 ||
      suite.frozen_corpus_sha256() != frozen_corpus_sha256 ||
      suite.coverage_definition_version() !=
          config.coverage_definition_version() ||
      suite.coverages_size() != config.objectives_size()) {
    return absl::FailedPreconditionError(
        "quality-regression native coverage identity mismatch");
  }

  const NativeCoverageMap* reference_map = nullptr;
  for (int objective_index = 0;
       objective_index < suite.coverages_size(); ++objective_index) {
    const ObjectiveNativeCoverage& coverage =
        suite.coverages(objective_index);
    if (coverage.objective() !=
        config.objectives(objective_index).objective()) {
      return absl::InvalidArgumentError(
          "quality-regression native coverage objective order mismatch");
    }
    status = ValidateCoverageMap(coverage.map(), frozen_corpus);
    if (!status.ok()) {
      return status;
    }
    if (reference_map == nullptr) {
      reference_map = &coverage.map();
      continue;
    }
    status = RequireSameCoverageMap(coverage.map(), *reference_map);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
