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

#include "engine/evaluation/quality_regression_development_scores.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_development_config.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_native_coverage.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"

namespace mozc::engine::evaluation {
namespace {

using CandidateKey = std::pair<converter::CandidateRankerSegmentId,
                               converter::CandidateRankerCandidateId>;

struct CanonicalCandidateScores {
  uint64_t full_continuation_score_bits = 0;
  uint32_t scored_continuation_token_count = 0;

  bool operator==(const CanonicalCandidateScores&) const = default;
};

using CanonicalScoreMap = std::map<CandidateKey, CanonicalCandidateScores>;

struct PreparedCase {
  converter::CandidateRankerRequest original_request;
  converter::CandidateRankerRequest permuted_request;
};

struct ValidatedEvaluationResult {
  CanonicalScoreMap scores;
  std::vector<converter::CandidateRankerSegmentOrder> orders;
  converter::CandidateRankerMergeOrder merge;
};

absl::StatusOr<uint64_t> CanonicalScoreBits(double score) {
  if (!std::isfinite(score)) {
    return absl::InvalidArgumentError(
        "quality-regression development score must be finite");
  }
  if (score == 0.0) {
    score = 0.0;
  }
  return std::bit_cast<uint64_t>(score);
}

bool SegmentOrderEqual(const converter::CandidateRankerSegmentOrder& lhs,
                       const converter::CandidateRankerSegmentOrder& rhs) {
  return lhs.segment_id == rhs.segment_id &&
         lhs.candidate_ids == rhs.candidate_ids;
}

bool SegmentOrdersEqual(
    absl::Span<const converter::CandidateRankerSegmentOrder> lhs,
    absl::Span<const converter::CandidateRankerSegmentOrder> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t index = 0; index < lhs.size(); ++index) {
    if (!SegmentOrderEqual(lhs[index], rhs[index])) {
      return false;
    }
  }
  return true;
}

const converter::CandidateRankerSegment* FindSegment(
    const converter::CandidateRankerRequest& request,
    converter::CandidateRankerSegmentId segment_id) {
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    if (segment.id == segment_id) {
      return &segment;
    }
  }
  return nullptr;
}

const converter::CandidateRankerSegmentOrder* FindOrder(
    absl::Span<const converter::CandidateRankerSegmentOrder> orders,
    converter::CandidateRankerSegmentId segment_id) {
  for (const converter::CandidateRankerSegmentOrder& order : orders) {
    if (order.segment_id == segment_id) {
      return &order;
    }
  }
  return nullptr;
}

absl::StatusOr<ValidatedEvaluationResult> ValidateEvaluationResult(
    const converter::CandidateRankerRequest& request,
    const NativeCoverageCase& coverage_case,
    const QualityRegressionDevelopmentEvaluationResult& result) {
  std::set<CandidateKey> selected_candidates;
  size_t expected_candidate_count = 0;
  size_t expected_segment_count = 0;
  for (const NativeCoverageSegment& coverage_segment :
       coverage_case.segments()) {
    if (coverage_segment.disposition() !=
        DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
      continue;
    }
    ++expected_segment_count;
    expected_candidate_count +=
        coverage_segment.selected_candidate_ids_size();
    for (uint64_t candidate_id :
         coverage_segment.selected_candidate_ids()) {
      selected_candidates.emplace(coverage_segment.segment_id(),
                                  candidate_id);
    }
  }
  if (result.response.token != request.token ||
      result.candidate_scores.size() != expected_candidate_count) {
    return absl::InvalidArgumentError(
        "quality-regression development score response identity mismatch");
  }

  ValidatedEvaluationResult validated;
  for (const QualityRegressionDevelopmentCandidateEvaluationScore& score :
       result.candidate_scores) {
    const CandidateKey key(score.segment_id, score.candidate_id);
    if (!selected_candidates.contains(key)) {
      return absl::InvalidArgumentError(
          "quality-regression development score references an unselected "
          "candidate");
    }
    absl::StatusOr<uint64_t> full_bits =
        CanonicalScoreBits(score.full_continuation_log_probability);
    if (!full_bits.ok()) {
      return full_bits.status();
    }
    if (score.scored_continuation_token_count == 0) {
      return absl::InvalidArgumentError(
          "quality-regression development continuation score has no token");
    }
    if (!validated.scores
             .emplace(key, CanonicalCandidateScores{
                               .full_continuation_score_bits = *full_bits,
                               .scored_continuation_token_count =
                                   score.scored_continuation_token_count})
             .second) {
      return absl::AlreadyExistsError(
          "quality-regression development candidate score is duplicated");
    }
  }

  ::mozc::engine::CandidateRankerBatchPlan plan;
  std::vector<double> sequence_scores;
  plan.segments.reserve(expected_segment_count);
  sequence_scores.reserve(expected_candidate_count);
  for (const NativeCoverageSegment& coverage_segment :
       coverage_case.segments()) {
    if (coverage_segment.disposition() !=
        DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
      continue;
    }
    const converter::CandidateRankerSegment* request_segment =
        FindSegment(request, coverage_segment.segment_id());
    if (request_segment == nullptr) {
      return absl::InvalidArgumentError(
          "quality-regression development score segment is absent");
    }
    std::set<uint64_t> selected_ids(
        coverage_segment.selected_candidate_ids().begin(),
        coverage_segment.selected_candidate_ids().end());
    ::mozc::engine::CandidateRankerSegmentPlan& plan_segment =
        plan.segments.emplace_back();
    plan_segment.segment_id = coverage_segment.segment_id();
    for (size_t candidate_index = 0;
         candidate_index < request_segment->candidates.size();
         ++candidate_index) {
      const converter::CandidateRankerCandidate& candidate =
          request_segment->candidates[candidate_index];
      if (!selected_ids.contains(candidate.id)) {
        continue;
      }
      if (candidate.is_protected) {
        return absl::PermissionDeniedError(
            "quality-regression selected score candidate is protected");
      }
      const auto score =
          validated.scores.find({coverage_segment.segment_id(), candidate.id});
      if (score == validated.scores.end()) {
        return absl::InvalidArgumentError(
            "quality-regression selected candidate score is missing");
      }
      const int32_t sequence_id =
          static_cast<int32_t>(sequence_scores.size());
      plan_segment.candidates.push_back(
          {.candidate_id = candidate.id,
           .original_candidate_index = candidate_index,
           .sequence_id = sequence_id});
      sequence_scores.push_back(std::bit_cast<double>(
          score->second.full_continuation_score_bits));
    }
    if (plan_segment.candidates.size() != selected_ids.size()) {
      return absl::InvalidArgumentError(
          "quality-regression selected score candidate membership mismatch");
    }
  }
  plan.sequence_count = sequence_scores.size();

  absl::StatusOr<std::vector<converter::CandidateRankerSegmentOrder>> orders =
      ::mozc::engine::OrderCandidateRankerContinuations(plan,
                                                        sequence_scores);
  if (!orders.ok()) {
    return orders.status();
  }
  if (!SegmentOrdersEqual(*orders, result.response.segment_orders)) {
    return absl::FailedPreconditionError(
        "quality-regression development response is not the production "
        "stable score order");
  }
  absl::StatusOr<converter::CandidateRankerMergeOrder> merge =
      converter::BuildCandidateRankerMergeOrder(request, result.response,
                                                request.token);
  if (!merge.ok()) {
    return merge.status();
  }
  validated.orders = std::move(*orders);
  validated.merge = std::move(*merge);
  return validated;
}

absl::Status CompareOriginalAndPermutedResults(
    const NativeCoverageCase& coverage_case,
    const ValidatedEvaluationResult& original,
    const ValidatedEvaluationResult& permuted) {
  if (original.scores != permuted.scores) {
    return absl::FailedPreconditionError(
        "quality-regression fixed permutation changed mapped score bits");
  }
  for (const NativeCoverageSegment& coverage_segment :
       coverage_case.segments()) {
    if (coverage_segment.disposition() !=
        DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
      continue;
    }
    const converter::CandidateRankerSegmentOrder* original_order =
        FindOrder(original.orders, coverage_segment.segment_id());
    const converter::CandidateRankerSegmentOrder* permuted_order =
        FindOrder(permuted.orders, coverage_segment.segment_id());
    const converter::CandidateRankerSegmentOrder* original_merge =
        FindOrder(original.merge.segment_orders, coverage_segment.segment_id());
    const converter::CandidateRankerSegmentOrder* permuted_merge =
        FindOrder(permuted.merge.segment_orders, coverage_segment.segment_id());
    if (original_order == nullptr || permuted_order == nullptr ||
        original_merge == nullptr || permuted_merge == nullptr) {
      return absl::InvalidArgumentError(
          "quality-regression fixed permutation segment result is missing");
    }

    std::vector<uint64_t> original_ordered_bits;
    std::vector<uint64_t> permuted_ordered_bits;
    std::set<uint64_t> distinct_bits;
    double maximum_score = -std::numeric_limits<double>::infinity();
    size_t maximum_count = 0;
    for (uint64_t candidate_id : original_order->candidate_ids) {
      const uint64_t bits =
          original.scores
              .at({coverage_segment.segment_id(), candidate_id})
              .full_continuation_score_bits;
      original_ordered_bits.push_back(bits);
      distinct_bits.insert(bits);
      const double score = std::bit_cast<double>(bits);
      if (score > maximum_score) {
        maximum_score = score;
        maximum_count = 1;
      } else if (score == maximum_score) {
        ++maximum_count;
      }
    }
    for (uint64_t candidate_id : permuted_order->candidate_ids) {
      permuted_ordered_bits.push_back(
          permuted.scores
              .at({coverage_segment.segment_id(), candidate_id})
              .full_continuation_score_bits);
    }
    if (original_ordered_bits != permuted_ordered_bits) {
      return absl::FailedPreconditionError(
          "quality-regression fixed permutation changed ordered score bits");
    }
    if (distinct_bits.size() == original_order->candidate_ids.size() &&
        (!SegmentOrderEqual(*original_order, *permuted_order) ||
         !SegmentOrderEqual(*original_merge, *permuted_merge))) {
      return absl::FailedPreconditionError(
          "quality-regression fixed permutation changed a distinct-score "
          "order or merge");
    }
    if (maximum_count == 1 &&
        original_order->candidate_ids.front() !=
            permuted_order->candidate_ids.front()) {
      return absl::FailedPreconditionError(
          "quality-regression fixed permutation changed the unique winner");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateIdentity(
    const DevelopmentObjectiveScores& scores,
    const DevelopmentScoreRunnerConfig& score_config,
    absl::string_view score_config_sha256,
    absl::string_view execution_config_sha256,
    absl::string_view frozen_corpus_sha256,
    absl::string_view native_coverage_sha256) {
  if (!scores.IsInitialized() || HasUnknownFieldsRecursively(scores)) {
    return absl::InvalidArgumentError(
        "quality-regression development scores are incomplete or have "
        "unknown fields");
  }
  const DevelopmentObjectiveScoresIdentity& identity = scores.identity();
  if (scores.schema_version() !=
          kQualityRegressionDevelopmentScoreArtifactSchemaVersion ||
      scores.schema_version() != score_config.score_artifact_schema_version() ||
      identity.score_config_sha256() != score_config_sha256 ||
      identity.execution_config_sha256() != execution_config_sha256 ||
      identity.frozen_corpus_sha256() != frozen_corpus_sha256 ||
      identity.native_coverage_suite_sha256() != native_coverage_sha256 ||
      identity.score_definition_version() !=
          score_config.score_definition_version() ||
      identity.numeric_profile() != score_config.numeric_profile() ||
      identity.layout() != score_config.layout()) {
    return absl::FailedPreconditionError(
        "quality-regression development score artifact identity mismatch");
  }
  return absl::OkStatus();
}

absl::Status PersistOriginalResult(
    const converter::CandidateRankerRequest& original_request,
    const NativeCoverageCase& coverage_case,
    const ValidatedEvaluationResult& original,
    DevelopmentCaseScores* output_case) {
  output_case->set_source_line(coverage_case.source_line());
  for (const NativeCoverageSegment& coverage_segment :
       coverage_case.segments()) {
    if (coverage_segment.disposition() !=
        DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
      continue;
    }
    const converter::CandidateRankerSegment* request_segment =
        FindSegment(original_request, coverage_segment.segment_id());
    const converter::CandidateRankerSegmentOrder* order =
        FindOrder(original.orders, coverage_segment.segment_id());
    if (request_segment == nullptr || order == nullptr) {
      return absl::InvalidArgumentError(
          "quality-regression original score result segment is missing");
    }
    DevelopmentSegmentScores* output_segment = output_case->add_segments();
    output_segment->set_segment_id(coverage_segment.segment_id());
    for (uint64_t candidate_id : order->candidate_ids) {
      size_t original_index = request_segment->candidates.size();
      for (size_t candidate_index = 0;
           candidate_index < request_segment->candidates.size();
           ++candidate_index) {
        if (request_segment->candidates[candidate_index].id == candidate_id) {
          original_index = candidate_index;
          break;
        }
      }
      if (original_index == request_segment->candidates.size()) {
        return absl::InvalidArgumentError(
            "quality-regression original score candidate is missing");
      }
      const converter::CandidateRankerCandidate& candidate =
          request_segment->candidates[original_index];
      DevelopmentCandidateScore* output_candidate =
          output_segment->add_candidates();
      output_candidate->set_candidate_id(candidate.id);
      output_candidate->set_original_candidate_index(original_index);
      output_candidate->set_mozc_cost(candidate.cost);
      const CanonicalCandidateScores& scores =
          original.scores.at({coverage_segment.segment_id(), candidate.id});
      output_candidate->set_full_continuation_score_bits(
          scores.full_continuation_score_bits);
      output_candidate->set_scored_continuation_token_count(
          scores.scored_continuation_token_count);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateQualityRegressionDevelopmentScoreRunnerConfig(
    const DevelopmentScoreRunnerConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "quality-regression development score config is incomplete or has "
        "unknown fields");
  }
  if (config.schema_version() !=
          kQualityRegressionDevelopmentScoreRunnerConfigSchemaVersion ||
      config.score_artifact_schema_version() !=
          kQualityRegressionDevelopmentScoreArtifactSchemaVersion ||
      config.score_definition_version() !=
          kQualityRegressionDevelopmentScoreDefinitionVersion ||
      config.numeric_profile() !=
          DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED ||
      config.layout() != DEVELOPMENT_EVALUATION_SHARED_TRIE) {
    return absl::InvalidArgumentError(
        "quality-regression development score schema or execution identity "
        "is invalid");
  }
  if (!IsQualityRegressionLowercaseHex(config.execution_config_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(config.frozen_corpus_sha256(), 64) ||
      !IsQualityRegressionLowercaseHex(
          config.native_coverage_suite_sha256(), 64)) {
    return absl::InvalidArgumentError(
        "quality-regression development score hash identity is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionDevelopmentScoreRunnerInputs(
    const DevelopmentScoreRunnerConfig& score_config,
    absl::string_view score_config_sha256,
    const DevelopmentExecutionConfig& execution_config,
    absl::string_view execution_config_sha256,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentNativeCoverageSuite& native_coverage,
    absl::string_view native_coverage_sha256,
    absl::Span<const ::mozc::engine::CandidateRankerModelManifest> manifests,
    absl::Span<const std::string> manifest_sha256s) {
  absl::Status status =
      ValidateQualityRegressionDevelopmentScoreRunnerConfig(score_config);
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(score_config_sha256, 64) ||
      score_config.execution_config_sha256() != execution_config_sha256 ||
      score_config.frozen_corpus_sha256() != frozen_corpus_sha256 ||
      score_config.native_coverage_suite_sha256() != native_coverage_sha256 ||
      !IsQualityRegressionLowercaseHex(native_coverage_sha256, 64)) {
    return absl::FailedPreconditionError(
        "quality-regression development score input hash mismatch");
  }
  status = ValidateQualityRegressionDevelopmentNativeCoverageSuite(
      native_coverage, frozen_corpus, frozen_corpus_sha256, execution_config,
      execution_config_sha256);
  if (!status.ok()) {
    return status;
  }
  return ValidateQualityRegressionDevelopmentManifests(
      execution_config, manifests, manifest_sha256s);
}

absl::StatusOr<DevelopmentObjectiveScores>
BuildQualityRegressionDevelopmentObjectiveScores(
    const DevelopmentScoreRunnerConfig& score_config,
    absl::string_view score_config_sha256,
    const DevelopmentExecutionConfig& execution_config,
    absl::string_view execution_config_sha256,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentNativeCoverageSuite& native_coverage,
    absl::string_view native_coverage_sha256,
    absl::Span<const ::mozc::engine::CandidateRankerModelManifest> manifests,
    absl::Span<const std::string> manifest_sha256s,
    QualityRegressionNativeCapacityCallback capacity_callback,
    QualityRegressionDevelopmentScoreCallback score_callback) {
  absl::Status status = ValidateQualityRegressionDevelopmentScoreRunnerInputs(
      score_config, score_config_sha256, execution_config,
      execution_config_sha256, frozen_corpus, frozen_corpus_sha256,
      native_coverage, native_coverage_sha256, manifests, manifest_sha256s);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::vector<PreparedCase>> prepared_objectives;
  prepared_objectives.resize(execution_config.objectives_size());
  for (int objective_index = 0;
       objective_index < execution_config.objectives_size();
       ++objective_index) {
    const DevelopmentObjectiveSpec& objective =
        execution_config.objectives(objective_index);
    const NativeCoverageMap& coverage_map =
        native_coverage.coverages(objective_index).map();
    std::vector<PreparedCase>& prepared_cases =
        prepared_objectives[objective_index];
    prepared_cases.reserve(frozen_corpus.cases_size());
    for (int case_index = 0; case_index < frozen_corpus.cases_size();
         ++case_index) {
      absl::StatusOr<converter::CandidateRankerRequest> original_request =
          ConvertFrozenCandidateRankerRequest(
              frozen_corpus.cases(case_index).request());
      if (!original_request.ok()) {
        return original_request.status();
      }
      absl::StatusOr<converter::CandidateRankerRequest> permuted_request =
          BuildQualityRegressionFixedPermutationRequest(
              coverage_map.cases(case_index), *original_request);
      if (!permuted_request.ok()) {
        return permuted_request.status();
      }
      absl::StatusOr<::mozc::engine::CandidateRankerCapacityResult> capacity =
          capacity_callback(objective, *permuted_request);
      if (!capacity.ok()) {
        return capacity.status();
      }
      status =
          ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
              coverage_map.cases(case_index), *original_request,
              *permuted_request, *capacity);
      if (!status.ok()) {
        return status;
      }
      prepared_cases.push_back({.original_request =
                                    std::move(*original_request),
                                .permuted_request =
                                    std::move(*permuted_request)});
    }
  }

  DevelopmentObjectiveScores scores;
  scores.set_schema_version(
      kQualityRegressionDevelopmentScoreArtifactSchemaVersion);
  DevelopmentObjectiveScoresIdentity* identity = scores.mutable_identity();
  identity->set_score_config_sha256(score_config_sha256);
  identity->set_execution_config_sha256(execution_config_sha256);
  identity->set_frozen_corpus_sha256(frozen_corpus_sha256);
  identity->set_native_coverage_suite_sha256(native_coverage_sha256);
  identity->set_score_definition_version(
      score_config.score_definition_version());
  identity->set_numeric_profile(score_config.numeric_profile());
  identity->set_layout(score_config.layout());

  for (int objective_index = 0;
       objective_index < execution_config.objectives_size();
       ++objective_index) {
    const DevelopmentObjectiveSpec& objective =
        execution_config.objectives(objective_index);
    const NativeCoverageMap& coverage_map =
        native_coverage.coverages(objective_index).map();
    DevelopmentObjectiveScoreSet* objective_scores =
        scores.add_objectives();
    objective_scores->set_objective(objective.objective());
    for (int case_index = 0; case_index < frozen_corpus.cases_size();
         ++case_index) {
      const PreparedCase& prepared =
          prepared_objectives[objective_index][case_index];
      absl::StatusOr<QualityRegressionDevelopmentEvaluationResult>
          original_result = score_callback(objective, prepared.original_request);
      if (!original_result.ok()) {
        return original_result.status();
      }
      absl::StatusOr<QualityRegressionDevelopmentEvaluationResult>
          permuted_result = score_callback(objective, prepared.permuted_request);
      if (!permuted_result.ok()) {
        return permuted_result.status();
      }
      absl::StatusOr<ValidatedEvaluationResult> original =
          ValidateEvaluationResult(prepared.original_request,
                                   coverage_map.cases(case_index),
                                   *original_result);
      if (!original.ok()) {
        return original.status();
      }
      absl::StatusOr<ValidatedEvaluationResult> permuted =
          ValidateEvaluationResult(prepared.permuted_request,
                                   coverage_map.cases(case_index),
                                   *permuted_result);
      if (!permuted.ok()) {
        return permuted.status();
      }
      status = CompareOriginalAndPermutedResults(
          coverage_map.cases(case_index), *original, *permuted);
      if (!status.ok()) {
        return status;
      }
      status = PersistOriginalResult(
          prepared.original_request, coverage_map.cases(case_index), *original,
          objective_scores->add_cases());
      if (!status.ok()) {
        return status;
      }
    }
  }

  status = ValidateQualityRegressionDevelopmentObjectiveScores(
      scores, score_config, score_config_sha256, execution_config,
      execution_config_sha256, frozen_corpus, frozen_corpus_sha256,
      native_coverage, native_coverage_sha256, manifests, manifest_sha256s);
  if (!status.ok()) {
    return status;
  }
  return scores;
}

absl::Status ValidateQualityRegressionDevelopmentObjectiveScores(
    const DevelopmentObjectiveScores& scores,
    const DevelopmentScoreRunnerConfig& score_config,
    absl::string_view score_config_sha256,
    const DevelopmentExecutionConfig& execution_config,
    absl::string_view execution_config_sha256,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentNativeCoverageSuite& native_coverage,
    absl::string_view native_coverage_sha256,
    absl::Span<const ::mozc::engine::CandidateRankerModelManifest> manifests,
    absl::Span<const std::string> manifest_sha256s) {
  absl::Status status = ValidateQualityRegressionDevelopmentScoreRunnerInputs(
      score_config, score_config_sha256, execution_config,
      execution_config_sha256, frozen_corpus, frozen_corpus_sha256,
      native_coverage, native_coverage_sha256, manifests, manifest_sha256s);
  if (!status.ok()) {
    return status;
  }
  status = ValidateIdentity(scores, score_config, score_config_sha256,
                            execution_config_sha256, frozen_corpus_sha256,
                            native_coverage_sha256);
  if (!status.ok()) {
    return status;
  }
  if (scores.objectives_size() != execution_config.objectives_size()) {
    return absl::InvalidArgumentError(
        "quality-regression development score objective count mismatch");
  }

  for (int objective_index = 0;
       objective_index < scores.objectives_size(); ++objective_index) {
    const DevelopmentObjectiveScoreSet& objective_scores =
        scores.objectives(objective_index);
    const NativeCoverageMap& coverage_map =
        native_coverage.coverages(objective_index).map();
    if (objective_scores.objective() !=
            execution_config.objectives(objective_index).objective() ||
        objective_scores.cases_size() != frozen_corpus.cases_size()) {
      return absl::InvalidArgumentError(
          "quality-regression development score objective identity mismatch");
    }
    for (int case_index = 0; case_index < objective_scores.cases_size();
         ++case_index) {
      const DevelopmentCaseScores& case_scores =
          objective_scores.cases(case_index);
      const NativeCoverageCase& coverage_case =
          coverage_map.cases(case_index);
      absl::StatusOr<converter::CandidateRankerRequest> request =
          ConvertFrozenCandidateRankerRequest(
              frozen_corpus.cases(case_index).request());
      if (!request.ok()) {
        return request.status();
      }
      size_t selected_segment_count = 0;
      for (const NativeCoverageSegment& coverage_segment :
           coverage_case.segments()) {
        if (coverage_segment.disposition() ==
            DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
          ++selected_segment_count;
        }
      }
      if (case_scores.source_line() != coverage_case.source_line() ||
          case_scores.segments_size() != selected_segment_count) {
        return absl::InvalidArgumentError(
            "quality-regression development score case shape mismatch");
      }

      QualityRegressionDevelopmentEvaluationResult stored_result;
      stored_result.response.token = request->token;
      int stored_segment_index = 0;
      for (const NativeCoverageSegment& coverage_segment :
           coverage_case.segments()) {
        if (coverage_segment.disposition() !=
            DEVELOPMENT_NATIVE_COVERAGE_SELECTED) {
          continue;
        }
        const DevelopmentSegmentScores& segment_scores =
            case_scores.segments(stored_segment_index++);
        const converter::CandidateRankerSegment* request_segment =
            FindSegment(*request, coverage_segment.segment_id());
        if (request_segment == nullptr ||
            segment_scores.segment_id() != coverage_segment.segment_id() ||
            segment_scores.candidates_size() !=
                coverage_segment.selected_candidate_ids_size()) {
          return absl::InvalidArgumentError(
              "quality-regression development score segment shape mismatch");
        }
        converter::CandidateRankerSegmentOrder& stored_order =
            stored_result.response.segment_orders.emplace_back();
        stored_order.segment_id = segment_scores.segment_id();
        std::set<uint64_t> accepted_ids(
            coverage_segment.selected_candidate_ids().begin(),
            coverage_segment.selected_candidate_ids().end());
        std::set<uint64_t> stored_ids;
        for (const DevelopmentCandidateScore& candidate_score :
             segment_scores.candidates()) {
          if (!accepted_ids.contains(candidate_score.candidate_id()) ||
              !stored_ids.insert(candidate_score.candidate_id()).second ||
              candidate_score.original_candidate_index() >=
                  request_segment->candidates.size()) {
            return absl::InvalidArgumentError(
                "quality-regression development score candidate identity "
                "mismatch");
          }
          const converter::CandidateRankerCandidate& candidate =
              request_segment->candidates[
                  candidate_score.original_candidate_index()];
          if (candidate.id != candidate_score.candidate_id() ||
              candidate.cost != candidate_score.mozc_cost() ||
              candidate.is_protected) {
            return absl::InvalidArgumentError(
                "quality-regression development score candidate metadata "
                "mismatch");
          }
          const double full_score = std::bit_cast<double>(
              candidate_score.full_continuation_score_bits());
          absl::StatusOr<uint64_t> canonical_full =
              CanonicalScoreBits(full_score);
          if (!canonical_full.ok() ||
              *canonical_full !=
                  candidate_score.full_continuation_score_bits() ||
              candidate_score.scored_continuation_token_count() == 0) {
            return absl::InvalidArgumentError(
                "quality-regression persisted score bits are invalid");
          }
          stored_order.candidate_ids.push_back(candidate.id);
          stored_result.candidate_scores.push_back(
              {.segment_id = segment_scores.segment_id(),
               .candidate_id = candidate.id,
               .full_continuation_log_probability = full_score,
               .scored_continuation_token_count =
                   candidate_score.scored_continuation_token_count()});
        }
      }
      absl::StatusOr<ValidatedEvaluationResult> validated =
          ValidateEvaluationResult(*request, coverage_case, stored_result);
      if (!validated.ok()) {
        return validated.status();
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
