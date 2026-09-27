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

#ifndef MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_DEVELOPMENT_SCORES_H_
#define MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_DEVELOPMENT_SCORES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_native_coverage.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"

namespace mozc::engine::evaluation {

inline constexpr uint32_t
    kQualityRegressionDevelopmentScoreRunnerConfigSchemaVersion = 1;
inline constexpr uint32_t
    kQualityRegressionDevelopmentScoreArtifactSchemaVersion = 2;
inline constexpr uint32_t
    kQualityRegressionDevelopmentScoreDefinitionVersion = 2;

struct QualityRegressionDevelopmentCandidateEvaluationScore {
  converter::CandidateRankerSegmentId segment_id = 0;
  converter::CandidateRankerCandidateId candidate_id = 0;
  double full_continuation_log_probability = 0.0;
  uint32_t scored_continuation_token_count = 0;
};

struct QualityRegressionDevelopmentEvaluationResult {
  converter::CandidateRankerResponse response;
  std::vector<QualityRegressionDevelopmentCandidateEvaluationScore>
      candidate_scores;
};

using QualityRegressionDevelopmentScoreCallback = absl::FunctionRef<
    absl::StatusOr<QualityRegressionDevelopmentEvaluationResult>(
        const DevelopmentObjectiveSpec&,
        const converter::CandidateRankerRequest&)>;

absl::Status ValidateQualityRegressionDevelopmentScoreRunnerConfig(
    const DevelopmentScoreRunnerConfig& config);

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
    absl::Span<const std::string> manifest_sha256s);

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
    QualityRegressionDevelopmentScoreCallback score_callback);

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
    absl::Span<const std::string> manifest_sha256s);

}  // namespace mozc::engine::evaluation

#endif  // MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_DEVELOPMENT_SCORES_H_
