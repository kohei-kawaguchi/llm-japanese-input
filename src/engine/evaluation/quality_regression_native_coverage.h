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

#ifndef MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_NATIVE_COVERAGE_H_
#define MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_NATIVE_COVERAGE_H_

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"

namespace mozc::engine::evaluation {

using QualityRegressionNativeCapacityCallback = absl::FunctionRef<
    absl::StatusOr<::mozc::engine::CandidateRankerCapacityResult>(
        const DevelopmentObjectiveSpec&,
        const converter::CandidateRankerRequest&)>;

absl::StatusOr<NativeCoverageCase>
BuildQualityRegressionDevelopmentNativeCoverageCase(
    uint64_t source_line,
    const converter::CandidateRankerRequest& production_request,
    const ::mozc::engine::CandidateRankerCapacityResult& capacity);

absl::StatusOr<converter::CandidateRankerRequest>
BuildQualityRegressionFixedPermutationRequest(
    const NativeCoverageCase& accepted_coverage,
    const converter::CandidateRankerRequest& original_request);

absl::Status
ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
    const NativeCoverageCase& accepted_coverage,
    const converter::CandidateRankerRequest& original_request,
    const converter::CandidateRankerRequest& permuted_request,
    const ::mozc::engine::CandidateRankerCapacityResult& permuted_capacity);

absl::StatusOr<DevelopmentNativeCoverageSuite>
BuildQualityRegressionDevelopmentNativeCoverageSuite(
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentExecutionConfig& config,
    absl::string_view execution_config_sha256,
    QualityRegressionNativeCapacityCallback capacity_callback);

absl::Status ValidateQualityRegressionDevelopmentNativeCoverageSuite(
    const DevelopmentNativeCoverageSuite& suite,
    const QualityRegressionFrozenCorpus& frozen_corpus,
    absl::string_view frozen_corpus_sha256,
    const DevelopmentExecutionConfig& config,
    absl::string_view execution_config_sha256);

}  // namespace mozc::engine::evaluation

#endif  // MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_NATIVE_COVERAGE_H_
