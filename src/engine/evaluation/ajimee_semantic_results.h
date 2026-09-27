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

#ifndef MOZC_ENGINE_EVALUATION_AJIMEE_SEMANTIC_RESULTS_H_
#define MOZC_ENGINE_EVALUATION_AJIMEE_SEMANTIC_RESULTS_H_

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_semantic_results.pb.h"

namespace mozc::engine::evaluation {

inline constexpr uint32_t kAjimeeSemanticRunnerConfigSchemaVersion = 1;
inline constexpr uint32_t kAjimeeSemanticResultsSchemaVersion = 1;

using AjimeeSemanticRankCallback = absl::FunctionRef<
    absl::StatusOr<converter::CandidateRankerResponse>(
        const converter::CandidateRankerRequest&)>;

absl::Status ValidateAjimeeSemanticRunnerConfig(
    const AjimeeSemanticRunnerConfig& config);

absl::Status ValidateAjimeeSemanticResults(
    const AjimeeSemanticResults& results,
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config);

absl::StatusOr<AjimeeSemanticResults> BuildAjimeeSemanticResults(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config,
    const AjimeeSemanticResultsIdentity& identity,
    AjimeeSemanticRankCallback rank_callback);

}  // namespace mozc::engine::evaluation

#endif  // MOZC_ENGINE_EVALUATION_AJIMEE_SEMANTIC_RESULTS_H_
