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

#ifndef MOZC_ENGINE_LLAMA_CANDIDATE_RANKER_H_
#define MOZC_ENGINE_LLAMA_CANDIDATE_RANKER_H_

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"

namespace mozc {
namespace engine {

enum class LlamaCandidateRankerEvaluationLayout : uint8_t {
  kSharedTrie,
  kIndependentCompleteSequences,
};

enum class LlamaCandidateRankerEvaluationProfile : uint8_t {
  kConfigured,
  kF32KvFlashDisabled,
};

enum class LlamaCandidateRankerEdgeContribution : uint8_t {
  kCandidateOrBoundary,
  kFollowingContext,
  kTerminal,
};

struct LlamaCandidateRankerEvaluationIdentity {
  std::string manifest_sha256;
  std::string gguf_file_name;
  std::string gguf_sha256;
  std::string tokenizer_sha256;
  std::string quantization;
  std::string source_revision;
  std::string runtime_revision;
};

absl::Status ValidateLlamaCandidateRankerModelContextCapacity(
    const CandidateRankerModelManifest& manifest, int32_t model_context);

LlamaCandidateRankerEvaluationIdentity
BuildLlamaCandidateRankerEvaluationIdentity(
    const CandidateRankerModelManifest& manifest,
    absl::string_view manifest_bytes);

void WriteLlamaCandidateRankerEvaluationIdentity(
    std::ostream& output,
    const LlamaCandidateRankerEvaluationIdentity& identity);

void WriteLlamaCandidateRankerEvaluationNumericProfile(
    std::ostream& output, LlamaCandidateRankerEvaluationProfile profile);

LlamaCandidateRankerEdgeContribution
ClassifyLlamaCandidateRankerEdgeContribution(
    uint32_t target_position, bool is_terminal,
    uint32_t following_context_start_position);

struct LlamaCandidateRankerEdgeScore {
  uint32_t target_position = 0;
  int32_t target_token_id = 0;
  bool is_terminal = false;
  LlamaCandidateRankerEdgeContribution contribution =
      LlamaCandidateRankerEdgeContribution::kCandidateOrBoundary;
  double log_probability = 0.0;
};

struct LlamaCandidateRankerCandidateScore {
  converter::CandidateRankerSegmentId segment_id = 0;
  converter::CandidateRankerCandidateId candidate_id = 0;
  uint32_t common_prefix_token_count = 0;
  std::vector<int32_t> input_token_ids;
  std::vector<LlamaCandidateRankerEdgeScore> edges;
  double full_continuation_log_probability = 0.0;
  uint32_t scored_continuation_token_count = 0;
};

struct LlamaCandidateRankerEvaluationResult {
  converter::CandidateRankerResponse response;
  std::vector<LlamaCandidateRankerCandidateScore> candidate_scores;
  uint64_t input_node_count = 0;
  uint64_t output_row_count = 0;
  uint64_t decode_batch_count = 0;
};

class LlamaCandidateRankerCapacityAuditor final {
 public:
  static absl::StatusOr<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>>
  Create(absl::string_view manifest_path,
         absl::string_view model_directory,
         const LlamaCandidateRankerEvaluationIdentity& expected_identity);

  ~LlamaCandidateRankerCapacityAuditor();

  absl::StatusOr<CandidateRankerCapacityResult> Audit(
      const converter::CandidateRankerRequest& request,
      const converter::CandidateRankerCancellation& cancellation);

  const LlamaCandidateRankerEvaluationIdentity& EvaluationIdentity() const {
    return evaluation_identity_;
  }

 private:
  struct RuntimeState;

  LlamaCandidateRankerCapacityAuditor(
      CandidateRankerModelManifest manifest,
      LlamaCandidateRankerEvaluationIdentity evaluation_identity,
      std::unique_ptr<RuntimeState> runtime);

  CandidateRankerModelManifest manifest_;
  const LlamaCandidateRankerEvaluationIdentity evaluation_identity_;
  std::unique_ptr<RuntimeState> runtime_;
};

class LlamaCandidateRanker final
    : public converter::CandidateRankerBackendInterface {
 public:
  static absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> Create(
      absl::string_view manifest_path, absl::string_view model_directory);
  static absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> Create(
      absl::string_view manifest_path, absl::string_view model_directory,
      const LlamaCandidateRankerEvaluationIdentity& expected_identity);

  ~LlamaCandidateRanker() override;

  absl::StatusOr<converter::CandidateRankerResponse> Rank(
      const converter::CandidateRankerRequest& request,
      const converter::CandidateRankerCancellation& cancellation) override;

  absl::StatusOr<LlamaCandidateRankerEvaluationResult> ScoreForEvaluation(
      const converter::CandidateRankerRequest& request,
      const converter::CandidateRankerCancellation& cancellation,
      LlamaCandidateRankerEvaluationLayout layout,
      LlamaCandidateRankerEvaluationProfile profile);

  const LlamaCandidateRankerEvaluationIdentity& EvaluationIdentity() const {
    return evaluation_identity_;
  }

 private:
  struct RuntimeState;

  static absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> CreateImpl(
      absl::string_view manifest_path, absl::string_view model_directory,
      const LlamaCandidateRankerEvaluationIdentity* expected_identity);

  LlamaCandidateRanker(CandidateRankerModelManifest manifest,
                       LlamaCandidateRankerEvaluationIdentity
                           evaluation_identity,
                       std::unique_ptr<RuntimeState> runtime);

  CandidateRankerModelManifest manifest_;
  const LlamaCandidateRankerEvaluationIdentity evaluation_identity_;
  std::unique_ptr<RuntimeState> runtime_;
};

}  // namespace engine
}  // namespace mozc

#endif  // MOZC_ENGINE_LLAMA_CANDIDATE_RANKER_H_
