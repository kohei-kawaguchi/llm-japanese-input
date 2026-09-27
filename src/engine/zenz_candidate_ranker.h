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

#ifndef MOZC_ENGINE_ZENZ_CANDIDATE_RANKER_H_
#define MOZC_ENGINE_ZENZ_CANDIDATE_RANKER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"

namespace mozc {
namespace engine {

enum class ZenzRightWindow : uint8_t {
  kFull,
  kNext,
  kNone,
};

struct ZenzCandidateRankerOptions {
  std::string model_path;
  int32_t top_k = 0;
  ZenzRightWindow right_window = ZenzRightWindow::kNext;
  double copy_penalty = 0.0;
  double order_prior = 0.0;
  int32_t thread_count = 0;
  int32_t context_size = 0;
};

// The installed model and the settings selected on the development corpus in
// .cursor/plans/local-reranker-zenz.md.
inline constexpr char kInstalledZenzModelFileName[] =
    "zenz-v3.2-small-Q5_K_M.gguf";

ZenzCandidateRankerOptions InstalledZenzCandidateRankerOptions(
    std::string model_path);

// Returns the ranker for the model installed in the server directory, or
// nullptr when no model is installed there.
absl::StatusOr<std::shared_ptr<converter::CandidateRankerBackendInterface>>
CreateInstalledZenzCandidateRanker();

struct ZenzScoredOutput {
  converter::CandidateRankerSegmentId segment_id = 0;
  converter::CandidateRankerCandidateId candidate_id = 0;
  std::string prefix;
  std::string body;
  bool add_end = false;
};

// zenz scores the conversion of the complete reading, so it ranks only
// conversion requests and leaves prediction and suggestion in Mozc order.
bool IsZenzRankedMode(converter::CandidateRankerMode mode);

std::string BuildZenzPrompt(const converter::CandidateRankerRequest& request);

// Returns the scored outputs of every segment in request order, limited to the
// first `top_k` unprotected candidates of each segment.
std::vector<ZenzScoredOutput> BuildZenzScoredOutputs(
    const converter::CandidateRankerRequest& request, int32_t top_k,
    ZenzRightWindow right_window);

// Orders the scored candidates of `segment` by calibrated score. `scores`
// holds the raw log probability of each of the first `scores.size()`
// unprotected candidates in Mozc order.
converter::CandidateRankerSegmentOrder OrderZenzSegment(
    const converter::CandidateRankerSegment& segment,
    const std::vector<double>& scores, double copy_penalty,
    double order_prior);

class ZenzCandidateRanker final
    : public converter::CandidateRankerBackendInterface {
 public:
  static absl::StatusOr<std::shared_ptr<ZenzCandidateRanker>> Create(
      const ZenzCandidateRankerOptions& options);

  ~ZenzCandidateRanker() override;

  absl::StatusOr<converter::CandidateRankerResponse> Rank(
      const converter::CandidateRankerRequest& request,
      const converter::CandidateRankerCancellation& cancellation) override;

 private:
  struct RuntimeState;

  ZenzCandidateRanker(ZenzCandidateRankerOptions options,
                      std::unique_ptr<RuntimeState> runtime);

  const ZenzCandidateRankerOptions options_;
  std::unique_ptr<RuntimeState> runtime_;
};

}  // namespace engine
}  // namespace mozc

#endif  // MOZC_ENGINE_ZENZ_CANDIDATE_RANKER_H_
