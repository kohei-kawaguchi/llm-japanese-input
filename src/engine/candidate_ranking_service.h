// Copyright 2026 LLM Japanese Input Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of LLM Japanese Input Authors nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MOZC_ENGINE_CANDIDATE_RANKING_SERVICE_H_
#define MOZC_ENGINE_CANDIDATE_RANKING_SERVICE_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/time/time.h"
#include "converter/candidate_ranker.h"

namespace mozc {
namespace engine {

enum class CandidateRankingResultCode : uint8_t {
  kSuccess,
  kTimedOut,
  kSuperseded,
  kSessionCancelled,
  kShutdown,
  kBackendMissing,
  kBackendError,
  kInvalidResponse,
};

struct CandidateRankingResult {
  CandidateRankingResultCode code = CandidateRankingResultCode::kBackendError;
  std::optional<converter::CandidateRankerResponse> response;
  absl::Duration elapsed = absl::ZeroDuration();

  bool ok() const { return code == CandidateRankingResultCode::kSuccess; }
};

class CandidateRankingServiceInterface {
 public:
  virtual ~CandidateRankingServiceInterface() = default;

  virtual bool HasBackend() const = 0;
  virtual bool IsAvailable() const = 0;
  virtual CandidateRankingResult RankUntil(
      converter::CandidateRankerRequest request, absl::Duration max_wait) = 0;
  virtual void Invalidate(
      const converter::CandidateRankerToken& live_token) = 0;
  // Terminal for this unique generation. The caller must not submit or
  // invalidate work for it afterward.
  virtual void CancelSession(uint64_t session_generation) = 0;
  virtual void Shutdown() = 0;
};

class CandidateRankingServiceTestPeer;

// Owns one persistent worker and serializes every backend invocation on it.
class CandidateRankingService final : public CandidateRankingServiceInterface {
 public:
  explicit CandidateRankingService(
      std::shared_ptr<converter::CandidateRankerBackendInterface> backend);
  ~CandidateRankingService() override;

  CandidateRankingService(const CandidateRankingService&) = delete;
  CandidateRankingService& operator=(const CandidateRankingService&) = delete;

  bool HasBackend() const override;
  bool IsAvailable() const override;

  CandidateRankingResult RankUntil(converter::CandidateRankerRequest request,
                                   absl::Duration max_wait) override;

  // Cancels stale work for the live token's session generation.
  void Invalidate(const converter::CandidateRankerToken& live_token) override;
  void CancelSession(uint64_t session_generation) override;

  // Stops admission, cancels outstanding work, and joins the worker.
  void Shutdown() override;

 private:
  friend class CandidateRankingServiceTestPeer;
  struct State;

  void WaitUntilPendingForTesting(
      const converter::CandidateRankerToken& token) const;
  bool HasWorkerForTesting() const;

  std::unique_ptr<State> state_;
};

}  // namespace engine
}  // namespace mozc

#endif  // MOZC_ENGINE_CANDIDATE_RANKING_SERVICE_H_
