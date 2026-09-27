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

#include "engine/candidate_ranking_service.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "base/thread.h"
#include "converter/candidate_ranker.h"

namespace mozc {
namespace engine {
namespace {

class ServiceCancellation final
    : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override {
    return requested_.load(std::memory_order_acquire);
  }

  void Cancel() { requested_.store(true, std::memory_order_release); }

 private:
  std::atomic<bool> requested_ = false;
};

CandidateRankingResult ImmediateResult(CandidateRankingResultCode code,
                                       absl::Time start) {
  return CandidateRankingResult{
      .code = code,
      .elapsed = absl::Now() - start,
  };
}

bool IsOlder(const converter::CandidateRankerToken& token,
             const converter::CandidateRankerToken& live_token) {
  return token.state_revision < live_token.state_revision ||
         (token.state_revision == live_token.state_revision &&
          token.request_sequence < live_token.request_sequence);
}

}  // namespace

struct CandidateRankingService::State {
  struct Job {
    converter::CandidateRankerRequest request;
    absl::Time start;
    absl::Time deadline;
    std::shared_ptr<ServiceCancellation> cancellation =
        std::make_shared<ServiceCancellation>();
    bool finished = false;
    CandidateRankingResult result;
    absl::CondVar done;
  };

  explicit State(
      std::shared_ptr<converter::CandidateRankerBackendInterface> backend)
      : backend(std::move(backend)) {
    if (this->backend != nullptr) {
      worker = std::make_unique<Thread>([this] { WorkerLoop(); });
    }
  }

  struct SessionLiveState {
    converter::CandidateRankerToken token;
    bool has_token = false;
  };

  bool HasBackend() const { return backend != nullptr; }

  bool IsAvailable() const {
    absl::MutexLock lock(&mutex);
    return backend != nullptr && !shutdown_started;
  }

  void Complete(const std::shared_ptr<Job>& job,
                CandidateRankingResultCode code)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    if (job->finished) {
      return;
    }
    job->result.code = code;
    job->result.elapsed = absl::Now() - job->start;
    job->finished = true;
    job->done.SignalAll();
    state_changed.SignalAll();
  }

  void CancelAndComplete(const std::shared_ptr<Job>& job,
                         CandidateRankingResultCode code)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    job->cancellation->Cancel();
    Complete(job, code);
  }

  void RemovePending(const std::shared_ptr<Job>& job)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    for (auto it = pending.begin(); it != pending.end(); ++it) {
      if (*it == job) {
        pending.erase(it);
        state_changed.SignalAll();
        return;
      }
    }
  }

  CandidateRankingResult RankUntil(converter::CandidateRankerRequest request,
                                   absl::Duration max_wait) {
    const absl::Time start = absl::Now();
    auto job = std::make_shared<Job>();
    job->request = std::move(request);
    job->start = start;
    job->deadline = start + max_wait;

    mutex.Lock();
    if (shutdown_started) {
      mutex.Unlock();
      return ImmediateResult(CandidateRankingResultCode::kShutdown, start);
    }
    if (backend == nullptr) {
      mutex.Unlock();
      return ImmediateResult(CandidateRankingResultCode::kBackendMissing,
                             start);
    }
    if (max_wait <= absl::ZeroDuration()) {
      mutex.Unlock();
      return ImmediateResult(CandidateRankingResultCode::kTimedOut, start);
    }

    const uint64_t session_generation = job->request.token.session_generation;
    SessionLiveState& live_state = sessions[session_generation];
    if (live_state.has_token && IsOlder(job->request.token, live_state.token)) {
      mutex.Unlock();
      return ImmediateResult(CandidateRankingResultCode::kSuperseded, start);
    }
    if (!live_state.has_token ||
        IsOlder(live_state.token, job->request.token)) {
      live_state.token = job->request.token;
      live_state.has_token = true;
    }
    bool replaced_pending = false;
    for (std::shared_ptr<Job>& pending_job : pending) {
      if (pending_job->request.token.session_generation == session_generation) {
        CancelAndComplete(pending_job, CandidateRankingResultCode::kSuperseded);
        pending_job = job;
        replaced_pending = true;
        break;
      }
    }
    if (!replaced_pending) {
      pending.push_back(job);
    }
    if (in_flight != nullptr &&
        in_flight->request.token.session_generation == session_generation) {
      CancelAndComplete(in_flight, CandidateRankingResultCode::kSuperseded);
    }
    work_available.Signal();
    state_changed.SignalAll();

    while (!job->finished) {
      if (job->done.WaitWithDeadline(&mutex, job->deadline) && !job->finished) {
        RemovePending(job);
        CancelAndComplete(job, CandidateRankingResultCode::kTimedOut);
      }
    }
    CandidateRankingResult result = std::move(job->result);
    mutex.Unlock();
    return result;
  }

  void Invalidate(const converter::CandidateRankerToken& live_token) {
    absl::MutexLock lock(&mutex);
    SessionLiveState& live_state = sessions[live_token.session_generation];
    if (!live_state.has_token || IsOlder(live_state.token, live_token)) {
      live_state.token = live_token;
      live_state.has_token = true;
    }
    for (auto it = pending.begin(); it != pending.end();) {
      const std::shared_ptr<Job>& job = *it;
      if (job->request.token.session_generation ==
              live_token.session_generation &&
          IsOlder(job->request.token, live_state.token)) {
        CancelAndComplete(job, CandidateRankingResultCode::kSuperseded);
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
    if (in_flight != nullptr &&
        in_flight->request.token.session_generation ==
            live_token.session_generation &&
        IsOlder(in_flight->request.token, live_state.token)) {
      CancelAndComplete(in_flight, CandidateRankingResultCode::kSuperseded);
    }
    state_changed.SignalAll();
  }

  void CancelSession(uint64_t session_generation) {
    absl::MutexLock lock(&mutex);
    for (auto it = pending.begin(); it != pending.end();) {
      const std::shared_ptr<Job>& job = *it;
      if (job->request.token.session_generation == session_generation) {
        CancelAndComplete(job, CandidateRankingResultCode::kSessionCancelled);
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
    if (in_flight != nullptr &&
        in_flight->request.token.session_generation == session_generation) {
      CancelAndComplete(in_flight,
                        CandidateRankingResultCode::kSessionCancelled);
    }
    sessions.erase(session_generation);
    state_changed.SignalAll();
  }

  void Shutdown() {
    mutex.Lock();
    if (worker_joined) {
      mutex.Unlock();
      return;
    }
    if (join_started) {
      while (!worker_joined) {
        join_finished.Wait(&mutex);
      }
      mutex.Unlock();
      return;
    }

    join_started = true;
    shutdown_started = true;
    for (const std::shared_ptr<Job>& job : pending) {
      CancelAndComplete(job, CandidateRankingResultCode::kShutdown);
    }
    pending.clear();
    if (in_flight != nullptr) {
      CancelAndComplete(in_flight, CandidateRankingResultCode::kShutdown);
    }
    work_available.SignalAll();
    state_changed.SignalAll();
    if (worker == nullptr) {
      worker_joined = true;
      join_finished.SignalAll();
      mutex.Unlock();
      return;
    }
    mutex.Unlock();

    worker->Join();

    mutex.Lock();
    worker_joined = true;
    join_finished.SignalAll();
    mutex.Unlock();
  }

  void WaitUntilPendingForTesting(
      const converter::CandidateRankerToken& token) const {
    mutex.Lock();
    for (;;) {
      for (const std::shared_ptr<Job>& job : pending) {
        if (job->request.token == token) {
          mutex.Unlock();
          return;
        }
      }
      state_changed.Wait(&mutex);
    }
  }

  bool HasWorkerForTesting() const { return worker != nullptr; }

  void WorkerLoop() {
    for (;;) {
      mutex.Lock();
      while (!shutdown_started && pending.empty()) {
        work_available.Wait(&mutex);
      }
      if (shutdown_started) {
        mutex.Unlock();
        return;
      }

      const std::shared_ptr<Job> job = pending.front();
      pending.pop_front();
      state_changed.SignalAll();
      if (absl::Now() >= job->deadline) {
        CancelAndComplete(job, CandidateRankingResultCode::kTimedOut);
        mutex.Unlock();
        continue;
      }
      in_flight = job;
      mutex.Unlock();

      absl::StatusOr<converter::CandidateRankerResponse> backend_result =
          backend->Rank(job->request, *job->cancellation);

      mutex.Lock();
      if (in_flight == job) {
        in_flight.reset();
      }
      if (!job->finished) {
        if (absl::Now() >= job->deadline) {
          CancelAndComplete(job, CandidateRankingResultCode::kTimedOut);
        } else if (!backend_result.ok()) {
          LOG(ERROR) << "Candidate ranking backend failed with status code "
                     << static_cast<int>(backend_result.status().code());
          Complete(job, CandidateRankingResultCode::kBackendError);
        } else if (backend_result->token != job->request.token) {
          Complete(job, CandidateRankingResultCode::kInvalidResponse);
        } else {
          job->result.code = CandidateRankingResultCode::kSuccess;
          job->result.response = std::move(backend_result).value();
          job->result.elapsed = absl::Now() - job->start;
          job->finished = true;
          job->done.SignalAll();
          state_changed.SignalAll();
        }
      }
      mutex.Unlock();
    }
  }

  const std::shared_ptr<converter::CandidateRankerBackendInterface> backend;
  mutable absl::Mutex mutex;
  mutable absl::CondVar state_changed;
  absl::CondVar work_available;
  absl::CondVar join_finished;
  std::deque<std::shared_ptr<Job>> pending ABSL_GUARDED_BY(mutex);
  std::shared_ptr<Job> in_flight ABSL_GUARDED_BY(mutex);
  std::map<uint64_t, SessionLiveState> sessions ABSL_GUARDED_BY(mutex);
  bool shutdown_started ABSL_GUARDED_BY(mutex) = false;
  bool join_started ABSL_GUARDED_BY(mutex) = false;
  bool worker_joined ABSL_GUARDED_BY(mutex) = false;
  std::unique_ptr<Thread> worker;
};

CandidateRankingService::CandidateRankingService(
    std::shared_ptr<converter::CandidateRankerBackendInterface> backend)
    : state_(std::make_unique<State>(std::move(backend))) {}

CandidateRankingService::~CandidateRankingService() { Shutdown(); }

bool CandidateRankingService::HasBackend() const {
  return state_->HasBackend();
}

bool CandidateRankingService::IsAvailable() const {
  return state_->IsAvailable();
}

CandidateRankingResult CandidateRankingService::RankUntil(
    converter::CandidateRankerRequest request, absl::Duration max_wait) {
  return state_->RankUntil(std::move(request), max_wait);
}

void CandidateRankingService::Invalidate(
    const converter::CandidateRankerToken& live_token) {
  state_->Invalidate(live_token);
}

void CandidateRankingService::CancelSession(uint64_t session_generation) {
  state_->CancelSession(session_generation);
}

void CandidateRankingService::Shutdown() { state_->Shutdown(); }

void CandidateRankingService::WaitUntilPendingForTesting(
    const converter::CandidateRankerToken& token) const {
  state_->WaitUntilPendingForTesting(token);
}

bool CandidateRankingService::HasWorkerForTesting() const {
  return state_->HasWorkerForTesting();
}

}  // namespace engine
}  // namespace mozc
