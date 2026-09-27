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
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "base/thread.h"
#include "converter/candidate_ranker.h"
#include "testing/gunit.h"

namespace mozc {
namespace engine {

class CandidateRankingServiceTestPeer {
 public:
  static void WaitUntilPending(CandidateRankingService& service,
                               const converter::CandidateRankerToken& token) {
    service.WaitUntilPendingForTesting(token);
  }

  static bool HasWorker(const CandidateRankingService& service) {
    return service.HasWorkerForTesting();
  }
};

namespace {
using converter::CandidateRankerBackendInterface;
using converter::CandidateRankerCancellation;
using converter::CandidateRankerRequest;
using converter::CandidateRankerResponse;
using converter::CandidateRankerToken;

CandidateRankerRequest MakeRequest(uint64_t session_generation,
                                   uint64_t state_revision,
                                   uint64_t request_sequence) {
  CandidateRankerRequest request;
  request.token = CandidateRankerToken{
      .session_generation = session_generation,
      .state_revision = state_revision,
      .request_sequence = request_sequence,
  };
  return request;
}

class ImmediateBackend final : public CandidateRankerBackendInterface {
 public:
  enum class Behavior {
    kSuccess,
    kError,
    kWrongToken,
  };

  explicit ImmediateBackend(Behavior behavior = Behavior::kSuccess)
      : behavior_(behavior) {}

  absl::StatusOr<CandidateRankerResponse> Rank(
      const CandidateRankerRequest& request,
      const CandidateRankerCancellation& cancellation) override {
    thread_ids_.push_back(std::this_thread::get_id());
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("cancelled");
    }
    if (behavior_ == Behavior::kError) {
      return absl::InternalError("backend detail must not escape service");
    }
    CandidateRankerResponse response{.token = request.token};
    if (behavior_ == Behavior::kWrongToken) {
      ++response.token.request_sequence;
    }
    return response;
  }

  const std::vector<std::thread::id>& thread_ids() const { return thread_ids_; }

 private:
  const Behavior behavior_;
  std::vector<std::thread::id> thread_ids_;
};

struct BackendStep {
  absl::Notification started;
  absl::Notification release;
  CandidateRankerToken actual_token;
  std::thread::id thread_id;
  bool cancellation_seen = false;
};

class ScriptedBackend final : public CandidateRankerBackendInterface {
 public:
  explicit ScriptedBackend(std::vector<BackendStep*> steps)
      : steps_(std::move(steps)) {}

  absl::StatusOr<CandidateRankerResponse> Rank(
      const CandidateRankerRequest& request,
      const CandidateRankerCancellation& cancellation) override {
    const size_t index = call_count_.fetch_add(1, std::memory_order_relaxed);
    BackendStep& step = *steps_[index];
    const int active = active_.fetch_add(1, std::memory_order_acq_rel) + 1;
    int recorded_max = max_active_.load(std::memory_order_relaxed);
    while (active > recorded_max &&
           !max_active_.compare_exchange_weak(recorded_max, active)) {
    }
    step.actual_token = request.token;
    step.thread_id = std::this_thread::get_id();
    step.started.Notify();
    step.release.WaitForNotification();
    step.cancellation_seen = cancellation.IsCancellationRequested();
    active_.fetch_sub(1, std::memory_order_acq_rel);
    return CandidateRankerResponse{.token = request.token};
  }

  size_t call_count() const {
    return call_count_.load(std::memory_order_relaxed);
  }
  int max_active() const { return max_active_.load(std::memory_order_relaxed); }

 private:
  const std::vector<BackendStep*> steps_;
  std::atomic<size_t> call_count_ = 0;
  std::atomic<int> active_ = 0;
  std::atomic<int> max_active_ = 0;
};

TEST(CandidateRankingServiceTest, UsesWorkerAndReturnsStableResultCodes) {
  const std::thread::id caller_thread = std::this_thread::get_id();
  auto backend = std::make_shared<ImmediateBackend>();
  CandidateRankingService service(backend);
  EXPECT_TRUE(service.HasBackend());
  EXPECT_TRUE(service.IsAvailable());
  EXPECT_TRUE(CandidateRankingServiceTestPeer::HasWorker(service));

  CandidateRankingResult first =
      service.RankUntil(MakeRequest(1, 1, 1), absl::Seconds(5));
  CandidateRankingResult second =
      service.RankUntil(MakeRequest(2, 1, 1), absl::Seconds(5));
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(first.response.has_value());
  ASSERT_TRUE(second.response.has_value());
  ASSERT_EQ(backend->thread_ids().size(), 2);
  EXPECT_NE(backend->thread_ids()[0], caller_thread);
  EXPECT_EQ(backend->thread_ids()[0], backend->thread_ids()[1]);

  CandidateRankingService missing_backend(nullptr);
  EXPECT_FALSE(missing_backend.HasBackend());
  EXPECT_FALSE(CandidateRankingServiceTestPeer::HasWorker(missing_backend));
  EXPECT_EQ(
      missing_backend.RankUntil(MakeRequest(3, 1, 1), absl::Seconds(5)).code,
      CandidateRankingResultCode::kBackendMissing);

  CandidateRankingService failing_service(
      std::make_shared<ImmediateBackend>(ImmediateBackend::Behavior::kError));
  EXPECT_EQ(
      failing_service.RankUntil(MakeRequest(4, 1, 1), absl::Seconds(5)).code,
      CandidateRankingResultCode::kBackendError);

  CandidateRankingService invalid_service(std::make_shared<ImmediateBackend>(
      ImmediateBackend::Behavior::kWrongToken));
  EXPECT_EQ(
      invalid_service.RankUntil(MakeRequest(5, 1, 1), absl::Seconds(5)).code,
      CandidateRankingResultCode::kInvalidResponse);
}

TEST(CandidateRankingServiceTest,
     ReplacesSameSessionWorkWithoutReorderingOtherSessions) {
  BackendStep first_a;
  BackendStep b;
  BackendStep latest_a;
  auto backend = std::make_shared<ScriptedBackend>(std::vector<BackendStep*>{
      &first_a,
      &b,
      &latest_a,
  });
  CandidateRankingService service(backend);
  const CandidateRankerRequest a1 = MakeRequest(10, 1, 1);
  const CandidateRankerRequest b1 = MakeRequest(20, 1, 1);
  const CandidateRankerRequest a2 = MakeRequest(10, 2, 2);
  const CandidateRankerRequest a3 = MakeRequest(10, 3, 3);
  CandidateRankingResult a1_result;
  CandidateRankingResult b_result;
  CandidateRankingResult a2_result;
  CandidateRankingResult a3_result;

  Thread a1_thread(
      [&] { a1_result = service.RankUntil(a1, absl::Seconds(30)); });
  first_a.started.WaitForNotification();

  Thread b_thread([&] { b_result = service.RankUntil(b1, absl::Seconds(30)); });
  CandidateRankingServiceTestPeer::WaitUntilPending(service, b1.token);

  Thread a2_thread(
      [&] { a2_result = service.RankUntil(a2, absl::Seconds(30)); });
  CandidateRankingServiceTestPeer::WaitUntilPending(service, a2.token);
  a1_thread.Join();
  EXPECT_EQ(a1_result.code, CandidateRankingResultCode::kSuperseded);

  Thread a3_thread(
      [&] { a3_result = service.RankUntil(a3, absl::Seconds(30)); });
  CandidateRankingServiceTestPeer::WaitUntilPending(service, a3.token);
  a2_thread.Join();
  EXPECT_EQ(a2_result.code, CandidateRankingResultCode::kSuperseded);

  first_a.release.Notify();
  b.started.WaitForNotification();
  EXPECT_EQ(first_a.actual_token, a1.token);
  EXPECT_EQ(b.actual_token, b1.token);
  b.release.Notify();
  b_thread.Join();
  EXPECT_TRUE(b_result.ok());

  latest_a.started.WaitForNotification();
  EXPECT_EQ(latest_a.actual_token, a3.token);
  latest_a.release.Notify();
  a3_thread.Join();
  EXPECT_TRUE(a3_result.ok());
  EXPECT_TRUE(first_a.cancellation_seen);
  EXPECT_EQ(backend->call_count(), 3);
  EXPECT_EQ(backend->max_active(), 1);
  EXPECT_EQ(first_a.thread_id, b.thread_id);
  EXPECT_EQ(b.thread_id, latest_a.thread_id);
}

TEST(CandidateRankingServiceTest, InvalidatesLiveStateAndCancelsSession) {
  BackendStep running;
  auto backend =
      std::make_shared<ScriptedBackend>(std::vector<BackendStep*>{&running});
  CandidateRankingService service(backend);
  const CandidateRankerRequest stale = MakeRequest(30, 1, 1);
  const CandidateRankerRequest other = MakeRequest(40, 1, 1);
  CandidateRankingResult stale_result;
  CandidateRankingResult other_result;

  Thread stale_thread(
      [&] { stale_result = service.RankUntil(stale, absl::Seconds(30)); });
  running.started.WaitForNotification();
  Thread other_thread(
      [&] { other_result = service.RankUntil(other, absl::Seconds(30)); });
  CandidateRankingServiceTestPeer::WaitUntilPending(service, other.token);

  CandidateRankerToken live = stale.token;
  ++live.state_revision;
  service.Invalidate(live);
  service.CancelSession(other.token.session_generation);
  stale_thread.Join();
  other_thread.Join();
  EXPECT_EQ(stale_result.code, CandidateRankingResultCode::kSuperseded);
  EXPECT_EQ(other_result.code, CandidateRankingResultCode::kSessionCancelled);

  const CandidateRankingResult late_stale =
      service.RankUntil(stale, absl::Seconds(30));
  EXPECT_EQ(late_stale.code, CandidateRankingResultCode::kSuperseded);

  running.release.Notify();
  service.Shutdown();
  EXPECT_TRUE(running.cancellation_seen);
  EXPECT_EQ(backend->call_count(), 1);
}

TEST(CandidateRankingServiceTest, OlderInvalidationCannotCancelNewerLiveWork) {
  BackendStep running;
  auto backend =
      std::make_shared<ScriptedBackend>(std::vector<BackendStep*>{&running});
  CandidateRankingService service(backend);
  const CandidateRankerRequest old_request = MakeRequest(45, 4, 4);
  const CandidateRankerRequest live_request = MakeRequest(45, 5, 5);
  service.Invalidate(live_request.token);

  CandidateRankingResult live_result;
  Thread live_thread([&] {
    live_result = service.RankUntil(live_request, absl::Seconds(30));
  });
  running.started.WaitForNotification();
  service.Invalidate(old_request.token);
  EXPECT_EQ(service.RankUntil(old_request, absl::Seconds(30)).code,
            CandidateRankingResultCode::kSuperseded);

  running.release.Notify();
  live_thread.Join();
  EXPECT_TRUE(live_result.ok());
  EXPECT_FALSE(running.cancellation_seen);
  EXPECT_EQ(backend->call_count(), 1);
}

TEST(CandidateRankingServiceTest, DeadlineIncludesQueueWait) {
  BackendStep queue_blocker;
  auto queue_backend = std::make_shared<ScriptedBackend>(
      std::vector<BackendStep*>{&queue_blocker});
  CandidateRankingService queue_service(queue_backend);
  CandidateRankingResult blocker_result;
  Thread blocker_thread([&] {
    blocker_result =
        queue_service.RankUntil(MakeRequest(50, 1, 1), absl::Seconds(30));
  });
  queue_blocker.started.WaitForNotification();
  const CandidateRankingResult queued_result =
      queue_service.RankUntil(MakeRequest(60, 1, 1), absl::Milliseconds(200));
  EXPECT_EQ(queued_result.code, CandidateRankingResultCode::kTimedOut);
  EXPECT_EQ(queue_backend->call_count(), 1);
  queue_blocker.release.Notify();
  blocker_thread.Join();
  EXPECT_TRUE(blocker_result.ok());
}

TEST(CandidateRankingServiceTest, DeadlineIncludesInference) {
  BackendStep inference_blocker;
  auto inference_backend = std::make_shared<ScriptedBackend>(
      std::vector<BackendStep*>{&inference_blocker});
  CandidateRankingService inference_service(inference_backend);
  CandidateRankingResult inference_result;
  Thread inference_thread([&] {
    inference_result = inference_service.RankUntil(MakeRequest(70, 1, 1),
                                                   absl::Milliseconds(200));
  });
  ASSERT_TRUE(inference_blocker.started.WaitForNotificationWithTimeout(
      absl::Seconds(5)));
  inference_thread.Join();
  EXPECT_EQ(inference_result.code, CandidateRankingResultCode::kTimedOut);
  inference_blocker.release.Notify();
  inference_service.Shutdown();
  EXPECT_TRUE(inference_blocker.cancellation_seen);
}

TEST(CandidateRankingServiceTest, ShutdownCancelsWakesAndJoinsOnce) {
  BackendStep running;
  auto backend =
      std::make_shared<ScriptedBackend>(std::vector<BackendStep*>{&running});
  CandidateRankingService service(backend);
  CandidateRankingResult rank_result;
  Thread rank_thread([&] {
    rank_result = service.RankUntil(MakeRequest(80, 1, 1), absl::Seconds(30));
  });
  running.started.WaitForNotification();

  Thread shutdown_thread([&] { service.Shutdown(); });
  rank_thread.Join();
  EXPECT_EQ(rank_result.code, CandidateRankingResultCode::kShutdown);
  running.release.Notify();
  shutdown_thread.Join();
  EXPECT_TRUE(running.cancellation_seen);
  EXPECT_FALSE(service.IsAvailable());

  service.Shutdown();
  EXPECT_EQ(service.RankUntil(MakeRequest(80, 2, 2), absl::Seconds(30)).code,
            CandidateRankingResultCode::kShutdown);
}

}  // namespace
}  // namespace engine
}  // namespace mozc
