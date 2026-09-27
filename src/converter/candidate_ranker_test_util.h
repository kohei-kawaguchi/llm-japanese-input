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

#ifndef MOZC_CONVERTER_CANDIDATE_RANKER_TEST_UTIL_H_
#define MOZC_CONVERTER_CANDIDATE_RANKER_TEST_UTIL_H_

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "converter/candidate_ranker.h"

namespace mozc {
namespace converter {

class RecordingCandidateRanker final : public CandidateRankerBackendInterface {
 public:
  absl::StatusOr<CandidateRankerResponse> Rank(
      const CandidateRankerRequest& request,
      const CandidateRankerCancellation& cancellation) override {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking cancelled");
    }
    requests_.push_back(request);
    if (!status_.ok()) {
      return status_;
    }

    CandidateRankerResponse response{.token = request.token};
    response.segment_orders.reserve(request.segments.size());
    for (const CandidateRankerSegment& segment : request.segments) {
      CandidateRankerSegmentOrder& order =
          response.segment_orders.emplace_back();
      order.segment_id = segment.id;
      for (auto it = segment.candidates.rbegin();
           it != segment.candidates.rend(); ++it) {
        if (!it->is_protected) {
          order.candidate_ids.push_back(it->id);
        }
      }
    }
    return response;
  }

  void set_status(absl::Status status) { status_ = std::move(status); }

  const std::vector<CandidateRankerRequest>& requests() const {
    return requests_;
  }

 private:
  absl::Status status_ = absl::OkStatus();
  std::vector<CandidateRankerRequest> requests_;
};

}  // namespace converter
}  // namespace mozc

#endif  // MOZC_CONVERTER_CANDIDATE_RANKER_TEST_UTIL_H_
