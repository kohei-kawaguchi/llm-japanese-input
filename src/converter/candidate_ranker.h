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

#ifndef MOZC_CONVERTER_CANDIDATE_RANKER_H_
#define MOZC_CONVERTER_CANDIDATE_RANKER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace mozc {
namespace converter {

class Segments;

enum class CandidateRankerMode : uint8_t {
  kSuggestion,
  kPrediction,
  kConversion,
};

struct CandidateRankerToken {
  uint64_t session_generation = 0;
  uint64_t state_revision = 0;
  uint64_t request_sequence = 0;

  friend bool operator==(const CandidateRankerToken& lhs,
                         const CandidateRankerToken& rhs) {
    return lhs.session_generation == rhs.session_generation &&
           lhs.state_revision == rhs.state_revision &&
           lhs.request_sequence == rhs.request_sequence;
  }
  friend bool operator!=(const CandidateRankerToken& lhs,
                         const CandidateRankerToken& rhs) {
    return !(lhs == rhs);
  }
};

using CandidateRankerSegmentId = uint64_t;
using CandidateRankerCandidateId = uint64_t;

struct CandidateRankerCandidate {
  CandidateRankerCandidateId id = 0;
  std::string key;
  std::string value;
  int32_t cost = 0;
  uint32_t attributes = 0;
  uint32_t consumed_key_size = 0;
  bool is_protected = false;
};

struct CandidateRankerSegment {
  CandidateRankerSegmentId id = 0;
  std::string key;
  std::vector<CandidateRankerCandidate> candidates;
};

struct CandidateRankerRequest {
  CandidateRankerToken token;
  CandidateRankerMode mode = CandidateRankerMode::kConversion;
  std::string preceding_text;
  std::string following_text;
  std::string reading;
  CandidateRankerSegmentId focused_segment_id = 0;
  std::vector<CandidateRankerSegment> segments;
};

struct CandidateRankerSegmentOrder {
  CandidateRankerSegmentId segment_id = 0;
  std::vector<CandidateRankerCandidateId> candidate_ids;
};

struct CandidateRankerResponse {
  CandidateRankerToken token;
  std::vector<CandidateRankerSegmentOrder> segment_orders;
};

struct CandidateRankerMergeOrder {
  std::vector<CandidateRankerSegmentOrder> segment_orders;
};

class CandidateRankerCancellation {
 public:
  virtual ~CandidateRankerCancellation() = default;
  virtual bool IsCancellationRequested() const = 0;
};

class CandidateRankerBackendInterface {
 public:
  virtual ~CandidateRankerBackendInterface() = default;

  virtual absl::StatusOr<CandidateRankerResponse> Rank(
      const CandidateRankerRequest& request,
      const CandidateRankerCancellation& cancellation) = 0;
};

CandidateRankerRequest BuildCandidateRankerRequest(
    const CandidateRankerToken& token, CandidateRankerMode mode,
    absl::string_view reading, absl::string_view preceding_text,
    absl::string_view following_text,
    CandidateRankerSegmentId focused_segment_id, const Segments& segments);

// Returns the complete candidate-ID order for every request segment. Protected
// candidates remain in their original positions, and candidates omitted from
// `response` retain their relative order after explicitly ranked candidates.
absl::StatusOr<CandidateRankerMergeOrder> BuildCandidateRankerMergeOrder(
    const CandidateRankerRequest& request,
    const CandidateRankerResponse& response,
    const CandidateRankerToken& current_token);

// Applies `response` only if its token and the snapshot in `request` still
// match the current conversion state. Validation is transactional: an error
// leaves `segments` unchanged.
absl::Status ApplyCandidateRankerResponse(
    const CandidateRankerRequest& request,
    const CandidateRankerResponse& response,
    const CandidateRankerToken& current_token, Segments* segments);

}  // namespace converter
}  // namespace mozc

#endif  // MOZC_CONVERTER_CANDIDATE_RANKER_H_
