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

#include "converter/candidate_ranker.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/attribute.h"
#include "converter/candidate.h"
#include "converter/segments.h"

namespace mozc {
namespace converter {
namespace {

constexpr uint32_t kProtectedAttributes = Attribute::NO_MODIFICATION |
                                          Attribute::COMMAND_CANDIDATE |
                                          Attribute::DISABLE_RESCORING;

bool IsProtected(const Candidate& candidate) {
  return (candidate.attributes & kProtectedAttributes) != 0 ||
         candidate.command != Candidate::DEFAULT_COMMAND;
}

absl::Status ValidateTokens(const CandidateRankerRequest& request,
                            const CandidateRankerResponse& response,
                            const CandidateRankerToken& current_token) {
  if (request.token != current_token || response.token != request.token) {
    return absl::AbortedError("candidate ranking token is stale");
  }
  return absl::OkStatus();
}

absl::Status ValidateSnapshot(const CandidateRankerRequest& request,
                              const Segments& segments) {
  if (segments.conversion_segments_size() != request.segments.size()) {
    return absl::FailedPreconditionError(
        "conversion segment count no longer matches ranker snapshot");
  }

  CandidateRankerCandidateId expected_candidate_id = 0;
  for (size_t segment_index = 0; segment_index < request.segments.size();
       ++segment_index) {
    const CandidateRankerSegment& snapshot_segment =
        request.segments[segment_index];
    const Segment& segment = segments.conversion_segment(segment_index);
    if (snapshot_segment.id != segment_index ||
        snapshot_segment.key != segment.key() ||
        snapshot_segment.candidates.size() != segment.candidates_size()) {
      return absl::FailedPreconditionError(
          "conversion segment no longer matches ranker snapshot");
    }

    for (size_t candidate_index = 0;
         candidate_index < snapshot_segment.candidates.size();
         ++candidate_index) {
      const CandidateRankerCandidate& snapshot_candidate =
          snapshot_segment.candidates[candidate_index];
      const Candidate& candidate = segment.candidate(candidate_index);
      if (snapshot_candidate.id != expected_candidate_id ||
          snapshot_candidate.key != candidate.key ||
          snapshot_candidate.value != candidate.value ||
          snapshot_candidate.cost != candidate.cost ||
          snapshot_candidate.attributes != candidate.attributes ||
          snapshot_candidate.consumed_key_size != candidate.consumed_key_size ||
          snapshot_candidate.is_protected != IsProtected(candidate)) {
        return absl::FailedPreconditionError(
            "candidate no longer matches ranker snapshot");
      }
      ++expected_candidate_id;
    }
  }

  return absl::OkStatus();
}

struct SegmentMergePlan {
  Segment* segment = nullptr;
  std::vector<Candidate*> target_order;
};

}  // namespace

CandidateRankerRequest BuildCandidateRankerRequest(
    const CandidateRankerToken& token, CandidateRankerMode mode,
    absl::string_view reading, absl::string_view preceding_text,
    absl::string_view following_text,
    CandidateRankerSegmentId focused_segment_id, const Segments& segments) {
  CandidateRankerRequest request;
  request.token = token;
  request.mode = mode;
  request.reading = std::string(reading);
  request.preceding_text = std::string(preceding_text);
  request.following_text = std::string(following_text);
  request.focused_segment_id = focused_segment_id;
  request.segments.reserve(segments.conversion_segments_size());

  CandidateRankerCandidateId next_candidate_id = 0;
  for (size_t segment_index = 0;
       segment_index < segments.conversion_segments_size(); ++segment_index) {
    const Segment& segment = segments.conversion_segment(segment_index);
    CandidateRankerSegment& snapshot_segment = request.segments.emplace_back();
    snapshot_segment.id = segment_index;
    snapshot_segment.key = std::string(segment.key());
    snapshot_segment.candidates.reserve(segment.candidates_size());
    for (size_t candidate_index = 0;
         candidate_index < segment.candidates_size(); ++candidate_index) {
      const Candidate& candidate = segment.candidate(candidate_index);
      CandidateRankerCandidate& snapshot_candidate =
          snapshot_segment.candidates.emplace_back();
      snapshot_candidate.id = next_candidate_id++;
      snapshot_candidate.key = candidate.key;
      snapshot_candidate.value = candidate.value;
      snapshot_candidate.cost = candidate.cost;
      snapshot_candidate.attributes = candidate.attributes;
      snapshot_candidate.consumed_key_size = candidate.consumed_key_size;
      snapshot_candidate.is_protected = IsProtected(candidate);
    }
  }
  return request;
}

absl::StatusOr<CandidateRankerMergeOrder> BuildCandidateRankerMergeOrder(
    const CandidateRankerRequest& request,
    const CandidateRankerResponse& response,
    const CandidateRankerToken& current_token) {
  const absl::Status token_status =
      ValidateTokens(request, response, current_token);
  if (!token_status.ok()) {
    return token_status;
  }

  CandidateRankerMergeOrder merge_order;
  merge_order.segment_orders.reserve(request.segments.size());
  for (const CandidateRankerSegment& segment : request.segments) {
    CandidateRankerSegmentOrder& original_order =
        merge_order.segment_orders.emplace_back();
    original_order.segment_id = segment.id;
    original_order.candidate_ids.reserve(segment.candidates.size());
    for (const CandidateRankerCandidate& candidate : segment.candidates) {
      original_order.candidate_ids.push_back(candidate.id);
    }
  }

  std::vector<CandidateRankerSegmentId> seen_segment_ids;
  for (const CandidateRankerSegmentOrder& segment_order :
       response.segment_orders) {
    for (CandidateRankerSegmentId seen_id : seen_segment_ids) {
      if (seen_id == segment_order.segment_id) {
        return absl::AlreadyExistsError(
            "candidate ranking references a segment more than once");
      }
    }
    seen_segment_ids.push_back(segment_order.segment_id);

    size_t segment_index = request.segments.size();
    for (size_t i = 0; i < request.segments.size(); ++i) {
      if (request.segments[i].id == segment_order.segment_id) {
        segment_index = i;
        break;
      }
    }
    if (segment_index == request.segments.size()) {
      return absl::NotFoundError(
          "candidate ranking references an unknown segment");
    }

    const CandidateRankerSegment& snapshot_segment =
        request.segments[segment_index];
    std::vector<CandidateRankerCandidateId> seen_candidate_ids;
    std::vector<CandidateRankerCandidateId> ranked_candidate_ids;
    ranked_candidate_ids.reserve(snapshot_segment.candidates.size());
    for (CandidateRankerCandidateId candidate_id :
         segment_order.candidate_ids) {
      size_t candidate_index = snapshot_segment.candidates.size();
      for (size_t i = 0; i < snapshot_segment.candidates.size(); ++i) {
        if (snapshot_segment.candidates[i].id == candidate_id) {
          candidate_index = i;
          break;
        }
      }
      if (candidate_index == snapshot_segment.candidates.size()) {
        return absl::NotFoundError(
            "candidate ranking references an unknown candidate");
      }
      for (CandidateRankerCandidateId seen_id : seen_candidate_ids) {
        if (seen_id == candidate_id) {
          return absl::AlreadyExistsError(
              "candidate ranking references a candidate more than once");
        }
      }
      if (snapshot_segment.candidates[candidate_index].is_protected) {
        return absl::PermissionDeniedError(
            "candidate ranking references a protected candidate");
      }
      seen_candidate_ids.push_back(candidate_id);
      ranked_candidate_ids.push_back(candidate_id);
    }

    for (const CandidateRankerCandidate& candidate :
         snapshot_segment.candidates) {
      if (candidate.is_protected) {
        continue;
      }
      bool was_ranked = false;
      for (CandidateRankerCandidateId seen_id : seen_candidate_ids) {
        if (seen_id == candidate.id) {
          was_ranked = true;
          break;
        }
      }
      if (!was_ranked) {
        ranked_candidate_ids.push_back(candidate.id);
      }
    }

    CandidateRankerSegmentOrder& target_order =
        merge_order.segment_orders[segment_index];
    target_order.candidate_ids.clear();
    target_order.candidate_ids.reserve(snapshot_segment.candidates.size());
    size_t rankable_index = 0;
    for (const CandidateRankerCandidate& candidate :
         snapshot_segment.candidates) {
      if (candidate.is_protected) {
        target_order.candidate_ids.push_back(candidate.id);
      } else {
        target_order.candidate_ids.push_back(
            ranked_candidate_ids[rankable_index++]);
      }
    }
  }

  return merge_order;
}

absl::Status ApplyCandidateRankerResponse(
    const CandidateRankerRequest& request,
    const CandidateRankerResponse& response,
    const CandidateRankerToken& current_token, Segments* segments) {
  if (segments == nullptr) {
    return absl::InvalidArgumentError("segments must not be null");
  }
  const absl::Status token_status =
      ValidateTokens(request, response, current_token);
  if (!token_status.ok()) {
    return token_status;
  }

  const absl::Status snapshot_status = ValidateSnapshot(request, *segments);
  if (!snapshot_status.ok()) {
    return snapshot_status;
  }

  const absl::StatusOr<CandidateRankerMergeOrder> merge_order =
      BuildCandidateRankerMergeOrder(request, response, current_token);
  if (!merge_order.ok()) {
    return merge_order.status();
  }

  std::vector<SegmentMergePlan> plans;
  plans.reserve(request.segments.size());
  for (size_t segment_index = 0; segment_index < request.segments.size();
       ++segment_index) {
    const CandidateRankerSegment& snapshot_segment =
        request.segments[segment_index];
    Segment* segment = segments->mutable_conversion_segment(segment_index);
    SegmentMergePlan& plan = plans.emplace_back();
    plan.segment = segment;
    plan.target_order.reserve(snapshot_segment.candidates.size());
    for (CandidateRankerCandidateId candidate_id :
         merge_order->segment_orders[segment_index].candidate_ids) {
      size_t candidate_index = snapshot_segment.candidates.size();
      for (size_t i = 0; i < snapshot_segment.candidates.size(); ++i) {
        if (snapshot_segment.candidates[i].id == candidate_id) {
          candidate_index = i;
          break;
        }
      }
      plan.target_order.push_back(segment->mutable_candidate(candidate_index));
    }
  }

  for (const SegmentMergePlan& plan : plans) {
    for (size_t target_index = 0; target_index < plan.target_order.size();
         ++target_index) {
      if (plan.segment->mutable_candidate(target_index) ==
          plan.target_order[target_index]) {
        continue;
      }
      for (size_t current_index = target_index + 1;
           current_index < plan.target_order.size(); ++current_index) {
        if (plan.segment->mutable_candidate(current_index) ==
            plan.target_order[target_index]) {
          plan.segment->move_candidate(current_index, target_index);
          break;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace converter
}  // namespace mozc
