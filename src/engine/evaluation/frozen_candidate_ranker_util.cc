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

#include "engine/evaluation/frozen_candidate_ranker_util.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/strings/unicode.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"

namespace mozc::engine::evaluation {
namespace {

absl::StatusOr<converter::CandidateRankerMode> ConvertFrozenMode(
    FrozenCandidateRankerRequest::Mode mode) {
  switch (mode) {
    case FrozenCandidateRankerRequest::MODE_SUGGESTION:
      return converter::CandidateRankerMode::kSuggestion;
    case FrozenCandidateRankerRequest::MODE_PREDICTION:
      return converter::CandidateRankerMode::kPrediction;
    case FrozenCandidateRankerRequest::MODE_CONVERSION:
      return converter::CandidateRankerMode::kConversion;
    case FrozenCandidateRankerRequest::MODE_UNSPECIFIED:
      return absl::InvalidArgumentError(
          "frozen candidate-ranker request mode is unspecified");
  }
  return absl::InvalidArgumentError(
      "frozen candidate-ranker request mode is invalid");
}

}  // namespace

absl::Status ValidateFrozenCandidateRankerRequestStructure(
    const FrozenCandidateRankerRequest& request) {
  if (!request.IsInitialized() || HasUnknownFieldsRecursively(request)) {
    return absl::InvalidArgumentError(
        "frozen candidate-ranker request is incomplete or has unknown fields");
  }
  if (!ConvertFrozenMode(request.mode()).ok()) {
    return absl::InvalidArgumentError(
        "frozen candidate-ranker request mode is invalid");
  }
  if (!strings::IsValidUtf8(request.preceding_text()) ||
      !strings::IsValidUtf8(request.following_text()) ||
      !strings::IsValidUtf8(request.reading()) || request.segments().empty()) {
    return absl::InvalidArgumentError(
        "frozen candidate-ranker request text or segments are invalid");
  }

  uint64_t expected_candidate_id = 0;
  for (int segment_index = 0; segment_index < request.segments_size();
       ++segment_index) {
    const FrozenCandidateRankerSegment& segment =
        request.segments(segment_index);
    if (segment.id() != static_cast<uint64_t>(segment_index) ||
        !strings::IsValidUtf8(segment.key()) || segment.candidates().empty()) {
      return absl::InvalidArgumentError(
          "frozen candidate-ranker segment is invalid");
    }
    for (const FrozenCandidateRankerCandidate& candidate :
         segment.candidates()) {
      if (candidate.id() != expected_candidate_id++ ||
          !strings::IsValidUtf8(candidate.key()) ||
          !strings::IsValidUtf8(candidate.value())) {
        return absl::InvalidArgumentError(
            "frozen candidate-ranker candidate is invalid or unordered");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<converter::CandidateRankerRequest>
ConvertFrozenCandidateRankerRequest(
    const FrozenCandidateRankerRequest& request) {
  absl::Status status =
      ValidateFrozenCandidateRankerRequestStructure(request);
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<converter::CandidateRankerMode> mode =
      ConvertFrozenMode(request.mode());
  if (!mode.ok()) {
    return mode.status();
  }

  converter::CandidateRankerRequest converted;
  converted.token = converter::CandidateRankerToken{
      .session_generation = request.token().session_generation(),
      .state_revision = request.token().state_revision(),
      .request_sequence = request.token().request_sequence(),
  };
  converted.mode = *mode;
  converted.preceding_text = request.preceding_text();
  converted.following_text = request.following_text();
  converted.reading = request.reading();
  converted.focused_segment_id = request.focused_segment_id();
  converted.segments.reserve(request.segments_size());
  for (const FrozenCandidateRankerSegment& frozen_segment :
       request.segments()) {
    converter::CandidateRankerSegment segment{
        .id = frozen_segment.id(),
        .key = frozen_segment.key(),
    };
    segment.candidates.reserve(frozen_segment.candidates_size());
    for (const FrozenCandidateRankerCandidate& frozen_candidate :
         frozen_segment.candidates()) {
      segment.candidates.push_back(converter::CandidateRankerCandidate{
          .id = frozen_candidate.id(),
          .key = frozen_candidate.key(),
          .value = frozen_candidate.value(),
          .cost = frozen_candidate.cost(),
          .attributes = frozen_candidate.attributes(),
          .consumed_key_size = frozen_candidate.consumed_key_size(),
          .is_protected = frozen_candidate.is_protected(),
      });
    }
    converted.segments.push_back(std::move(segment));
  }
  return converted;
}

}  // namespace mozc::engine::evaluation
