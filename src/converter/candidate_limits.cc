// Copyright 2010-2021, Google Inc.
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
//     * Neither the name of Google Inc. nor the names of its
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

#include "converter/candidate_limits.h"

#include <algorithm>
#include <cstddef>

#include "converter/segments.h"
#include "protocol/commands.pb.h"
#include "request/conversion_request.h"

namespace mozc {
namespace converter {

void ApplySuggestionCandidateLimit(const ConversionRequest& request,
                                   Segments* segments) {
  if (request.request_type() != ConversionRequest::SUGGESTION ||
      segments->conversion_segments_size() != 1 ||
      request.request().mixed_conversion()) {
    return;
  }

  const size_t max_suggestions = request.config().suggestions_size();
  Segment* segment = segments->mutable_conversion_segment(0);
  const size_t candidate_size = segment->candidates_size();
  if (candidate_size > max_suggestions) {
    segment->erase_candidates(max_suggestions,
                              candidate_size - max_suggestions);
  }
}

void ApplyRequestCandidateLimit(const ConversionRequest& request,
                                Segments* segments) {
  const commands::Request& request_proto = request.request();
  if (!request_proto.has_candidates_size_limit()) {
    return;
  }

  const int limit = request_proto.candidates_size_limit();
  for (Segment& segment : segments->conversion_segments()) {
    const int candidates_size = segment.candidates_size();
    const int candidates_limit =
        std::max<int>(1, limit - segment.meta_candidates_size());
    if (candidates_size < candidates_limit) {
      continue;
    }
    segment.erase_candidates(candidates_limit,
                             candidates_size - candidates_limit);
  }
}

void ApplyCandidateLimits(const ConversionRequest& request,
                          Segments* segments) {
  ApplySuggestionCandidateLimit(request, segments);
  ApplyRequestCandidateLimit(request, segments);
}

}  // namespace converter
}  // namespace mozc
