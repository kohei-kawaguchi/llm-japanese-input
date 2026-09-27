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

#include "converter/candidate_limits.h"

#include <string>

#include "converter/segments.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"
#include "testing/gunit.h"

namespace mozc {
namespace converter {
namespace {

ConversionRequest MakeSuggestionRequest(bool defer_candidate_limits,
                                        int suggestions_size = 3,
                                        int candidates_size_limit = 3) {
  commands::Request request;
  request.set_candidates_size_limit(candidates_size_limit);
  config::Config config;
  config.set_suggestions_size(suggestions_size);
  ConversionOptions options;
  options.request_type = ConversionRequest::SUGGESTION;
  options.defer_candidate_limits = defer_candidate_limits;
  return ConversionRequestBuilder()
      .SetRequest(request)
      .SetConfig(config)
      .SetOptions(options)
      .Build();
}

Segment* MakeSegment(Segments* segments) {
  Segment* segment = segments->add_segment();
  for (const char* value : {"A", "B", "C", "D", "E"}) {
    segment->add_candidate()->value = value;
  }
  return segment;
}

TEST(CandidateLimitsTest, DefaultPhasesPreservePreSuppressionSuggestionLimit) {
  const ConversionRequest request = MakeSuggestionRequest(false);
  Segments segments;
  Segment* segment = MakeSegment(&segments);

  ApplySuggestionCandidateLimit(request, &segments);
  ASSERT_EQ(segment->candidates_size(), 3);
  segment->erase_candidate(1);
  ApplyRequestCandidateLimit(request, &segments);

  ASSERT_EQ(segment->candidates_size(), 2);
  EXPECT_EQ(segment->candidate(0).value, "A");
  EXPECT_EQ(segment->candidate(1).value, "C");
}

TEST(CandidateLimitsTest, FinalizerAppliesBothLimitsAfterSuppression) {
  {
    const ConversionRequest request = MakeSuggestionRequest(true, 3, 5);
    Segments segments;
    Segment* segment = MakeSegment(&segments);

    segment->erase_candidate(1);
    ApplyCandidateLimits(request, &segments);

    ASSERT_EQ(segment->candidates_size(), 3);
    EXPECT_EQ(segment->candidate(0).value, "A");
    EXPECT_EQ(segment->candidate(1).value, "C");
    EXPECT_EQ(segment->candidate(2).value, "D");
  }

  {
    const ConversionRequest request = MakeSuggestionRequest(true, 5, 2);
    Segments segments;
    Segment* segment = MakeSegment(&segments);

    segment->erase_candidate(1);
    ApplyCandidateLimits(request, &segments);

    ASSERT_EQ(segment->candidates_size(), 2);
    EXPECT_EQ(segment->candidate(0).value, "A");
    EXPECT_EQ(segment->candidate(1).value, "C");
  }
}

}  // namespace
}  // namespace converter
}  // namespace mozc
