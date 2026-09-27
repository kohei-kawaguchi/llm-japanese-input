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

#include "engine/zenz_candidate_ranker.h"

#include <vector>

#include "converter/candidate_ranker.h"
#include "testing/gmock.h"
#include "testing/gunit.h"

namespace mozc {
namespace engine {
namespace {

using ::testing::ElementsAre;

converter::CandidateRankerCandidate Candidate(uint64_t id,
                                              absl::string_view value,
                                              bool is_protected = false) {
  return converter::CandidateRankerCandidate{
      .id = id, .value = std::string(value), .is_protected = is_protected};
}

converter::CandidateRankerRequest Request() {
  converter::CandidateRankerRequest request;
  request.preceding_text = "傘を持って";
  request.following_text = "から";
  request.reading = "あめがふる";
  request.segments.push_back(converter::CandidateRankerSegment{
      .id = 0,
      .key = "あめが",
      .candidates = {Candidate(0, "飴が"), Candidate(1, "アメガ", true),
                     Candidate(2, "雨が")},
  });
  request.segments.push_back(converter::CandidateRankerSegment{
      .id = 1,
      .key = "ふる",
      .candidates = {Candidate(3, "降る"), Candidate(4, "振る")},
  });
  return request;
}

TEST(ZenzCandidateRankerTest, PromptOrdersContextBeforeKatakanaInput) {
  EXPECT_EQ(BuildZenzPrompt(Request()),
            "\uEE02傘を持って\uEE07から\uEE00アメガフル\uEE01");
}

TEST(ZenzCandidateRankerTest, NextWindowScoresNextSegmentOrEnd) {
  const std::vector<ZenzScoredOutput> outputs =
      BuildZenzScoredOutputs(Request(), 5, ZenzRightWindow::kNext);
  ASSERT_EQ(outputs.size(), 4);
  EXPECT_EQ(outputs[0].candidate_id, 0);
  EXPECT_EQ(outputs[0].prefix, "");
  EXPECT_EQ(outputs[0].body, "飴が降る");
  EXPECT_FALSE(outputs[0].add_end);
  EXPECT_EQ(outputs[1].candidate_id, 2);
  EXPECT_EQ(outputs[3].prefix, "飴が");
  EXPECT_EQ(outputs[3].body, "振る");
  EXPECT_TRUE(outputs[3].add_end);
}

TEST(ZenzCandidateRankerTest, TopKLimitsUnprotectedCandidates) {
  const std::vector<ZenzScoredOutput> outputs =
      BuildZenzScoredOutputs(Request(), 1, ZenzRightWindow::kNone);
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].body, "飴が");
  EXPECT_EQ(outputs[1].body, "降る");
}

TEST(ZenzCandidateRankerTest, OrderAppliesPriorAndKeepsMozcOrderOnTies) {
  const converter::CandidateRankerSegment segment = Request().segments[0];
  EXPECT_THAT(OrderZenzSegment(segment, {-5.0, -1.0}, 0.0, 0.0).candidate_ids,
              ElementsAre(2, 0));
  EXPECT_THAT(OrderZenzSegment(segment, {-2.0, -1.0}, 0.0, 1.5).candidate_ids,
              ElementsAre(0, 2));
  EXPECT_THAT(OrderZenzSegment(segment, {0.0, 0.0}, 0.0, 0.0).candidate_ids,
              ElementsAre(0, 2));
}

TEST(ZenzCandidateRankerTest, CopyPenaltyAppliesOnlyWhenMozcTopIsNotACopy) {
  const converter::CandidateRankerSegment segment{
      .id = 0,
      .key = "あめが",
      .candidates = {Candidate(0, "雨が"), Candidate(1, "あめが"),
                     Candidate(2, "アメガ")},
  };
  EXPECT_THAT(
      OrderZenzSegment(segment, {-3.0, -1.0, -2.0}, 5.0, 0.0).candidate_ids,
      ElementsAre(0, 1, 2));
  const converter::CandidateRankerSegment copy_top{
      .id = 0,
      .key = "あめが",
      .candidates = {Candidate(0, "あめが"), Candidate(1, "雨が")},
  };
  EXPECT_THAT(OrderZenzSegment(copy_top, {-1.0, -3.0}, 100.0, 0.0).candidate_ids,
              ElementsAre(0, 1));
}

}  // namespace
}  // namespace engine
}  // namespace mozc
