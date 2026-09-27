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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "converter/attribute.h"
#include "converter/candidate.h"
#include "converter/segments.h"
#include "testing/gunit.h"

namespace mozc {
namespace converter {
namespace {

Candidate* AddCandidate(
    Segment* segment, std::string value, int32_t cost,
    uint32_t attributes = Attribute::DEFAULT_ATTRIBUTE,
    Candidate::Command command = Candidate::DEFAULT_COMMAND) {
  Candidate* candidate = segment->add_candidate();
  candidate->key = std::string(segment->key());
  candidate->value = std::move(value);
  candidate->content_key = candidate->key;
  candidate->content_value = candidate->value;
  candidate->description = candidate->value;
  candidate->cost = cost;
  candidate->wcost = cost + 1;
  candidate->structure_cost = cost + 2;
  candidate->cost_before_rescoring = cost + 3;
  candidate->attributes = attributes;
  candidate->command = command;
  return candidate;
}

TEST(CandidateRankerTest, BuildsImmutableModelRequest) {
  Segments segments;
  Segment* history = segments.add_segment();
  history->set_segment_type(Segment::HISTORY);
  history->set_key("きのう");
  AddCandidate(history, "昨日", 1);

  Segment* first = segments.add_segment();
  first->set_key("しれ");
  Candidate* partial =
      AddCandidate(first, "知れ", 10, Attribute::PARTIALLY_KEY_CONSUMED);
  partial->consumed_key_size = 2;
  AddCandidate(first, "しれ", 20, Attribute::NO_MODIFICATION);

  const CandidateRankerToken token{
      .session_generation = 4,
      .state_revision = 5,
      .request_sequence = 6,
  };
  const CandidateRankerRequest request =
      BuildCandidateRankerRequest(token, CandidateRankerMode::kSuggestion,
                                  "しれません", "昨日は", "。", 0, segments);

  EXPECT_EQ(request.token, token);
  EXPECT_EQ(request.mode, CandidateRankerMode::kSuggestion);
  EXPECT_EQ(request.preceding_text, "昨日は");
  EXPECT_EQ(request.following_text, "。");
  EXPECT_EQ(request.reading, "しれません");
  EXPECT_EQ(request.focused_segment_id, 0);
  ASSERT_EQ(request.segments.size(), 1);
  EXPECT_EQ(request.segments[0].id, 0);
  EXPECT_EQ(request.segments[0].key, "しれ");
  ASSERT_EQ(request.segments[0].candidates.size(), 2);
  EXPECT_EQ(request.segments[0].candidates[0].id, 0);
  EXPECT_EQ(request.segments[0].candidates[0].value, "知れ");
  EXPECT_EQ(request.segments[0].candidates[0].cost, 10);
  EXPECT_EQ(request.segments[0].candidates[0].consumed_key_size, 2);
  EXPECT_FALSE(request.segments[0].candidates[0].is_protected);
  EXPECT_EQ(request.segments[0].candidates[1].id, 1);
  EXPECT_TRUE(request.segments[0].candidates[1].is_protected);
}

TEST(CandidateRankerTest,
     BuildsCompleteMergeOrderWithProtectedSlotsAndStableOmissions) {
  Segments segments;
  Segment* first_segment = segments.add_segment();
  first_segment->set_key("だいいち");
  AddCandidate(first_segment, "第一", 10);
  AddCandidate(first_segment, "第壱", 20, Attribute::NO_MODIFICATION);
  AddCandidate(first_segment, "題一", 30);
  AddCandidate(first_segment, "台一", 40,
               Attribute::DISABLE_RESCORING);
  AddCandidate(first_segment, "大一", 50);
  Segment* second_segment = segments.add_segment();
  second_segment->set_key("だいに");
  AddCandidate(second_segment, "第二", 60);
  AddCandidate(second_segment, "第弐", 70);

  const CandidateRankerToken token{
      .session_generation = 40,
      .state_revision = 41,
      .request_sequence = 42,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kConversion, "だいいちだいに", "", "", 0,
      segments);
  const CandidateRankerResponse response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {4, 0}}},
  };

  const absl::StatusOr<CandidateRankerMergeOrder> order =
      BuildCandidateRankerMergeOrder(request, response, token);
  ASSERT_TRUE(order.ok()) << order.status();
  ASSERT_EQ(order->segment_orders.size(), 2);
  EXPECT_EQ(order->segment_orders[0].segment_id, 0);
  EXPECT_EQ(order->segment_orders[0].candidate_ids,
            (std::vector<CandidateRankerCandidateId>{4, 1, 0, 3, 2}));
  EXPECT_EQ(order->segment_orders[1].segment_id, 1);
  EXPECT_EQ(order->segment_orders[1].candidate_ids,
            (std::vector<CandidateRankerCandidateId>{5, 6}));
}

TEST(CandidateRankerTest, BuildsMergeOrderForZeroAndOneRankableCandidate) {
  Segments segments;
  Segment* no_rankable = segments.add_segment();
  no_rankable->set_key("こてい");
  AddCandidate(no_rankable, "固定", 10, Attribute::NO_MODIFICATION);
  AddCandidate(no_rankable, "コテイ", 20,
               Attribute::DISABLE_RESCORING);
  Segment* one_rankable = segments.add_segment();
  one_rankable->set_key("ひとつ");
  AddCandidate(one_rankable, "一つ", 30, Attribute::COMMAND_CANDIDATE);
  AddCandidate(one_rankable, "一", 40);
  AddCandidate(one_rankable, "ひとつ", 50, Attribute::NO_MODIFICATION);

  const CandidateRankerToken token{
      .session_generation = 43,
      .state_revision = 44,
      .request_sequence = 45,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kConversion, "こていひとつ", "", "", 0,
      segments);
  const CandidateRankerResponse response{
      .token = token,
      .segment_orders = {
          {.segment_id = 0, .candidate_ids = {}},
          {.segment_id = 1, .candidate_ids = {3}},
      },
  };

  const absl::StatusOr<CandidateRankerMergeOrder> order =
      BuildCandidateRankerMergeOrder(request, response, token);
  ASSERT_TRUE(order.ok()) << order.status();
  EXPECT_EQ(order->segment_orders[0].candidate_ids,
            (std::vector<CandidateRankerCandidateId>{0, 1}));
  EXPECT_EQ(order->segment_orders[1].candidate_ids,
            (std::vector<CandidateRankerCandidateId>{2, 3, 4}));
}

TEST(CandidateRankerTest, PureMergeOrderRejectsMalformedReferencesAndTokens) {
  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("けんしょう");
  AddCandidate(segment, "検証", 10);
  AddCandidate(segment, "懸賞", 20, Attribute::NO_MODIFICATION);

  const CandidateRankerToken token{
      .session_generation = 46,
      .state_revision = 47,
      .request_sequence = 48,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kConversion, "けんしょう", "", "", 0,
      segments);

  CandidateRankerToken stale_token = token;
  ++stale_token.state_revision;
  const CandidateRankerResponse valid_response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {0}}},
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, valid_response,
                                           stale_token)
                .status()
                .code(),
            absl::StatusCode::kAborted);

  CandidateRankerToken wrong_response_token = token;
  ++wrong_response_token.request_sequence;
  const CandidateRankerResponse mismatched_token{
      .token = wrong_response_token,
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, mismatched_token, token)
                .status()
                .code(),
            absl::StatusCode::kAborted);

  const CandidateRankerResponse duplicate_segment{
      .token = token,
      .segment_orders = {
          {.segment_id = 0, .candidate_ids = {}},
          {.segment_id = 0, .candidate_ids = {}},
      },
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, duplicate_segment, token)
                .status()
                .code(),
            absl::StatusCode::kAlreadyExists);

  const CandidateRankerResponse unknown_segment{
      .token = token,
      .segment_orders = {{.segment_id = 99, .candidate_ids = {}}},
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, unknown_segment, token)
                .status()
                .code(),
            absl::StatusCode::kNotFound);

  const CandidateRankerResponse unknown_candidate{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {99}}},
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, unknown_candidate, token)
                .status()
                .code(),
            absl::StatusCode::kNotFound);

  const CandidateRankerResponse duplicate_candidate{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {0, 0}}},
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, duplicate_candidate, token)
                .status()
                .code(),
            absl::StatusCode::kAlreadyExists);

  const CandidateRankerResponse protected_candidate{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {1}}},
  };
  EXPECT_EQ(BuildCandidateRankerMergeOrder(request, protected_candidate, token)
                .status()
                .code(),
            absl::StatusCode::kPermissionDenied);
}

TEST(CandidateRankerTest, PureMergeOrderMatchesAppliedPointerOrder) {
  Segments segments;
  Segment* first_segment = segments.add_segment();
  first_segment->set_key("ひと");
  Candidate* candidate_zero = AddCandidate(first_segment, "人", 10);
  Candidate* candidate_one =
      AddCandidate(first_segment, "一", 20, Attribute::NO_MODIFICATION);
  Candidate* candidate_two = AddCandidate(first_segment, "日と", 30);
  Segment* second_segment = segments.add_segment();
  second_segment->set_key("つぎ");
  Candidate* candidate_three = AddCandidate(second_segment, "次", 40);
  Candidate* candidate_four = AddCandidate(second_segment, "継ぎ", 50);
  const std::vector<Candidate*> candidates_by_id = {
      candidate_zero, candidate_one, candidate_two, candidate_three,
      candidate_four,
  };

  const CandidateRankerToken token{
      .session_generation = 49,
      .state_revision = 50,
      .request_sequence = 51,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kConversion, "ひとつぎ", "", "", 0,
      segments);
  const CandidateRankerResponse response{
      .token = token,
      .segment_orders = {
          {.segment_id = 1, .candidate_ids = {4}},
          {.segment_id = 0, .candidate_ids = {2}},
      },
  };
  const absl::StatusOr<CandidateRankerMergeOrder> order =
      BuildCandidateRankerMergeOrder(request, response, token);
  ASSERT_TRUE(order.ok()) << order.status();

  const absl::Status status =
      ApplyCandidateRankerResponse(request, response, token, &segments);
  ASSERT_TRUE(status.ok()) << status;
  for (size_t segment_index = 0; segment_index < order->segment_orders.size();
       ++segment_index) {
    Segment* applied_segment =
        segments.mutable_conversion_segment(segment_index);
    const std::vector<CandidateRankerCandidateId>& candidate_ids =
        order->segment_orders[segment_index].candidate_ids;
    ASSERT_EQ(applied_segment->candidates_size(), candidate_ids.size());
    for (size_t candidate_index = 0; candidate_index < candidate_ids.size();
         ++candidate_index) {
      EXPECT_EQ(applied_segment->mutable_candidate(candidate_index),
                candidates_by_id[candidate_ids[candidate_index]]);
    }
  }
}

TEST(CandidateRankerTest, ReranksPointersAndPreservesMetadata) {
  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("こうしょう");
  Candidate* first = AddCandidate(segment, "交渉", 10);
  Candidate* second = AddCandidate(segment, "高尚", 20);
  Candidate* third = AddCandidate(segment, "校章", 30);
  Candidate* fourth = AddCandidate(segment, "公称", 40);
  const std::array<int32_t, 4> original_costs = {
      first->cost_before_rescoring,
      second->cost_before_rescoring,
      third->cost_before_rescoring,
      fourth->cost_before_rescoring,
  };

  const CandidateRankerToken token{
      .session_generation = 1,
      .state_revision = 2,
      .request_sequence = 3,
  };
  const CandidateRankerRequest request =
      BuildCandidateRankerRequest(token, CandidateRankerMode::kConversion,
                                  "こうしょう", "", "", 0, segments);
  const CandidateRankerResponse response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {2, 0}}},
  };

  const absl::Status status =
      ApplyCandidateRankerResponse(request, response, token, &segments);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(segment->mutable_candidate(0), third);
  EXPECT_EQ(segment->mutable_candidate(1), first);
  EXPECT_EQ(segment->mutable_candidate(2), second);
  EXPECT_EQ(segment->mutable_candidate(3), fourth);
  EXPECT_EQ(third->cost_before_rescoring, original_costs[2]);
  EXPECT_EQ(first->cost_before_rescoring, original_costs[0]);
  EXPECT_EQ(second->cost_before_rescoring, original_costs[1]);
  EXPECT_EQ(fourth->cost_before_rescoring, original_costs[3]);
}

TEST(CandidateRankerTest, PinsProtectedCandidatesAndLeavesMetaAndHistory) {
  Segments segments;
  Segment* history = segments.add_segment();
  history->set_segment_type(Segment::HISTORY);
  history->set_key("まえ");
  Candidate* history_candidate = AddCandidate(history, "前", 1);

  Segment* segment = segments.add_segment();
  segment->set_key("こうほ");
  Candidate* first = AddCandidate(segment, "候補一", 10);
  Candidate* no_modification =
      AddCandidate(segment, "候補二", 20, Attribute::NO_MODIFICATION);
  Candidate* third = AddCandidate(segment, "候補三", 30);
  Candidate* no_rescoring =
      AddCandidate(segment, "候補四", 40, Attribute::DISABLE_RESCORING);
  Candidate* command_attribute =
      AddCandidate(segment, "候補五", 50, Attribute::COMMAND_CANDIDATE);
  Candidate* command =
      AddCandidate(segment, "候補六", 60, Attribute::DEFAULT_ATTRIBUTE,
                   Candidate::ENABLE_INCOGNITO_MODE);
  Candidate* meta = segment->add_meta_candidate();
  meta->key = "こうほ";
  meta->value = "コウホ";

  const CandidateRankerToken token{
      .session_generation = 10,
      .state_revision = 11,
      .request_sequence = 12,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kPrediction, "こうほ", "", "", 0, segments);
  EXPECT_TRUE(request.segments[0].candidates[4].is_protected);
  const CandidateRankerResponse protected_response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {4}}},
  };
  const absl::Status protected_status = ApplyCandidateRankerResponse(
      request, protected_response, token, &segments);
  EXPECT_EQ(protected_status.code(), absl::StatusCode::kPermissionDenied);
  EXPECT_EQ(segment->mutable_candidate(0), first);
  EXPECT_EQ(segment->mutable_candidate(1), no_modification);
  EXPECT_EQ(segment->mutable_candidate(2), third);

  const CandidateRankerResponse valid_response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {2, 0}}},
  };
  const absl::Status valid_status =
      ApplyCandidateRankerResponse(request, valid_response, token, &segments);
  ASSERT_TRUE(valid_status.ok()) << valid_status;
  EXPECT_EQ(segment->mutable_candidate(0), third);
  EXPECT_EQ(segment->mutable_candidate(1), no_modification);
  EXPECT_EQ(segment->mutable_candidate(2), first);
  EXPECT_EQ(segment->mutable_candidate(3), no_rescoring);
  EXPECT_EQ(segment->mutable_candidate(4), command_attribute);
  EXPECT_EQ(segment->mutable_candidate(5), command);
  EXPECT_EQ(&segment->meta_candidate(0), meta);
  EXPECT_EQ(history->mutable_candidate(0), history_candidate);
}

TEST(CandidateRankerTest, RejectsMalformedResponseBeforeAnyMutation) {
  Segments segments;
  Segment* first_segment = segments.add_segment();
  first_segment->set_key("さいしょ");
  Candidate* first_zero = AddCandidate(first_segment, "最初", 10);
  Candidate* first_one = AddCandidate(first_segment, "採書", 20);
  Segment* second_segment = segments.add_segment();
  second_segment->set_key("つぎ");
  Candidate* second_zero = AddCandidate(second_segment, "次", 30);
  Candidate* second_one = AddCandidate(second_segment, "継ぎ", 40);

  const CandidateRankerToken token{
      .session_generation = 20,
      .state_revision = 21,
      .request_sequence = 22,
  };
  const CandidateRankerRequest request =
      BuildCandidateRankerRequest(token, CandidateRankerMode::kConversion,
                                  "さいしょつぎ", "", "", 0, segments);

  const CandidateRankerResponse invalid_later_segment{
      .token = token,
      .segment_orders =
          {
              {.segment_id = 0, .candidate_ids = {1, 0}},
              {.segment_id = 1, .candidate_ids = {999}},
          },
  };
  const absl::Status invalid_status = ApplyCandidateRankerResponse(
      request, invalid_later_segment, token, &segments);
  EXPECT_EQ(invalid_status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(first_segment->mutable_candidate(0), first_zero);
  EXPECT_EQ(first_segment->mutable_candidate(1), first_one);
  EXPECT_EQ(second_segment->mutable_candidate(0), second_zero);
  EXPECT_EQ(second_segment->mutable_candidate(1), second_one);

  CandidateRankerToken wrong_response_token = token;
  ++wrong_response_token.request_sequence;
  const CandidateRankerResponse mismatched_token{
      .token = wrong_response_token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {1, 0}}},
  };
  const absl::Status mismatched_token_status =
      ApplyCandidateRankerResponse(request, mismatched_token, token, &segments);
  EXPECT_EQ(mismatched_token_status.code(), absl::StatusCode::kAborted);

  const CandidateRankerResponse duplicate_segment{
      .token = token,
      .segment_orders =
          {
              {.segment_id = 0, .candidate_ids = {1, 0}},
              {.segment_id = 0, .candidate_ids = {}},
          },
  };
  const absl::Status duplicate_segment_status = ApplyCandidateRankerResponse(
      request, duplicate_segment, token, &segments);
  EXPECT_EQ(duplicate_segment_status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(first_segment->mutable_candidate(0), first_zero);
  EXPECT_EQ(first_segment->mutable_candidate(1), first_one);

  const CandidateRankerResponse unknown_segment{
      .token = token,
      .segment_orders = {{.segment_id = 999, .candidate_ids = {}}},
  };
  const absl::Status unknown_segment_status =
      ApplyCandidateRankerResponse(request, unknown_segment, token, &segments);
  EXPECT_EQ(unknown_segment_status.code(), absl::StatusCode::kNotFound);

  const CandidateRankerResponse cross_segment_candidate{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {2}}},
  };
  const absl::Status cross_segment_status = ApplyCandidateRankerResponse(
      request, cross_segment_candidate, token, &segments);
  EXPECT_EQ(cross_segment_status.code(), absl::StatusCode::kNotFound);

  const CandidateRankerResponse duplicate_candidate{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {0, 0}}},
  };
  const absl::Status duplicate_status = ApplyCandidateRankerResponse(
      request, duplicate_candidate, token, &segments);
  EXPECT_EQ(duplicate_status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(first_segment->mutable_candidate(0), first_zero);
  EXPECT_EQ(first_segment->mutable_candidate(1), first_one);

  CandidateRankerToken newer_token = token;
  ++newer_token.state_revision;
  const CandidateRankerResponse valid_response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {1, 0}}},
  };
  const absl::Status stale_status = ApplyCandidateRankerResponse(
      request, valid_response, newer_token, &segments);
  EXPECT_EQ(stale_status.code(), absl::StatusCode::kAborted);
  EXPECT_EQ(first_segment->mutable_candidate(0), first_zero);
  EXPECT_EQ(first_segment->mutable_candidate(1), first_one);
}

TEST(CandidateRankerTest, RejectsChangedSnapshot) {
  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("へんこう");
  Candidate* first = AddCandidate(segment, "変更", 10);
  Candidate* second =
      AddCandidate(segment, "偏光", 20, Attribute::PARTIALLY_KEY_CONSUMED);
  second->consumed_key_size = 3;
  const CandidateRankerToken token{
      .session_generation = 30,
      .state_revision = 31,
      .request_sequence = 32,
  };
  const CandidateRankerRequest request = BuildCandidateRankerRequest(
      token, CandidateRankerMode::kConversion, "へんこう", "", "", 0, segments);
  second->consumed_key_size = 4;
  const CandidateRankerResponse response{
      .token = token,
      .segment_orders = {{.segment_id = 0, .candidate_ids = {1, 0}}},
  };

  const absl::Status status =
      ApplyCandidateRankerResponse(request, response, token, &segments);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(segment->mutable_candidate(0), first);
  EXPECT_EQ(segment->mutable_candidate(1), second);
}

}  // namespace
}  // namespace converter
}  // namespace mozc
