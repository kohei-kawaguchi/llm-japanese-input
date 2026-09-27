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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/protobuf/message.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "testing/gunit.h"

namespace mozc::engine::evaluation {
namespace {

FrozenCandidateRankerCandidate* AddCandidate(
    FrozenCandidateRankerSegment* segment, uint64_t id, const std::string& key,
    const std::string& value, int32_t cost, uint32_t attributes,
    uint32_t consumed_key_size, bool is_protected) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key(key);
  candidate->set_value(value);
  candidate->set_cost(cost);
  candidate->set_attributes(attributes);
  candidate->set_consumed_key_size(consumed_key_size);
  candidate->set_is_protected(is_protected);
  return candidate;
}

FrozenCandidateRankerRequest MakeRequest(
    FrozenCandidateRankerRequest::Mode mode) {
  FrozenCandidateRankerRequest request;
  request.mutable_token()->set_session_generation(11);
  request.mutable_token()->set_state_revision(12);
  request.mutable_token()->set_request_sequence(13);
  request.set_mode(mode);
  request.set_preceding_text("左文脈");
  request.set_following_text("右文脈");
  request.set_reading("よみ");
  request.set_focused_segment_id(17);

  FrozenCandidateRankerSegment* first = request.add_segments();
  first->set_id(0);
  first->set_key("第一");
  AddCandidate(first, 0, "だいいち", "第壱", -101, 0x10, 4, true);
  AddCandidate(first, 1, "だいいち", "第一", 102, 0x20, 5, false);

  FrozenCandidateRankerSegment* second = request.add_segments();
  second->set_id(1);
  second->set_key("第二");
  AddCandidate(second, 2, "だいに", "第二", 103, 0x40, 6, false);
  return request;
}

void ExpectInvalid(const FrozenCandidateRankerRequest& request) {
  EXPECT_EQ(ValidateFrozenCandidateRankerRequestStructure(request).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ConvertFrozenCandidateRankerRequest(request).status().code(),
            absl::StatusCode::kInvalidArgument);
}

void AddUnknownField(protobuf::Message* message) {
  message->GetReflection()->MutableUnknownFields(message)->AddVarint(100, 1);
}

TEST(FrozenCandidateRankerUtilTest,
     ConvertsEveryRequestFieldAndPreservesOrderForEveryMode) {
  struct ModeCase {
    FrozenCandidateRankerRequest::Mode frozen;
    converter::CandidateRankerMode converted;
  };
  constexpr ModeCase kModes[] = {
      {FrozenCandidateRankerRequest::MODE_SUGGESTION,
       converter::CandidateRankerMode::kSuggestion},
      {FrozenCandidateRankerRequest::MODE_PREDICTION,
       converter::CandidateRankerMode::kPrediction},
      {FrozenCandidateRankerRequest::MODE_CONVERSION,
       converter::CandidateRankerMode::kConversion},
  };

  for (const ModeCase& mode : kModes) {
    const FrozenCandidateRankerRequest frozen = MakeRequest(mode.frozen);
    absl::StatusOr<converter::CandidateRankerRequest> converted =
        ConvertFrozenCandidateRankerRequest(frozen);
    ASSERT_TRUE(converted.ok()) << converted.status();
    EXPECT_EQ(converted->token.session_generation, 11);
    EXPECT_EQ(converted->token.state_revision, 12);
    EXPECT_EQ(converted->token.request_sequence, 13);
    EXPECT_EQ(converted->mode, mode.converted);
    EXPECT_EQ(converted->preceding_text, "左文脈");
    EXPECT_EQ(converted->following_text, "右文脈");
    EXPECT_EQ(converted->reading, "よみ");
    EXPECT_EQ(converted->focused_segment_id, 17);
    ASSERT_EQ(converted->segments.size(), 2);
    EXPECT_EQ(converted->segments[0].id, 0);
    EXPECT_EQ(converted->segments[0].key, "第一");
    EXPECT_EQ(converted->segments[1].id, 1);
    EXPECT_EQ(converted->segments[1].key, "第二");
    ASSERT_EQ(converted->segments[0].candidates.size(), 2);
    const converter::CandidateRankerCandidate& first =
        converted->segments[0].candidates[0];
    EXPECT_EQ(first.id, 0);
    EXPECT_EQ(first.key, "だいいち");
    EXPECT_EQ(first.value, "第壱");
    EXPECT_EQ(first.cost, -101);
    EXPECT_EQ(first.attributes, 0x10);
    EXPECT_EQ(first.consumed_key_size, 4);
    EXPECT_TRUE(first.is_protected);
    const converter::CandidateRankerCandidate& second =
        converted->segments[0].candidates[1];
    EXPECT_EQ(second.id, 1);
    EXPECT_EQ(second.key, "だいいち");
    EXPECT_EQ(second.value, "第一");
    EXPECT_EQ(second.cost, 102);
    EXPECT_EQ(second.attributes, 0x20);
    EXPECT_EQ(second.consumed_key_size, 5);
    EXPECT_FALSE(second.is_protected);
    ASSERT_EQ(converted->segments[1].candidates.size(), 1);
    const converter::CandidateRankerCandidate& third =
        converted->segments[1].candidates[0];
    EXPECT_EQ(third.id, 2);
    EXPECT_EQ(third.key, "だいに");
    EXPECT_EQ(third.value, "第二");
    EXPECT_EQ(third.cost, 103);
    EXPECT_EQ(third.attributes, 0x40);
    EXPECT_EQ(third.consumed_key_size, 6);
    EXPECT_FALSE(third.is_protected);
  }
}

TEST(FrozenCandidateRankerUtilTest, AllowsEmptyExternalContext) {
  FrozenCandidateRankerRequest request =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  request.set_preceding_text("");
  request.set_following_text("");
  ASSERT_TRUE(ValidateFrozenCandidateRankerRequestStructure(request).ok());
  absl::StatusOr<converter::CandidateRankerRequest> converted =
      ConvertFrozenCandidateRankerRequest(request);
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_EQ(converted->preceding_text, "");
  EXPECT_EQ(converted->following_text, "");
}

TEST(FrozenCandidateRankerUtilTest,
     RejectsIncompleteUnspecifiedAndUnknownModes) {
  FrozenCandidateRankerRequest incomplete =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  incomplete.clear_reading();
  ExpectInvalid(incomplete);

  const FrozenCandidateRankerRequest unspecified =
      MakeRequest(FrozenCandidateRankerRequest::MODE_UNSPECIFIED);
  ExpectInvalid(unspecified);

  FrozenCandidateRankerRequest unknown_mode =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  unknown_mode.clear_mode();
  std::string unknown_mode_wire;
  ASSERT_TRUE(unknown_mode.SerializePartialToString(&unknown_mode_wire));
  unknown_mode_wire.append("\x10\x63", 2);
  FrozenCandidateRankerRequest parsed_unknown_mode;
  ASSERT_TRUE(parsed_unknown_mode.ParsePartialFromString(unknown_mode_wire));
  ExpectInvalid(parsed_unknown_mode);
}

TEST(FrozenCandidateRankerUtilTest, RejectsRecursiveUnknownFields) {
  FrozenCandidateRankerRequest request =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  AddUnknownField(&request);
  ExpectInvalid(request);

  request = MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  AddUnknownField(request.mutable_token());
  ExpectInvalid(request);

  request = MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  AddUnknownField(request.mutable_segments(0));
  ExpectInvalid(request);

  request = MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  AddUnknownField(request.mutable_segments(0)->mutable_candidates(0));
  ExpectInvalid(request);
}

TEST(FrozenCandidateRankerUtilTest,
     RejectsMissingCollectionsAndNonSequentialIds) {
  FrozenCandidateRankerRequest no_segments =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  no_segments.clear_segments();
  ExpectInvalid(no_segments);

  FrozenCandidateRankerRequest no_candidates =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  no_candidates.mutable_segments(0)->clear_candidates();
  ExpectInvalid(no_candidates);

  FrozenCandidateRankerRequest segment_ids =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  segment_ids.mutable_segments(1)->set_id(0);
  ExpectInvalid(segment_ids);

  FrozenCandidateRankerRequest candidate_ids =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  candidate_ids.mutable_segments(1)->mutable_candidates(0)->set_id(1);
  ExpectInvalid(candidate_ids);
}

TEST(FrozenCandidateRankerUtilTest, RejectsMalformedUtf8InEveryStringLayer) {
  const std::string invalid_utf8(1, static_cast<char>(0xff));

  FrozenCandidateRankerRequest preceding =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  preceding.set_preceding_text(invalid_utf8);
  ExpectInvalid(preceding);

  FrozenCandidateRankerRequest following =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  following.set_following_text(invalid_utf8);
  ExpectInvalid(following);

  FrozenCandidateRankerRequest reading =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  reading.set_reading(invalid_utf8);
  ExpectInvalid(reading);

  FrozenCandidateRankerRequest segment =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  segment.mutable_segments(0)->set_key(invalid_utf8);
  ExpectInvalid(segment);

  FrozenCandidateRankerRequest candidate_key =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  candidate_key.mutable_segments(0)->mutable_candidates(0)->set_key(
      invalid_utf8);
  ExpectInvalid(candidate_key);

  FrozenCandidateRankerRequest candidate_value =
      MakeRequest(FrozenCandidateRankerRequest::MODE_CONVERSION);
  candidate_value.mutable_segments(0)->mutable_candidates(0)->set_value(
      invalid_utf8);
  ExpectInvalid(candidate_value);
}

}  // namespace
}  // namespace mozc::engine::evaluation
