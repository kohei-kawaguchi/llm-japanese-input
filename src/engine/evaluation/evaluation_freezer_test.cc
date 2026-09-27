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
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

#include "engine/evaluation/evaluation_freezer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "base/clock.h"
#include "base/clock_mock.h"
#include "converter/candidate.h"
#include "converter/candidate_ranker.h"
#include "converter/converter_mock.h"
#include "converter/segments.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"
#include "testing/gmock.h"
#include "testing/gunit.h"

namespace mozc::engine::evaluation {
namespace {

using ::testing::_;
using ::testing::Return;

constexpr char kEvaluationClockUtcRfc3339[] = "2000-01-01T00:00:00Z";

void AddCandidate(Segment* segment, std::string key, std::string value,
                  int32_t cost, uint32_t attributes, uint32_t consumed_key_size,
                  converter::Candidate::Command command =
                      converter::Candidate::DEFAULT_COMMAND) {
  converter::Candidate* candidate = segment->push_back_candidate();
  candidate->key = std::move(key);
  candidate->value = std::move(value);
  candidate->cost = cost;
  candidate->attributes = attributes;
  candidate->consumed_key_size = consumed_key_size;
  candidate->command = command;
}

TEST(EvaluationFreezerTest, CopiesCompleteRankerRequestInExistingOrder) {
  converter::CandidateRankerRequest source;
  source.token = {
      .session_generation = 11,
      .state_revision = 22,
      .request_sequence = 33,
  };
  source.mode = converter::CandidateRankerMode::kPrediction;
  source.preceding_text = "preceding";
  source.following_text = "following";
  source.reading = "reading";
  source.focused_segment_id = 7;
  source.segments = {
      {
          .id = 9,
          .key = "first-key",
          .candidates =
              {
                  {
                      .id = 41,
                      .key = "first-candidate-key",
                      .value = "first-value",
                      .cost = -101,
                      .attributes = 0x10,
                      .consumed_key_size = 3,
                      .is_protected = true,
                  },
                  {
                      .id = 42,
                      .key = "second-candidate-key",
                      .value = "second-value",
                      .cost = 202,
                      .attributes = 0x20,
                      .consumed_key_size = 6,
                      .is_protected = false,
                  },
              },
      },
      {
          .id = 10,
          .key = "second-key",
          .candidates =
              {
                  {
                      .id = 43,
                      .key = "third-candidate-key",
                      .value = "third-value",
                      .cost = 303,
                      .attributes = 0x40,
                      .consumed_key_size = 9,
                      .is_protected = false,
                  },
              },
      },
  };

  absl::StatusOr<FrozenCandidateRankerRequest> frozen =
      FreezeCandidateRankerRequest(source);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_EQ(frozen->token().session_generation(), 11);
  EXPECT_EQ(frozen->token().state_revision(), 22);
  EXPECT_EQ(frozen->token().request_sequence(), 33);
  EXPECT_EQ(frozen->mode(), FrozenCandidateRankerRequest::MODE_PREDICTION);
  EXPECT_EQ(frozen->preceding_text(), "preceding");
  EXPECT_EQ(frozen->following_text(), "following");
  EXPECT_EQ(frozen->reading(), "reading");
  EXPECT_EQ(frozen->focused_segment_id(), 7);
  ASSERT_EQ(frozen->segments_size(), 2);
  EXPECT_EQ(frozen->segments(0).id(), 9);
  EXPECT_EQ(frozen->segments(1).id(), 10);
  ASSERT_EQ(frozen->segments(0).candidates_size(), 2);
  ASSERT_EQ(frozen->segments(1).candidates_size(), 1);
  const FrozenCandidateRankerCandidate& first =
      frozen->segments(0).candidates(0);
  EXPECT_EQ(first.id(), 41);
  EXPECT_EQ(first.key(), "first-candidate-key");
  EXPECT_EQ(first.value(), "first-value");
  EXPECT_EQ(first.cost(), -101);
  EXPECT_EQ(first.attributes(), 0x10);
  EXPECT_EQ(first.consumed_key_size(), 3);
  EXPECT_TRUE(first.is_protected());
  EXPECT_EQ(frozen->segments(0).candidates(1).id(), 42);
  EXPECT_EQ(frozen->segments(1).candidates(0).id(), 43);
}

TEST(EvaluationFreezerTest, RejectsUnknownRankerMode) {
  converter::CandidateRankerRequest source;
  source.mode = static_cast<converter::CandidateRankerMode>(255);
  EXPECT_EQ(FreezeCandidateRankerRequest(source).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(EvaluationFreezerTest,
     OwnsTemporaryDesktopProtosAndRestoresPrecedingClock) {
  const absl::Time preceding_time = absl::FromCivil(
      absl::CivilSecond(2026, 8, 19, 10, 38, 0), absl::UTCTimeZone());
  ScopedClockMock preceding_clock(preceding_time);
  StrictMockConverter converter;
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(
          kEvaluationClockUtcRfc3339, MakeDefaultDesktopEvaluationRequest(),
          MakeDefaultDesktopEvaluationConfig(), converter);
  ASSERT_TRUE(session.ok()) << session.status();
  EXPECT_EQ(Clock::GetAbslTime(),
            absl::FromCivil(absl::CivilSecond(2000, 1, 1, 0, 0, 0),
                            absl::UTCTimeZone()));

  EXPECT_CALL(converter, ResetConversion(_)).WillOnce([](Segments* segments) {
    segments->Clear();
  });
  EXPECT_CALL(converter, StartConversion(_, _))
      .WillOnce([](const ConversionRequest& request, Segments* segments) {
        EXPECT_TRUE(request.options().defer_candidate_limits);
        EXPECT_EQ(request.key(), "m");
        EXPECT_TRUE(request.context().has_preceding_text());
        EXPECT_EQ(request.context().preceding_text(), "");
        EXPECT_TRUE(request.context().has_following_text());
        EXPECT_EQ(request.context().following_text(), "");
        Segment* segment = segments->add_segment();
        segment->set_key("m");
        AddCandidate(segment, "m", "候補", 101, 0x10, 3);
        AddCandidate(segment, "m",
                     absl::FormatTime("%H:%M", Clock::GetAbslTime(),
                                      Clock::GetTimeZone()),
                     202, 0x20, 6, converter::Candidate::ENABLE_INCOGNITO_MODE);
        return true;
      });

  absl::StatusOr<EvaluationFreezeCaseResult> result = (*session)->FreezeCase({
      .request_sequence = 17,
      .preedit_text = "ｍ",
      .preceding_text = "",
      .following_text = "",
      .reading_source = FrozenReadingSource::kPreeditText,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->conversion_query, "m");
  EXPECT_FALSE(result->history_reconstructed);
  EXPECT_EQ(result->baseline_output, "候補");
  EXPECT_EQ(result->request.reading(), "ｍ");
  EXPECT_EQ(result->request.token().session_generation(), 0);
  EXPECT_EQ(result->request.token().state_revision(), 0);
  EXPECT_EQ(result->request.token().request_sequence(), 17);
  ASSERT_EQ(result->request.segments_size(), 1);
  ASSERT_EQ(result->request.segments(0).candidates_size(), 2);
  EXPECT_EQ(result->request.segments(0).candidates(0).id(), 0);
  EXPECT_EQ(result->request.segments(0).candidates(1).id(), 1);
  EXPECT_EQ(result->request.segments(0).candidates(1).value(), "00:00");
  EXPECT_TRUE(result->request.segments(0).candidates(1).is_protected());

  session->reset();
  EXPECT_EQ(Clock::GetAbslTime(), preceding_time);
}

TEST(EvaluationFreezerTest,
     UsesConversionQueryAndPreservesFailedHistoryReconstruction) {
  commands::Request request = MakeDefaultDesktopEvaluationRequest();
  config::Config config = MakeDefaultDesktopEvaluationConfig();
  config.set_use_typing_correction(true);
  StrictMockConverter converter;
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(kEvaluationClockUtcRfc3339, request,
                                       config, converter);
  ASSERT_TRUE(session.ok()) << session.status();
  EXPECT_CALL(converter, ReconstructHistory(_, "前")).WillOnce(Return(false));
  EXPECT_CALL(converter, StartConversion(_, _))
      .WillOnce(
          [](const ConversionRequest& conversion_request, Segments* segments) {
            EXPECT_TRUE(conversion_request.options().defer_candidate_limits);
            EXPECT_TRUE(conversion_request.config().use_typing_correction());
            EXPECT_EQ(conversion_request.key(), "m");
            EXPECT_EQ(conversion_request.context().preceding_text(), "前");
            EXPECT_EQ(conversion_request.context().following_text(), "後");
            Segment* first = segments->add_segment();
            first->set_key("m");
            AddCandidate(first, "m", "第一", -101, 0x10, 3);
            AddCandidate(first, "m", "第壱", 202, 0x20, 6);
            Segment* second = segments->add_segment();
            second->set_key("next");
            AddCandidate(second, "next", "第二", 303, 0x40, 9);
            return true;
          });

  absl::StatusOr<EvaluationFreezeCaseResult> result = (*session)->FreezeCase({
      .request_sequence = 23,
      .preedit_text = "ｍ",
      .preceding_text = "前",
      .following_text = "後",
      .reading_source = FrozenReadingSource::kConversionQuery,
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->conversion_query, "m");
  EXPECT_FALSE(result->history_reconstructed);
  EXPECT_EQ(result->baseline_output, "第一第二");
  EXPECT_EQ(result->request.mode(),
            FrozenCandidateRankerRequest::MODE_CONVERSION);
  EXPECT_EQ(result->request.preceding_text(), "前");
  EXPECT_EQ(result->request.following_text(), "後");
  EXPECT_EQ(result->request.reading(), "m");
  EXPECT_EQ(result->request.focused_segment_id(), 0);
  ASSERT_EQ(result->request.segments_size(), 2);
  EXPECT_EQ(result->request.segments(0).id(), 0);
  EXPECT_EQ(result->request.segments(1).id(), 1);
  EXPECT_EQ(result->request.segments(0).candidates(0).id(), 0);
  EXPECT_EQ(result->request.segments(0).candidates(1).id(), 1);
  EXPECT_EQ(result->request.segments(1).candidates(0).id(), 2);
}

TEST(EvaluationFreezerTest, RejectsInvalidSessionAndCaseInputs) {
  StrictMockConverter converter;
  config::Config ranking_config = MakeDefaultDesktopEvaluationConfig();
  ranking_config.mutable_candidate_ranking_config()->set_enabled(false);
  EXPECT_EQ(
      EvaluationFreezerSession::Create(kEvaluationClockUtcRfc3339,
                                       MakeDefaultDesktopEvaluationRequest(),
                                       ranking_config, converter)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      EvaluationFreezerSession::Create(
          "2000-01-01T00:00:00+00:00", MakeDefaultDesktopEvaluationRequest(),
          MakeDefaultDesktopEvaluationConfig(), converter)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);

  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(
          kEvaluationClockUtcRfc3339, MakeDefaultDesktopEvaluationRequest(),
          MakeDefaultDesktopEvaluationConfig(), converter);
  ASSERT_TRUE(session.ok()) << session.status();
  EXPECT_EQ((*session)
                ->FreezeCase({
                    .request_sequence = 0,
                    .preedit_text = "reading",
                    .reading_source = FrozenReadingSource::kPreeditText,
                })
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*session)
                ->FreezeCase({
                    .request_sequence = 1,
                    .preedit_text = "reading",
                    .reading_source = FrozenReadingSource::kUnspecified,
                })
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(EvaluationFreezerTest, RejectsMissingSegmentsAndCandidates) {
  const auto create_session = [](const ConverterInterface& converter) {
    return EvaluationFreezerSession::Create(
        kEvaluationClockUtcRfc3339, MakeDefaultDesktopEvaluationRequest(),
        MakeDefaultDesktopEvaluationConfig(), converter);
  };
  const EvaluationFreezeCaseInput input = {
      .request_sequence = 1,
      .preedit_text = "reading",
      .reading_source = FrozenReadingSource::kConversionQuery,
  };

  StrictMockConverter no_segments_converter;
  EXPECT_CALL(no_segments_converter, ResetConversion(_));
  EXPECT_CALL(no_segments_converter, StartConversion(_, _))
      .WillOnce(Return(true));
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>>
      no_segments_session = create_session(no_segments_converter);
  ASSERT_TRUE(no_segments_session.ok()) << no_segments_session.status();
  EXPECT_EQ((*no_segments_session)->FreezeCase(input).status().code(),
            absl::StatusCode::kFailedPrecondition);
  no_segments_session->reset();

  StrictMockConverter no_candidates_converter;
  EXPECT_CALL(no_candidates_converter, ResetConversion(_));
  EXPECT_CALL(no_candidates_converter, StartConversion(_, _))
      .WillOnce([](const ConversionRequest&, Segments* segments) {
        segments->add_segment()->set_key("reading");
        return true;
      });
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>>
      no_candidates_session = create_session(no_candidates_converter);
  ASSERT_TRUE(no_candidates_session.ok()) << no_candidates_session.status();
  EXPECT_EQ((*no_candidates_session)->FreezeCase(input).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(EvaluationFreezerTest, DefaultDesktopIdentityIsStable) {
  absl::StatusOr<std::string> request_sha256 =
      DeterministicMessageSha256(MakeDefaultDesktopEvaluationRequest());
  absl::StatusOr<std::string> config_sha256 =
      DeterministicMessageSha256(MakeDefaultDesktopEvaluationConfig());
  ASSERT_TRUE(request_sha256.ok()) << request_sha256.status();
  ASSERT_TRUE(config_sha256.ok()) << config_sha256.status();
  EXPECT_EQ(*request_sha256,
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(*config_sha256,
            "376212704866775205cd89f831ca3504"
            "300eb5a2762609fe07081db84a5a6e7b");
}

}  // namespace
}  // namespace mozc::engine::evaluation
