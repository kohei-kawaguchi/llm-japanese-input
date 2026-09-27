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

#include <string>

#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "testing/gunit.h"

namespace mozc {
namespace {

TEST(CandidateRankingProtoTest, ConfigDefaultsToDisabledWithoutTimeout) {
  const config::Config config;

  EXPECT_FALSE(config.has_candidate_ranking_config());
  EXPECT_FALSE(config.candidate_ranking_config().enabled());
  EXPECT_FALSE(
      config.candidate_ranking_config().has_max_wait_millisec());
}

TEST(CandidateRankingProtoTest, ConfigRoundTripPreservesExplicitValues) {
  config::Config input;
  config::Config::CandidateRankingConfig* candidate_ranking =
      input.mutable_candidate_ranking_config();
  candidate_ranking->set_enabled(true);
  candidate_ranking->set_max_wait_millisec(41);

  std::string serialized;
  ASSERT_TRUE(input.SerializeToString(&serialized));
  config::Config output;
  ASSERT_TRUE(output.ParseFromString(serialized));

  ASSERT_TRUE(output.has_candidate_ranking_config());
  EXPECT_TRUE(output.candidate_ranking_config().enabled());
  EXPECT_TRUE(output.candidate_ranking_config().has_max_wait_millisec());
  EXPECT_EQ(output.candidate_ranking_config().max_wait_millisec(), 41);
}

TEST(CandidateRankingProtoTest, DiagnosticsDefaultToAbsent) {
  const commands::Output output;

  EXPECT_FALSE(output.has_candidate_ranking_diagnostics());
}

TEST(CandidateRankingProtoTest, DiagnosticsRoundTripPreservesStableFields) {
  commands::Output input;
  commands::Output::CandidateRankingDiagnostics* diagnostics =
      input.mutable_candidate_ranking_diagnostics();
  diagnostics->set_outcome(
      commands::Output::CandidateRankingDiagnostics::REJECTED);
  diagnostics->set_reason(
      commands::Output::CandidateRankingDiagnostics::INVALID_RESPONSE);
  diagnostics->set_session_generation(11);
  diagnostics->set_state_revision(12);
  diagnostics->set_request_sequence(13);
  diagnostics->set_elapsed_millisec(14);

  std::string serialized;
  ASSERT_TRUE(input.SerializeToString(&serialized));
  commands::Output output;
  ASSERT_TRUE(output.ParseFromString(serialized));

  ASSERT_TRUE(output.has_candidate_ranking_diagnostics());
  const commands::Output::CandidateRankingDiagnostics& parsed =
      output.candidate_ranking_diagnostics();
  EXPECT_EQ(parsed.outcome(),
            commands::Output::CandidateRankingDiagnostics::REJECTED);
  EXPECT_EQ(parsed.reason(),
            commands::Output::CandidateRankingDiagnostics::INVALID_RESPONSE);
  EXPECT_EQ(parsed.session_generation(), 11);
  EXPECT_EQ(parsed.state_revision(), 12);
  EXPECT_EQ(parsed.request_sequence(), 13);
  EXPECT_EQ(parsed.elapsed_millisec(), 14);
}

}  // namespace
}  // namespace mozc
