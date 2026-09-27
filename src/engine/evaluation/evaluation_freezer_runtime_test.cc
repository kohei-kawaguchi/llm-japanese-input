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

#include "engine/evaluation/evaluation_freezer_runtime.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "base/clock.h"
#include "base/clock_mock.h"
#include "base/file/temp_dir.h"
#include "base/file_util.h"
#include "base/system_util.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kEvaluationClockUtcRfc3339[] = "2000-01-01T00:00:00Z";
constexpr char kInvalidData[] = "not a Mozc data set";

class EvaluationFreezerRuntimeTest : public testing::TestWithTempUserProfile {};

TEST_F(EvaluationFreezerRuntimeTest,
       RestoresCallerStateOnHashAndDataManagerFailures) {
  TempFile data_file = testing::MakeTempFileOrDie();
  ASSERT_TRUE(FileUtil::SetContents(data_file.path(), kInvalidData).ok());
  const std::string caller_profile = SystemUtil::GetUserProfileDirectory();
  const absl::Time caller_time = absl::FromCivil(
      absl::CivilSecond(2026, 8, 19, 10, 38, 0), absl::UTCTimeZone());
  ScopedClockMock caller_clock(caller_time);

  absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>> hash_mismatch =
      EvaluationFreezerRuntime::Create(data_file.path(),
                                       "00000000000000000000000000000000"
                                       "00000000000000000000000000000000",
                                       "oss", kEvaluationClockUtcRfc3339);
  ASSERT_FALSE(hash_mismatch.ok());
  EXPECT_EQ(hash_mismatch.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(hash_mismatch.status().message(), "mozc.data SHA256 mismatch");
  EXPECT_EQ(Clock::GetAbslTime(), caller_time);
  EXPECT_EQ(SystemUtil::GetUserProfileDirectory(), caller_profile);

  absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>> invalid_data =
      EvaluationFreezerRuntime::Create(data_file.path(),
                                       Sha256Bytes(kInvalidData), "oss",
                                       kEvaluationClockUtcRfc3339);
  ASSERT_FALSE(invalid_data.ok());
  EXPECT_EQ(invalid_data.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(Clock::GetAbslTime(), caller_time);
  EXPECT_EQ(SystemUtil::GetUserProfileDirectory(), caller_profile);
}

}  // namespace
}  // namespace mozc::engine::evaluation
