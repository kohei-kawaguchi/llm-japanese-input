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

#include "engine/evaluation/evaluation_freezer_runtime.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/clock_mock.h"
#include "base/file/temp_dir.h"
#include "base/file_util.h"
#include "base/system_util.h"
#include "converter/converter_interface.h"
#include "data_manager/data_manager.h"
#include "engine/engine.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_freezer.h"

namespace mozc::engine::evaluation {
namespace {

class ScopedEvaluationProfile {
 public:
  static absl::StatusOr<std::unique_ptr<ScopedEvaluationProfile>> Create() {
    absl::StatusOr<TempDirectory> profile =
        TempDirectory::Default().CreateTempDirectory();
    if (!profile.ok()) {
      return profile.status();
    }
    return std::unique_ptr<ScopedEvaluationProfile>(new ScopedEvaluationProfile(
        SystemUtil::GetUserProfileDirectory(), std::move(*profile)));
  }

  ScopedEvaluationProfile(const ScopedEvaluationProfile&) = delete;
  ScopedEvaluationProfile& operator=(const ScopedEvaluationProfile&) = delete;

  ~ScopedEvaluationProfile() {
    SystemUtil::SetUserProfileDirectory(previous_profile_);
  }

 private:
  ScopedEvaluationProfile(std::string previous_profile, TempDirectory profile)
      : previous_profile_(std::move(previous_profile)),
        profile_(std::move(profile)) {
    SystemUtil::SetUserProfileDirectory(profile_.path());
  }

  std::string previous_profile_;
  TempDirectory profile_;
};

}  // namespace

struct EvaluationFreezerRuntime::State {
  std::string data_bytes;
  std::unique_ptr<ScopedClockMock> fixed_clock;
  std::unique_ptr<ScopedEvaluationProfile> profile;
  std::unique_ptr<Engine> engine;
  std::shared_ptr<const ConverterInterface> converter;
  std::string data_sha256;
  std::string data_type;
  std::string evaluation_clock_utc_rfc3339;
};

absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>>
EvaluationFreezerRuntime::Create(
    absl::string_view data_path, absl::string_view expected_data_sha256,
    absl::string_view data_type,
    absl::string_view evaluation_clock_utc_rfc3339) {
  absl::StatusOr<absl::Time> evaluation_clock =
      ParseCanonicalEvaluationClockUtc(evaluation_clock_utc_rfc3339);
  if (!evaluation_clock.ok()) {
    return evaluation_clock.status();
  }
  absl::StatusOr<std::string> data_bytes = FileUtil::GetContents(data_path);
  if (!data_bytes.ok()) {
    return data_bytes.status();
  }
  const std::string actual_data_sha256 = Sha256Bytes(*data_bytes);
  if (actual_data_sha256 != expected_data_sha256) {
    return absl::FailedPreconditionError("mozc.data SHA256 mismatch");
  }

  auto state = std::make_unique<State>();
  state->data_bytes = std::move(*data_bytes);
  state->data_sha256 = actual_data_sha256;
  state->data_type = std::string(data_type);
  state->evaluation_clock_utc_rfc3339 =
      std::string(evaluation_clock_utc_rfc3339);
  state->fixed_clock = std::make_unique<ScopedClockMock>(*evaluation_clock);
  absl::StatusOr<std::unique_ptr<ScopedEvaluationProfile>> profile =
      ScopedEvaluationProfile::Create();
  if (!profile.ok()) {
    return profile.status();
  }
  state->profile = std::move(*profile);

  absl::StatusOr<std::unique_ptr<const DataManager>> data_manager =
      DataManager::CreateFromArray(
          state->data_bytes, DataManager::GetDataSetMagicNumber(data_type));
  if (!data_manager.ok()) {
    return data_manager.status();
  }
  absl::StatusOr<std::unique_ptr<Engine>> engine =
      Engine::CreateEngine(*std::move(data_manager));
  if (!engine.ok()) {
    return engine.status();
  }
  state->engine = std::move(*engine);
  state->converter = state->engine->GetConverter();
  if (state->converter == nullptr) {
    return absl::InternalError("evaluation converter is unavailable");
  }
  return std::unique_ptr<EvaluationFreezerRuntime>(
      new EvaluationFreezerRuntime(std::move(state)));
}

EvaluationFreezerRuntime::EvaluationFreezerRuntime(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

EvaluationFreezerRuntime::~EvaluationFreezerRuntime() = default;

const ConverterInterface& EvaluationFreezerRuntime::converter() const {
  return *state_->converter;
}

absl::string_view EvaluationFreezerRuntime::data_sha256() const {
  return state_->data_sha256;
}

absl::string_view EvaluationFreezerRuntime::data_type() const {
  return state_->data_type;
}

absl::string_view EvaluationFreezerRuntime::evaluation_clock_utc_rfc3339()
    const {
  return state_->evaluation_clock_utc_rfc3339;
}

}  // namespace mozc::engine::evaluation
