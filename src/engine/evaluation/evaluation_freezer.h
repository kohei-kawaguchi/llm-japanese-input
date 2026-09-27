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

#ifndef MOZC_ENGINE_EVALUATION_EVALUATION_FREEZER_H_
#define MOZC_ENGINE_EVALUATION_EVALUATION_FREEZER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "base/protobuf/message.h"
#include "converter/candidate_ranker.h"
#include "converter/converter_interface.h"
#include "engine/evaluation/evaluation_clock_util.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"

namespace mozc {
class ScopedClockMock;
}

namespace mozc::engine::evaluation {

enum class FrozenReadingSource : uint8_t {
  kUnspecified = 0,
  kPreeditText = 1,
  kConversionQuery = 2,
};

struct EvaluationFreezerDesktopIdentity {
  std::string request_sha256;
  std::string config_sha256;
};

struct EvaluationFreezeCaseInput {
  uint64_t request_sequence = 0;
  absl::string_view preedit_text;
  absl::string_view preceding_text;
  absl::string_view following_text;
  FrozenReadingSource reading_source = FrozenReadingSource::kUnspecified;
  converter::CandidateRankerMode mode =
      converter::CandidateRankerMode::kConversion;
};

struct EvaluationFreezeCaseResult {
  std::string conversion_query;
  bool history_reconstructed = false;
  std::string baseline_output;
  FrozenCandidateRankerRequest request;
};

absl::StatusOr<std::string> DeterministicMessageSha256(
    const protobuf::Message& message);

commands::Request MakeDefaultDesktopEvaluationRequest();
config::Config MakeDefaultDesktopEvaluationConfig();

absl::StatusOr<FrozenCandidateRankerRequest> FreezeCandidateRankerRequest(
    const converter::CandidateRankerRequest& request);

class EvaluationFreezerSession {
 public:
  static absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> Create(
      absl::string_view evaluation_clock_utc_rfc3339,
      const commands::Request& desktop_request,
      const config::Config& desktop_config,
      const ConverterInterface& converter);

  EvaluationFreezerSession(const EvaluationFreezerSession&) = delete;
  EvaluationFreezerSession& operator=(const EvaluationFreezerSession&) = delete;
  ~EvaluationFreezerSession();

  const EvaluationFreezerDesktopIdentity& desktop_identity() const {
    return desktop_identity_;
  }

  absl::StatusOr<EvaluationFreezeCaseResult> FreezeCase(
      const EvaluationFreezeCaseInput& input) const;

 private:
  EvaluationFreezerSession(absl::Time evaluation_clock,
                           const commands::Request& desktop_request,
                           const config::Config& desktop_config,
                           const ConverterInterface& converter,
                           EvaluationFreezerDesktopIdentity desktop_identity);

  commands::Request desktop_request_;
  config::Config desktop_config_;
  const ConverterInterface& converter_;
  EvaluationFreezerDesktopIdentity desktop_identity_;
  std::unique_ptr<ScopedClockMock> fixed_clock_;
};

}  // namespace mozc::engine::evaluation

#endif  // MOZC_ENGINE_EVALUATION_EVALUATION_FREEZER_H_
