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

#include "engine/evaluation/evaluation_freezer.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "base/clock_mock.h"
#include "composer/composer.h"
#include "composer/table.h"
#include "config/config_handler.h"
#include "converter/candidate_ranker.h"
#include "converter/segments.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "request/conversion_request.h"

namespace mozc::engine::evaluation {
namespace {

absl::StatusOr<FrozenCandidateRankerRequest::Mode> ToFrozenMode(
    converter::CandidateRankerMode mode) {
  switch (mode) {
    case converter::CandidateRankerMode::kSuggestion:
      return FrozenCandidateRankerRequest::MODE_SUGGESTION;
    case converter::CandidateRankerMode::kPrediction:
      return FrozenCandidateRankerRequest::MODE_PREDICTION;
    case converter::CandidateRankerMode::kConversion:
      return FrozenCandidateRankerRequest::MODE_CONVERSION;
  }
  return absl::InvalidArgumentError("candidate ranker mode is invalid");
}

}  // namespace

absl::StatusOr<std::string> DeterministicMessageSha256(
    const protobuf::Message& message) {
  absl::StatusOr<std::string> bytes = SerializeDeterministically(message);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return Sha256Bytes(*bytes);
}

commands::Request MakeDefaultDesktopEvaluationRequest() {
  return commands::Request();
}

config::Config MakeDefaultDesktopEvaluationConfig() {
  config::Config config = config::ConfigHandler::DefaultConfig();
  config.clear_candidate_ranking_config();
  return config;
}

absl::StatusOr<FrozenCandidateRankerRequest> FreezeCandidateRankerRequest(
    const converter::CandidateRankerRequest& request) {
  absl::StatusOr<FrozenCandidateRankerRequest::Mode> mode =
      ToFrozenMode(request.mode);
  if (!mode.ok()) {
    return mode.status();
  }

  FrozenCandidateRankerRequest frozen;
  FrozenCandidateRankerToken* token = frozen.mutable_token();
  token->set_session_generation(request.token.session_generation);
  token->set_state_revision(request.token.state_revision);
  token->set_request_sequence(request.token.request_sequence);
  frozen.set_mode(*mode);
  frozen.set_preceding_text(request.preceding_text);
  frozen.set_following_text(request.following_text);
  frozen.set_reading(request.reading);
  frozen.set_focused_segment_id(request.focused_segment_id);
  for (const converter::CandidateRankerSegment& source_segment :
       request.segments) {
    FrozenCandidateRankerSegment* destination_segment = frozen.add_segments();
    destination_segment->set_id(source_segment.id);
    destination_segment->set_key(source_segment.key);
    for (const converter::CandidateRankerCandidate& source_candidate :
         source_segment.candidates) {
      FrozenCandidateRankerCandidate* destination_candidate =
          destination_segment->add_candidates();
      destination_candidate->set_id(source_candidate.id);
      destination_candidate->set_key(source_candidate.key);
      destination_candidate->set_value(source_candidate.value);
      destination_candidate->set_cost(source_candidate.cost);
      destination_candidate->set_attributes(source_candidate.attributes);
      destination_candidate->set_consumed_key_size(
          source_candidate.consumed_key_size);
      destination_candidate->set_is_protected(source_candidate.is_protected);
    }
  }
  return frozen;
}

absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>>
EvaluationFreezerSession::Create(absl::string_view evaluation_clock_utc_rfc3339,
                                 const commands::Request& desktop_request,
                                 const config::Config& desktop_config,
                                 const ConverterInterface& converter) {
  absl::StatusOr<absl::Time> evaluation_clock =
      ParseCanonicalEvaluationClockUtc(evaluation_clock_utc_rfc3339);
  if (!evaluation_clock.ok()) {
    return evaluation_clock.status();
  }
  if (desktop_config.has_candidate_ranking_config()) {
    return absl::InvalidArgumentError(
        "evaluation desktop config must omit candidate ranking config");
  }
  absl::StatusOr<std::string> request_sha256 =
      DeterministicMessageSha256(desktop_request);
  if (!request_sha256.ok()) {
    return request_sha256.status();
  }
  absl::StatusOr<std::string> config_sha256 =
      DeterministicMessageSha256(desktop_config);
  if (!config_sha256.ok()) {
    return config_sha256.status();
  }
  return std::unique_ptr<EvaluationFreezerSession>(new EvaluationFreezerSession(
      *evaluation_clock, desktop_request, desktop_config, converter,
      EvaluationFreezerDesktopIdentity{
          .request_sha256 = std::move(*request_sha256),
          .config_sha256 = std::move(*config_sha256),
      }));
}

EvaluationFreezerSession::EvaluationFreezerSession(
    absl::Time evaluation_clock, const commands::Request& desktop_request,
    const config::Config& desktop_config, const ConverterInterface& converter,
    EvaluationFreezerDesktopIdentity desktop_identity)
    : desktop_request_(desktop_request),
      desktop_config_(desktop_config),
      converter_(converter),
      desktop_identity_(std::move(desktop_identity)),
      fixed_clock_(std::make_unique<ScopedClockMock>(evaluation_clock)) {}

EvaluationFreezerSession::~EvaluationFreezerSession() = default;

absl::StatusOr<EvaluationFreezeCaseResult> EvaluationFreezerSession::FreezeCase(
    const EvaluationFreezeCaseInput& input) const {
  if (input.request_sequence == 0 || input.preedit_text.empty()) {
    return absl::InvalidArgumentError(
        "evaluation request sequence and preedit text are required");
  }
  if (input.reading_source != FrozenReadingSource::kPreeditText &&
      input.reading_source != FrozenReadingSource::kConversionQuery) {
    return absl::InvalidArgumentError(
        "evaluation frozen reading source is invalid");
  }

  Segments segments;
  bool history_reconstructed = false;
  if (input.preceding_text.empty()) {
    converter_.ResetConversion(&segments);
  } else {
    history_reconstructed =
        converter_.ReconstructHistory(&segments, input.preceding_text);
  }

  commands::Context context;
  context.set_preceding_text(input.preceding_text);
  context.set_following_text(input.following_text);
  composer::Composer composer(std::make_shared<composer::Table>(),
                              desktop_request_, desktop_config_);
  composer.SetPreeditTextForTestOnly(input.preedit_text);
  ConversionRequest::Options options;
  options.defer_candidate_limits = true;
  const ConversionRequest conversion_request =
      ConversionRequestBuilder()
          .SetComposer(composer)
          .SetRequestView(desktop_request_)
          .SetContextView(context)
          .SetConfigView(desktop_config_)
          .SetOptions(options)
          .Build();
  if (!converter_.StartConversion(conversion_request, &segments)) {
    return absl::UnknownError("evaluation conversion failed");
  }
  if (segments.conversion_segments_size() == 0) {
    return absl::FailedPreconditionError(
        "evaluation conversion produced no conversion segments");
  }

  std::string baseline_output;
  for (const Segment& segment : segments.conversion_segments()) {
    if (segment.candidates_size() == 0) {
      return absl::FailedPreconditionError(
          "evaluation conversion segment produced no candidates");
    }
    baseline_output.append(segment.candidate(0).value);
  }

  const std::string conversion_query(conversion_request.key());
  const absl::string_view frozen_reading =
      input.reading_source == FrozenReadingSource::kPreeditText
          ? input.preedit_text
          : absl::string_view(conversion_query);
  const converter::CandidateRankerToken token = {
      .session_generation = 0,
      .state_revision = 0,
      .request_sequence = input.request_sequence,
  };
  const converter::CandidateRankerRequest ranker_request =
      converter::BuildCandidateRankerRequest(
          token, converter::CandidateRankerMode::kConversion, frozen_reading,
          input.preceding_text, input.following_text, 0, segments);
  absl::StatusOr<FrozenCandidateRankerRequest> frozen_request =
      FreezeCandidateRankerRequest(ranker_request);
  if (!frozen_request.ok()) {
    return frozen_request.status();
  }

  return EvaluationFreezeCaseResult{
      .conversion_query = conversion_query,
      .history_reconstructed = history_reconstructed,
      .baseline_output = std::move(baseline_output),
      .request = std::move(*frozen_request),
  };
}

}  // namespace mozc::engine::evaluation
