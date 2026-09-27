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

#include "engine/evaluation/ajimee_freezer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "base/strings/japanese.h"
#include "converter/converter_interface.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/evaluation_freezer.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"

namespace mozc::engine::evaluation {
namespace {

absl::Status RequireSameSourceIdentity(
    const EvaluationSourceIdentity& actual,
    const EvaluationSourceIdentity& expected) {
  absl::StatusOr<std::string> actual_bytes = SerializeDeterministically(actual);
  if (!actual_bytes.ok()) {
    return actual_bytes.status();
  }
  absl::StatusOr<std::string> expected_bytes =
      SerializeDeterministically(expected);
  if (!expected_bytes.ok()) {
    return expected_bytes.status();
  }
  if (*actual_bytes != *expected_bytes) {
    return absl::FailedPreconditionError(
        "AJIMEE source identity does not match freezer config");
  }
  return absl::OkStatus();
}

absl::Status ValidateInputCorpus(const AjimeeInputCorpus& input_corpus,
                                 absl::string_view input_corpus_sha256,
                                 const AjimeeFreezerConfig& freezer_config) {
  if (!input_corpus.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE input corpus is incomplete");
  }
  if (input_corpus.schema_version() != kAjimeeInputCorpusSchemaVersion ||
      input_corpus.schema_version() !=
          freezer_config.input_corpus_schema_version()) {
    return absl::InvalidArgumentError("AJIMEE input corpus schema mismatch");
  }
  absl::Status status = ValidateAjimeeSourceIdentity(input_corpus.source());
  if (!status.ok()) {
    return status;
  }
  status = RequireSameSourceIdentity(input_corpus.source(),
                                     freezer_config.expected_source());
  if (!status.ok()) {
    return status;
  }
  if (!IsAjimeeLowercaseHex(input_corpus_sha256, 64) ||
      input_corpus_sha256 != freezer_config.input_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE input corpus SHA256 does not match freezer config");
  }
  if (input_corpus.cases_size() != freezer_config.expected_case_count()) {
    return absl::InvalidArgumentError("AJIMEE input case count mismatch");
  }

  uint64_t previous_source_index = 0;
  bool has_previous_source_index = false;
  for (const AjimeeInputCase& input_case : input_corpus.cases()) {
    if (input_case.complete_katakana_reading().empty()) {
      return absl::InvalidArgumentError("AJIMEE input reading is empty");
    }
    if (has_previous_source_index &&
        input_case.source_index() <= previous_source_index) {
      return absl::InvalidArgumentError(
          "AJIMEE input source indexes are not strictly increasing");
    }
    has_previous_source_index = true;
    previous_source_index = input_case.source_index();

    const bool has_context = !input_case.published_preceding_context().empty();
    if ((has_context && input_case.context_slice() !=
                            AjimeeInputCase::CONTEXT_SLICE_HAS_CONTEXT) ||
        (!has_context && input_case.context_slice() !=
                             AjimeeInputCase::CONTEXT_SLICE_NO_CONTEXT)) {
      return absl::InvalidArgumentError(
          "AJIMEE input context slice does not match context text");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<AjimeeFrozenCase> FreezeCase(
    const AjimeeInputCase& input_case, uint64_t ordinal,
    const EvaluationFreezerSession& session) {
  AjimeeFrozenCase frozen_case;
  frozen_case.set_source_index(input_case.source_index());
  const std::string normalized_reading =
      japanese::KatakanaToHiragana(input_case.complete_katakana_reading());
  if (normalized_reading.empty()) {
    return absl::InvalidArgumentError(
        "AJIMEE normalized Hiragana reading is empty");
  }
  frozen_case.set_normalized_hiragana_reading(normalized_reading);
  absl::StatusOr<EvaluationFreezeCaseResult> result = session.FreezeCase({
      .request_sequence = ordinal + 1,
      .preedit_text = normalized_reading,
      .preceding_text = input_case.published_preceding_context(),
      .following_text = "",
      .reading_source = FrozenReadingSource::kPreeditText,
  });
  if (!result.ok()) {
    return result.status();
  }
  frozen_case.set_history_reconstructed(result->history_reconstructed);
  frozen_case.set_baseline_output(result->baseline_output);
  *frozen_case.mutable_request() = std::move(result->request);
  return frozen_case;
}

}  // namespace

commands::Request MakeAjimeeDefaultDesktopRequest() {
  return MakeDefaultDesktopEvaluationRequest();
}

config::Config MakeAjimeeDefaultDesktopConfig() {
  return MakeDefaultDesktopEvaluationConfig();
}

absl::Status ValidateAjimeeFreezerInput(
    const AjimeeInputCorpus& input_corpus,
    absl::string_view input_corpus_sha256,
    const AjimeeFreezerConfig& freezer_config) {
  absl::Status status = ValidateAjimeeFreezerConfig(freezer_config);
  if (!status.ok()) {
    return status;
  }
  return ValidateInputCorpus(input_corpus, input_corpus_sha256, freezer_config);
}

absl::Status ValidateAjimeeFreezerConfig(
    const AjimeeFreezerConfig& freezer_config) {
  if (!freezer_config.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE freezer config is incomplete");
  }
  if (freezer_config.schema_version() != kAjimeeFreezerConfigSchemaVersion ||
      freezer_config.input_corpus_schema_version() !=
          kAjimeeInputCorpusSchemaVersion ||
      freezer_config.frozen_corpus_schema_version() !=
          kAjimeeFrozenCorpusSchemaVersion) {
    return absl::InvalidArgumentError("AJIMEE freezer config schema mismatch");
  }
  absl::Status status =
      ValidateAjimeeSourceIdentity(freezer_config.expected_source());
  if (!status.ok()) {
    return status;
  }
  if (freezer_config.expected_case_count() == 0 ||
      !IsAjimeeLowercaseHex(freezer_config.input_corpus_sha256(), 64) ||
      !IsAjimeeLowercaseHex(freezer_config.mozc_source_revision(), 40) ||
      freezer_config.mozc_data_type().empty() ||
      !IsAjimeeLowercaseHex(freezer_config.mozc_data_sha256(), 64) ||
      !IsAjimeeLowercaseHex(freezer_config.default_desktop_request_sha256(),
                            64) ||
      !IsAjimeeLowercaseHex(freezer_config.default_desktop_config_sha256(),
                            64)) {
    return absl::InvalidArgumentError("AJIMEE freezer config identity invalid");
  }
  absl::StatusOr<absl::Time> evaluation_clock = ParseAjimeeEvaluationClockUtc(
      freezer_config.evaluation_clock_utc_rfc3339());
  if (!evaluation_clock.ok()) {
    return evaluation_clock.status();
  }
  return absl::OkStatus();
}

absl::Status ValidateAjimeeFrozenCorpus(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeFreezerConfig& freezer_config) {
  absl::Status status = ValidateAjimeeFreezerConfig(freezer_config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  if (corpus.schema_version() !=
      freezer_config.frozen_corpus_schema_version()) {
    return absl::InvalidArgumentError("AJIMEE frozen corpus schema mismatch");
  }
  status = RequireSameSourceIdentity(corpus.source(),
                                     freezer_config.expected_source());
  if (!status.ok()) {
    return status;
  }
  if (corpus.input_corpus_sha256() != freezer_config.input_corpus_sha256() ||
      corpus.cases_size() != freezer_config.expected_case_count() ||
      corpus.mozc().source_revision() !=
          freezer_config.mozc_source_revision() ||
      corpus.mozc().data_type() != freezer_config.mozc_data_type() ||
      corpus.mozc().data_sha256() != freezer_config.mozc_data_sha256() ||
      corpus.mozc().default_desktop_request_sha256() !=
          freezer_config.default_desktop_request_sha256() ||
      corpus.mozc().default_desktop_config_sha256() !=
          freezer_config.default_desktop_config_sha256() ||
      corpus.mozc().evaluation_clock_utc_rfc3339() !=
          freezer_config.evaluation_clock_utc_rfc3339()) {
    return absl::FailedPreconditionError(
        "AJIMEE frozen corpus identity does not match freezer config");
  }
  return absl::OkStatus();
}

absl::StatusOr<AjimeeFrozenCorpus> FreezeAjimeeCorpus(
    const AjimeeInputCorpus& input_corpus,
    absl::string_view input_corpus_sha256,
    const AjimeeFreezerConfig& freezer_config,
    const commands::Request& desktop_request,
    const config::Config& desktop_config, const ConverterInterface& converter) {
  absl::Status status = ValidateAjimeeFreezerInput(
      input_corpus, input_corpus_sha256, freezer_config);
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(
          freezer_config.evaluation_clock_utc_rfc3339(), desktop_request,
          desktop_config, converter);
  if (!session.ok()) {
    return session.status();
  }
  const EvaluationFreezerDesktopIdentity& desktop_identity =
      (*session)->desktop_identity();
  if (desktop_identity.request_sha256 !=
          freezer_config.default_desktop_request_sha256() ||
      desktop_identity.config_sha256 !=
          freezer_config.default_desktop_config_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE default desktop Request or Config hash mismatch");
  }

  AjimeeFrozenCorpus corpus;
  corpus.set_schema_version(kAjimeeFrozenCorpusSchemaVersion);
  *corpus.mutable_source() = input_corpus.source();
  corpus.set_input_corpus_sha256(input_corpus_sha256);
  EvaluationMozcIdentity* mozc = corpus.mutable_mozc();
  mozc->set_source_revision(freezer_config.mozc_source_revision());
  mozc->set_data_type(freezer_config.mozc_data_type());
  mozc->set_data_sha256(freezer_config.mozc_data_sha256());
  mozc->set_default_desktop_request_sha256(desktop_identity.request_sha256);
  mozc->set_default_desktop_config_sha256(desktop_identity.config_sha256);
  mozc->set_evaluation_clock_utc_rfc3339(
      freezer_config.evaluation_clock_utc_rfc3339());

  for (int case_index = 0; case_index < input_corpus.cases_size();
       ++case_index) {
    absl::StatusOr<AjimeeFrozenCase> frozen_case =
        FreezeCase(input_corpus.cases(case_index),
                   static_cast<uint64_t>(case_index), *(*session));
    if (!frozen_case.ok()) {
      return absl::Status(
          frozen_case.status().code(),
          absl::StrCat("AJIMEE case ",
                       input_corpus.cases(case_index).source_index(), ": ",
                       frozen_case.status().message()));
    }
    *corpus.add_cases() = std::move(*frozen_case);
  }

  status = ValidateAjimeeFrozenCorpus(corpus, freezer_config);
  if (!status.ok()) {
    return status;
  }
  return corpus;
}

}  // namespace mozc::engine::evaluation
