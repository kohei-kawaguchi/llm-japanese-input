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

#include "engine/evaluation/quality_regression_freezer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "converter/converter_interface.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/evaluation_freezer.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"

namespace mozc::engine::evaluation {
namespace {

void CopySourceIdentity(const EvaluationSourceIdentity& source,
                        EvaluationSourceIdentity* destination) {
  destination->set_benchmark_name(source.benchmark_name());
  destination->set_source_revision(source.source_revision());
  destination->set_source_relative_path(source.source_relative_path());
  destination->set_source_sha256(source.source_sha256());
  destination->set_creator(source.creator());
  destination->set_source_url(source.source_url());
  destination->set_license_identifier(source.license_identifier());
  destination->set_license_url(source.license_url());
  destination->set_upstream_dataset_url(source.upstream_dataset_url());
  destination->set_changes_notice(source.changes_notice());
}

void CopyCorpusIdentity(const QualityRegressionCorpusIdentity& source,
                        QualityRegressionCorpusIdentity* destination) {
  destination->set_role(source.role());
  CopySourceIdentity(source.source(), destination->mutable_source());
  destination->set_parser_definition_version(
      source.parser_definition_version());
  destination->set_normalization_definition_version(
      source.normalization_definition_version());
  destination->set_import_config_sha256(source.import_config_sha256());
}

void CopyMozcIdentity(const EvaluationMozcIdentity& source,
                      EvaluationMozcIdentity* destination) {
  destination->set_source_revision(source.source_revision());
  destination->set_data_type(source.data_type());
  destination->set_data_sha256(source.data_sha256());
  destination->set_default_desktop_request_sha256(
      source.default_desktop_request_sha256());
  destination->set_default_desktop_config_sha256(
      source.default_desktop_config_sha256());
  destination->set_evaluation_clock_utc_rfc3339(
      source.evaluation_clock_utc_rfc3339());
}

}  // namespace

commands::Request MakeQualityRegressionDefaultDesktopRequest() {
  return MakeDefaultDesktopEvaluationRequest();
}

config::Config MakeQualityRegressionDefaultDesktopConfig() {
  config::Config config = MakeDefaultDesktopEvaluationConfig();
  config.set_use_typing_correction(true);
  return config;
}

absl::StatusOr<QualityRegressionFrozenCorpus>
FreezeQualityRegressionCorpus(
    const QualityRegressionInputCorpus& input_corpus,
    absl::string_view input_corpus_sha256,
    const QualityRegressionFreezerConfig& freezer_config,
    absl::string_view freezer_config_sha256,
    const ConverterInterface& converter) {
  absl::Status status = ValidateQualityRegressionFreezerInput(
      input_corpus, input_corpus_sha256, freezer_config);
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(freezer_config_sha256, 64)) {
    return absl::InvalidArgumentError(
        "quality-regression freezer config SHA256 is invalid");
  }

  const commands::Request desktop_request =
      MakeQualityRegressionDefaultDesktopRequest();
  const config::Config desktop_config =
      MakeQualityRegressionDefaultDesktopConfig();
  if (desktop_config.has_candidate_ranking_config()) {
    return absl::InternalError(
        "quality-regression desktop Config has candidate ranking config");
  }
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(
          freezer_config.mozc().evaluation_clock_utc_rfc3339(),
          desktop_request, desktop_config, converter);
  if (!session.ok()) {
    return session.status();
  }
  const EvaluationFreezerDesktopIdentity& desktop_identity =
      (*session)->desktop_identity();
  if (desktop_identity.request_sha256 !=
          freezer_config.mozc().default_desktop_request_sha256() ||
      desktop_identity.config_sha256 !=
          freezer_config.mozc().default_desktop_config_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression desktop Request or Config SHA256 mismatch");
  }

  QualityRegressionFrozenCorpus corpus;
  corpus.set_schema_version(kQualityRegressionFrozenCorpusSchemaVersion);
  CopyCorpusIdentity(input_corpus.identity(), corpus.mutable_identity());
  corpus.set_input_corpus_sha256(input_corpus_sha256);
  CopyMozcIdentity(freezer_config.mozc(), corpus.mutable_mozc());
  corpus.mutable_mozc()->set_default_desktop_request_sha256(
      desktop_identity.request_sha256);
  corpus.mutable_mozc()->set_default_desktop_config_sha256(
      desktop_identity.config_sha256);
  corpus.set_freezer_config_sha256(freezer_config_sha256);

  for (int case_index = 0; case_index < input_corpus.cases_size();
       ++case_index) {
    const QualityRegressionInputCase& input_case =
        input_corpus.cases(case_index);
    absl::StatusOr<EvaluationFreezeCaseResult> result =
        (*session)->FreezeCase({
            .request_sequence = static_cast<uint64_t>(case_index) + 1,
            .preedit_text = input_case.reading(),
            .preceding_text = "",
            .following_text = "",
            .reading_source = FrozenReadingSource::kConversionQuery,
        });
    if (!result.ok()) {
      return absl::Status(
          result.status().code(),
          absl::StrCat("quality-regression case ", input_case.source_line(),
                       ": ", result.status().message()));
    }
    if (result->history_reconstructed) {
      return absl::InternalError(
          "quality-regression empty context reconstructed history");
    }
    QualityRegressionFrozenCase* frozen_case = corpus.add_cases();
    frozen_case->set_source_line(input_case.source_line());
    frozen_case->set_conversion_reading(result->conversion_query);
    frozen_case->set_mozc_baseline_output(result->baseline_output);
    *frozen_case->mutable_request() = std::move(result->request);
  }

  status = ValidateQualityRegressionFrozenCorpus(
      corpus, freezer_config, freezer_config_sha256);
  if (!status.ok()) {
    return status;
  }
  return corpus;
}

}  // namespace mozc::engine::evaluation
