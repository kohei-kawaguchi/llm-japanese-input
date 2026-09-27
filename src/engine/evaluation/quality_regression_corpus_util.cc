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

#include "engine/evaluation/quality_regression_corpus_util.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "base/strings/unicode.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"

namespace mozc::engine::evaluation {
absl::Status ValidateQualityRegressionCorpusIdentity(
    const QualityRegressionCorpusIdentity& identity) {
  if (!identity.IsInitialized() || HasUnknownFieldsRecursively(identity)) {
    return absl::InvalidArgumentError(
        "quality-regression corpus identity is not initialized or has "
        "unknown fields");
  }
  if (identity.role() != DEVELOPMENT && identity.role() != HOLDOUT) {
    return absl::InvalidArgumentError(
        "quality-regression corpus role is invalid");
  }
  absl::Status status =
      ValidateQualityRegressionSourceIdentity(identity.source());
  if (!status.ok()) {
    return status;
  }
  if (identity.parser_definition_version() !=
          kQualityRegressionParserDefinitionVersion ||
      identity.normalization_definition_version() !=
          kQualityRegressionNormalizationDefinitionVersion) {
    return absl::InvalidArgumentError(
        "quality-regression corpus definition version is invalid");
  }
  if (!IsQualityRegressionLowercaseHex(identity.import_config_sha256(), 64)) {
    return absl::InvalidArgumentError(
        "quality-regression import config SHA256 is invalid");
  }
  return absl::OkStatus();
}

bool IsQualityRegressionLowercaseHex(absl::string_view value, size_t size) {
  if (value.size() != size) {
    return false;
  }
  for (const char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

absl::Status ValidateQualityRegressionSourceIdentity(
    const EvaluationSourceIdentity& source) {
  if (!source.IsInitialized() || HasUnknownFieldsRecursively(source)) {
    return absl::InvalidArgumentError(
        "quality-regression source identity is not initialized or has "
        "unknown fields");
  }
  const absl::string_view fields[] = {
      source.benchmark_name(),       source.source_relative_path(),
      source.creator(),              source.source_url(),
      source.license_identifier(),   source.license_url(),
      source.upstream_dataset_url(), source.changes_notice(),
  };
  for (const absl::string_view field : fields) {
    if (field.empty() || !strings::IsValidUtf8(field)) {
      return absl::InvalidArgumentError(
          "quality-regression source identity strings must be nonempty "
          "valid UTF-8");
    }
  }
  if (!IsQualityRegressionLowercaseHex(source.source_revision(), 40)) {
    return absl::InvalidArgumentError(
        "quality-regression source revision must be a lowercase 40-digit "
        "hash");
  }
  if (!IsQualityRegressionLowercaseHex(source.source_sha256(), 64)) {
    return absl::InvalidArgumentError(
        "quality-regression source SHA256 must be a lowercase 64-digit hash");
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionInputCorpus(
    const QualityRegressionInputCorpus& corpus) {
  if (!corpus.IsInitialized() || HasUnknownFieldsRecursively(corpus)) {
    return absl::InvalidArgumentError(
        "quality-regression input corpus is not initialized or has unknown "
        "fields");
  }
  if (corpus.schema_version() != kQualityRegressionCorpusSchemaVersion) {
    return absl::InvalidArgumentError(
        "quality-regression input corpus schema is invalid");
  }
  absl::Status status =
      ValidateQualityRegressionCorpusIdentity(corpus.identity());
  if (!status.ok()) {
    return status;
  }
  if (corpus.cases().empty()) {
    return absl::InvalidArgumentError(
        "quality-regression input corpus must have cases");
  }
  uint64_t previous_source_line = 0;
  for (const QualityRegressionInputCase& item : corpus.cases()) {
    if (item.source_line() <= previous_source_line || item.reading().empty() ||
        !strings::IsValidUtf8(item.reading())) {
      return absl::InvalidArgumentError(
          "quality-regression input cases must have strictly increasing "
          "source lines and nonempty valid UTF-8 readings");
    }
    previous_source_line = item.source_line();
  }
  return absl::OkStatus();
}

absl::Status ValidateQualityRegressionAnswerCorpus(
    const QualityRegressionAnswerCorpus& corpus) {
  if (!corpus.IsInitialized() || HasUnknownFieldsRecursively(corpus)) {
    return absl::InvalidArgumentError(
        "quality-regression answer corpus is not initialized or has unknown "
        "fields");
  }
  if (corpus.schema_version() != kQualityRegressionCorpusSchemaVersion) {
    return absl::InvalidArgumentError(
        "quality-regression answer corpus schema is invalid");
  }
  absl::Status status =
      ValidateQualityRegressionCorpusIdentity(corpus.identity());
  if (!status.ok()) {
    return status;
  }
  if (corpus.cases().empty()) {
    return absl::InvalidArgumentError(
        "quality-regression answer corpus must have cases");
  }
  uint64_t previous_source_line = 0;
  for (const QualityRegressionAnswerCase& item : corpus.cases()) {
    if (item.source_line() <= previous_source_line ||
        item.normalized_whole_output().empty() ||
        !strings::IsValidUtf8(item.normalized_whole_output())) {
      return absl::InvalidArgumentError(
          "quality-regression answer cases must have strictly increasing "
          "source lines and nonempty valid UTF-8 outputs");
    }
    previous_source_line = item.source_line();
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
