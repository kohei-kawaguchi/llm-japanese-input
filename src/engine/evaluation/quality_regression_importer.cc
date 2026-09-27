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

#include "engine/evaluation/quality_regression_importer.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "base/strings/unicode.h"
#include "base/text_normalizer.h"
#include "converter/quality_regression_util.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"

namespace mozc::engine::evaluation {
namespace {

using ::mozc::quality_regression::QualityRegressionUtil;

constexpr absl::string_view kConversionExpected = "Conversion Expected";

struct SelectedCase {
  uint64_t source_line;
  std::string key;
  std::string expected_value;
};

struct ParsedSource {
  std::vector<SelectedCase> cases;
  uint32_t duplicate_pair_count;
};

absl::Status ValidateSourceConfig(
    const QualityRegressionImportPartition& config) {
  absl::Status status =
      ValidateQualityRegressionSourceIdentity(config.source());
  if (!status.ok()) {
    return status;
  }
  if (config.expected_parsed_row_count() == 0 ||
      config.expected_conversion_expected_row_count() == 0 ||
      config.expected_rank_zero_row_count() == 0 ||
      config.expected_final_case_count() == 0) {
    return absl::InvalidArgumentError(
        "quality-regression expected counts must be positive");
  }
  if (config.expected_conversion_expected_row_count() >
          config.expected_parsed_row_count() ||
      config.expected_rank_zero_row_count() >
          config.expected_conversion_expected_row_count() ||
      config.expected_final_case_count() >
          config.expected_rank_zero_row_count() ||
      config.expected_rank_zero_row_count() -
              config.expected_final_case_count() !=
          config.expected_duplicate_pair_count()) {
    return absl::InvalidArgumentError(
        "quality-regression expected counts are inconsistent");
  }
  return absl::OkStatus();
}

absl::StatusOr<ParsedSource> ParseSource(
    absl::string_view source,
    const QualityRegressionImportPartition& config) {
  if (!strings::IsValidUtf8(source)) {
    return absl::InvalidArgumentError(
        "quality-regression source must be valid UTF-8");
  }
  if (source.starts_with("\xef\xbb\xbf") ||
      source.find('\r') != absl::string_view::npos ||
      source.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "quality-regression source must be BOM-free LF-only UTF-8");
  }

  uint32_t parsed_rows = 0;
  uint32_t conversion_expected_rows = 0;
  uint32_t rank_zero_rows = 0;
  std::vector<SelectedCase> retained;
  uint32_t duplicate_pair_count = 0;
  std::map<std::pair<std::string, std::string>, uint64_t> first_line_by_pair;

  uint64_t source_line = 0;
  for (const absl::string_view line : absl::StrSplit(source, '\n')) {
    ++source_line;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::vector<absl::string_view> raw_tokens =
        absl::StrSplit(line, '\t');
    if (raw_tokens.size() < 4) {
      return absl::InvalidArgumentError(
          absl::StrCat("quality-regression line ", source_line,
                       " has fewer than four fields"));
    }
    for (const absl::string_view token : raw_tokens) {
      if (token.empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("quality-regression line ", source_line,
                         " contains an empty field"));
      }
    }
    QualityRegressionUtil::TestItem item;
    absl::Status status = item.ParseFromTSV(line);
    if (!status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("quality-regression line ", source_line,
                       " failed TestItem parsing: ", status.message()));
    }
    ++parsed_rows;
    if (item.label.rfind("DISABLED_", 0) == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("quality-regression line ", source_line,
                       " is disabled"));
    }
    const std::string normalized_expected =
        TextNormalizer::NormalizeTextWithFlag(raw_tokens[2],
                                              TextNormalizer::kAll);
    if (item.expected_value != normalized_expected) {
      return absl::FailedPreconditionError(absl::StrCat(
          "quality-regression line ", source_line,
          " production parser normalization does not match the pinned "
          "Windows rule"));
    }
    if (item.key.empty() || item.expected_value.empty() ||
        !strings::IsValidUtf8(item.key) ||
        !strings::IsValidUtf8(item.expected_value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("quality-regression line ", source_line,
                       " has an invalid key or expected value"));
    }
    if (item.command != kConversionExpected) {
      continue;
    }
    ++conversion_expected_rows;
    if (item.expected_rank != 0) {
      continue;
    }
    ++rank_zero_rows;

    const std::pair<std::string, std::string> pair(item.key,
                                                   item.expected_value);
    const bool inserted = first_line_by_pair.emplace(pair, source_line).second;
    if (!inserted) {
      ++duplicate_pair_count;
      continue;
    }
    retained.push_back(
        {.source_line = source_line,
         .key = std::move(item.key),
         .expected_value = std::move(item.expected_value)});
  }

  if (parsed_rows != config.expected_parsed_row_count() ||
      conversion_expected_rows !=
          config.expected_conversion_expected_row_count() ||
      rank_zero_rows != config.expected_rank_zero_row_count() ||
      retained.size() != config.expected_final_case_count()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "quality-regression source counts mismatch: parsed=", parsed_rows,
        ", conversion_expected=", conversion_expected_rows,
        ", rank_zero=", rank_zero_rows, ", retained=", retained.size()));
  }
  if (duplicate_pair_count != config.expected_duplicate_pair_count()) {
    return absl::FailedPreconditionError(
        "quality-regression duplicate count mismatch");
  }
  return ParsedSource{.cases = std::move(retained),
                      .duplicate_pair_count = duplicate_pair_count};
}

absl::Status RequireDisjoint(const ParsedSource& development,
                             const ParsedSource& holdout) {
  std::set<std::string> development_keys;
  std::set<std::string> development_answers;
  std::set<std::pair<std::string, std::string>> development_pairs;
  for (const SelectedCase& item : development.cases) {
    development_keys.insert(item.key);
    development_answers.insert(item.expected_value);
    development_pairs.emplace(item.key, item.expected_value);
  }
  for (const SelectedCase& item : holdout.cases) {
    if (development_keys.contains(item.key)) {
      return absl::FailedPreconditionError(
          "quality-regression development and holdout keys overlap");
    }
    if (development_answers.contains(item.expected_value)) {
      return absl::FailedPreconditionError(
          "quality-regression development and holdout answers overlap");
    }
    if (development_pairs.contains({item.key, item.expected_value})) {
      return absl::FailedPreconditionError(
          "quality-regression development and holdout pairs overlap");
    }
  }
  return absl::OkStatus();
}

void CopySourceIdentity(const EvaluationSourceIdentity& source,
                        EvaluationSourceIdentity* output) {
  output->set_benchmark_name(source.benchmark_name());
  output->set_source_revision(source.source_revision());
  output->set_source_relative_path(source.source_relative_path());
  output->set_source_sha256(source.source_sha256());
  output->set_creator(source.creator());
  output->set_source_url(source.source_url());
  output->set_license_identifier(source.license_identifier());
  output->set_license_url(source.license_url());
  output->set_upstream_dataset_url(source.upstream_dataset_url());
  output->set_changes_notice(source.changes_notice());
}

void FillIdentity(const QualityRegressionImportPartition& partition,
                  const QualityRegressionImporterConfig& config,
                  absl::string_view import_config_sha256,
                  QualityRegressionCorpusIdentity* identity) {
  identity->set_role(partition.role());
  CopySourceIdentity(partition.source(), identity->mutable_source());
  identity->set_parser_definition_version(config.parser_definition_version());
  identity->set_normalization_definition_version(
      config.normalization_definition_version());
  identity->set_import_config_sha256(import_config_sha256);
}

void FillCorpora(const ParsedSource& parsed,
                 const QualityRegressionImportPartition& partition,
                 const QualityRegressionImporterConfig& config,
                 absl::string_view import_config_sha256,
                 QualityRegressionInputCorpus* inputs,
                 QualityRegressionAnswerCorpus* answers) {
  inputs->set_schema_version(kQualityRegressionCorpusSchemaVersion);
  FillIdentity(partition, config, import_config_sha256,
               inputs->mutable_identity());
  answers->set_schema_version(kQualityRegressionCorpusSchemaVersion);
  FillIdentity(partition, config, import_config_sha256,
               answers->mutable_identity());
  for (const SelectedCase& item : parsed.cases) {
    QualityRegressionInputCase* input = inputs->add_cases();
    input->set_source_line(item.source_line);
    input->set_reading(item.key);
    QualityRegressionAnswerCase* answer = answers->add_cases();
    answer->set_source_line(item.source_line);
    answer->set_normalized_whole_output(item.expected_value);
  }
}

}  // namespace

absl::Status ValidateQualityRegressionImporterConfig(
    const QualityRegressionImporterConfig& config) {
  if (!config.IsInitialized()) {
    return absl::InvalidArgumentError(
        "quality-regression importer config is not initialized");
  }
  if (HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "quality-regression importer config has unknown fields");
  }
  if (config.schema_version() != kQualityRegressionCorpusSchemaVersion ||
      config.input_corpus_schema_version() !=
          kQualityRegressionCorpusSchemaVersion ||
      config.answer_corpus_schema_version() !=
          kQualityRegressionCorpusSchemaVersion ||
      config.parser_definition_version() !=
          kQualityRegressionParserDefinitionVersion ||
      config.normalization_definition_version() !=
          kQualityRegressionNormalizationDefinitionVersion) {
    return absl::InvalidArgumentError(
        "quality-regression importer contract version is invalid");
  }
  if (config.partitions_size() != 2 ||
      config.partitions(0).role() != DEVELOPMENT ||
      config.partitions(1).role() != HOLDOUT) {
    return absl::InvalidArgumentError(
        "quality-regression importer requires ordered development and holdout "
        "partitions");
  }
  absl::Status status = ValidateSourceConfig(config.partitions(0));
  if (!status.ok()) {
    return status;
  }
  status = ValidateSourceConfig(config.partitions(1));
  if (!status.ok()) {
    return status;
  }
  if (config.partitions(0).source().source_sha256() ==
          config.partitions(1).source().source_sha256() ||
      config.partitions(0).source().source_relative_path() ==
          config.partitions(1).source().source_relative_path()) {
    return absl::InvalidArgumentError(
        "development and holdout source identities must differ");
  }
  return absl::OkStatus();
}

absl::StatusOr<QualityRegressionCorpora> ImportQualityRegressionCorpora(
    absl::string_view development_source, absl::string_view holdout_source,
    const QualityRegressionImporterConfig& config,
    absl::string_view import_config_sha256) {
  absl::Status status = ValidateQualityRegressionImporterConfig(config);
  if (!status.ok()) {
    return status;
  }
  if (!IsQualityRegressionLowercaseHex(import_config_sha256, 64)) {
    return absl::InvalidArgumentError(
        "import config SHA256 must be a lowercase 64-digit hash");
  }
  const std::string development_sha256 = Sha256Bytes(development_source);
  const std::string holdout_sha256 = Sha256Bytes(holdout_source);
  if (development_sha256 != config.partitions(0).source().source_sha256() ||
      holdout_sha256 != config.partitions(1).source().source_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression raw source SHA256 mismatch");
  }
  absl::StatusOr<ParsedSource> development =
      ParseSource(development_source, config.partitions(0));
  if (!development.ok()) {
    return development.status();
  }
  absl::StatusOr<ParsedSource> holdout =
      ParseSource(holdout_source, config.partitions(1));
  if (!holdout.ok()) {
    return holdout.status();
  }
  status = RequireDisjoint(*development, *holdout);
  if (!status.ok()) {
    return status;
  }

  QualityRegressionCorpora result;
  FillCorpora(*development, config.partitions(0), config, import_config_sha256,
              &result.development_inputs, &result.development_answers);
  FillCorpora(*holdout, config.partitions(1), config, import_config_sha256,
              &result.holdout_inputs, &result.holdout_answers);
  status = ValidateQualityRegressionInputCorpus(result.development_inputs);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionAnswerCorpus(result.development_answers);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionInputCorpus(result.holdout_inputs);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionAnswerCorpus(result.holdout_answers);
  if (!status.ok()) {
    return status;
  }
  return result;
}

}  // namespace mozc::engine::evaluation
