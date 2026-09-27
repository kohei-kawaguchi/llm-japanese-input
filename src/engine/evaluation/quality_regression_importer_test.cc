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

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kRevision[] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

void FillSource(absl::string_view name, absl::string_view path,
                absl::string_view source_sha256,
                EvaluationSourceIdentity* source) {
  source->set_benchmark_name(name);
  source->set_source_revision(kRevision);
  source->set_source_relative_path(path);
  source->set_source_sha256(source_sha256);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic separated corpus.");
}

void FillPartition(QualityRegressionCorpusRole role,
                   absl::string_view source_name,
                   absl::string_view source_path,
                   absl::string_view source_bytes, uint32_t parsed_rows,
                   uint32_t conversion_expected_rows, uint32_t rank_zero_rows,
                   uint32_t duplicate_pairs, uint32_t final_cases,
                   QualityRegressionImportPartition* partition) {
  partition->set_role(role);
  FillSource(source_name, source_path, Sha256Bytes(source_bytes),
             partition->mutable_source());
  partition->set_expected_parsed_row_count(parsed_rows);
  partition->set_expected_conversion_expected_row_count(
      conversion_expected_rows);
  partition->set_expected_rank_zero_row_count(rank_zero_rows);
  partition->set_expected_duplicate_pair_count(duplicate_pairs);
  partition->set_expected_final_case_count(final_cases);
}

QualityRegressionImporterConfig MakeConfig(absl::string_view development,
                                           absl::string_view holdout) {
  QualityRegressionImporterConfig config;
  config.set_schema_version(kQualityRegressionCorpusSchemaVersion);
  config.set_input_corpus_schema_version(
      kQualityRegressionCorpusSchemaVersion);
  config.set_answer_corpus_schema_version(
      kQualityRegressionCorpusSchemaVersion);
  config.set_parser_definition_version(
      kQualityRegressionParserDefinitionVersion);
  config.set_normalization_definition_version(
      kQualityRegressionNormalizationDefinitionVersion);
  FillPartition(DEVELOPMENT, "Synthetic development", "synthetic/dev.tsv",
                development, 1, 1, 1, 0, 1, config.add_partitions());
  FillPartition(HOLDOUT, "Synthetic holdout", "synthetic/holdout.tsv", holdout,
                1, 1, 1, 0, 1, config.add_partitions());
  return config;
}

absl::StatusOr<std::string> ReadSource(
    std::initializer_list<absl::string_view> path) {
  return FileUtil::GetContents(testing::GetSourceFileOrDie(path));
}

absl::string_view PhysicalLine(absl::string_view source,
                               uint64_t one_based_line) {
  for (uint64_t line = 1; line < one_based_line; ++line) {
    const size_t separator = source.find('\n');
    if (separator == absl::string_view::npos) {
      return {};
    }
    source.remove_prefix(separator + 1);
  }
  const size_t separator = source.find('\n');
  return source.substr(0, separator);
}

TEST(QualityRegressionImporterTest,
     ImportsCheckedPartitionsWithExactCountsAndSeparatedAnswers) {
  absl::StatusOr<std::string> config_bytes = ReadSource(
      {"engine", "evaluation", "quality_regression_importer_config.textproto"});
  absl::StatusOr<std::string> development_source = ReadSource(
      {"data", "test", "quality_regression_test", "oss.tsv"});
  absl::StatusOr<std::string> holdout_source = ReadSource(
      {"data", "test", "quality_regression_test", "regression.tsv"});
  ASSERT_TRUE(config_bytes.ok()) << config_bytes.status();
  ASSERT_TRUE(development_source.ok()) << development_source.status();
  ASSERT_TRUE(holdout_source.ok()) << holdout_source.status();

  QualityRegressionImporterConfig config;
  ASSERT_TRUE(ParseTextproto(*config_bytes, &config).ok());
  ASSERT_TRUE(ValidateQualityRegressionImporterConfig(config).ok());
  ASSERT_EQ(config.partitions_size(), 2);
  EXPECT_EQ(config.partitions(0).expected_parsed_row_count(), 500);
  EXPECT_EQ(config.partitions(0).expected_conversion_expected_row_count(),
            408);
  EXPECT_EQ(config.partitions(0).expected_rank_zero_row_count(), 220);
  EXPECT_EQ(config.partitions(0).expected_duplicate_pair_count(), 1);
  EXPECT_EQ(config.partitions(0).expected_final_case_count(), 219);
  EXPECT_EQ(config.partitions(1).expected_parsed_row_count(), 101);
  EXPECT_EQ(config.partitions(1).expected_conversion_expected_row_count(),
            101);
  EXPECT_EQ(config.partitions(1).expected_rank_zero_row_count(), 72);
  EXPECT_EQ(config.partitions(1).expected_duplicate_pair_count(), 0);
  EXPECT_EQ(config.partitions(1).expected_final_case_count(), 72);
  EXPECT_EQ(PhysicalLine(*development_source, 340),
            "oss_issue950\tじょうきょうをちゅうしする\t状況を注視する\t"
            "Conversion Expected");
  EXPECT_EQ(PhysicalLine(*development_source, 469),
            "oss_issue\tじょうきょうをちゅうしする\t状況を注視する\t"
            "Conversion Expected");

  const std::string config_sha256 = Sha256Bytes(*config_bytes);
  EXPECT_EQ(config_sha256,
            "3cfb88ca988d9c46c3bca3095c9e8ee3"
            "b61cb43fce6df3c5be0d4ce100abbba5");
  absl::StatusOr<QualityRegressionCorpora> corpora =
      ImportQualityRegressionCorpora(*development_source, *holdout_source,
                                     config, config_sha256);
  ASSERT_TRUE(corpora.ok()) << corpora.status();
  ASSERT_EQ(corpora->development_inputs.cases_size(), 219);
  ASSERT_EQ(corpora->development_answers.cases_size(), 219);
  ASSERT_EQ(corpora->holdout_inputs.cases_size(), 72);
  ASSERT_EQ(corpora->holdout_answers.cases_size(), 72);

  EXPECT_EQ(corpora->development_inputs.identity().role(), DEVELOPMENT);
  EXPECT_EQ(corpora->holdout_inputs.identity().role(), HOLDOUT);
  EXPECT_EQ(corpora->development_inputs.identity().import_config_sha256(),
            config_sha256);
  EXPECT_EQ(corpora->development_answers.identity().import_config_sha256(),
            config_sha256);
  EXPECT_EQ(corpora->development_inputs.identity().parser_definition_version(),
            kQualityRegressionParserDefinitionVersion);
  EXPECT_EQ(
      corpora->development_inputs.identity().normalization_definition_version(),
      kQualityRegressionNormalizationDefinitionVersion);

  const auto contains_source_line = [](const auto& cases, uint64_t line) {
    return std::any_of(cases.begin(), cases.end(),
                       [line](const auto& item) {
                         return item.source_line() == line;
                       });
  };
  EXPECT_TRUE(contains_source_line(corpora->development_inputs.cases(), 340));
  EXPECT_FALSE(contains_source_line(corpora->development_inputs.cases(), 469));
  EXPECT_TRUE(contains_source_line(corpora->development_answers.cases(), 340));
  EXPECT_FALSE(
      contains_source_line(corpora->development_answers.cases(), 469));
  const auto input_340 = std::find_if(
      corpora->development_inputs.cases().begin(),
      corpora->development_inputs.cases().end(),
      [](const QualityRegressionInputCase& item) {
        return item.source_line() == 340;
      });
  const auto answer_340 = std::find_if(
      corpora->development_answers.cases().begin(),
      corpora->development_answers.cases().end(),
      [](const QualityRegressionAnswerCase& item) {
        return item.source_line() == 340;
      });
  ASSERT_NE(input_340, corpora->development_inputs.cases().end());
  ASSERT_NE(answer_340, corpora->development_answers.cases().end());
  EXPECT_EQ(input_340->reading(), "じょうきょうをちゅうしする");
  EXPECT_EQ(answer_340->normalized_whole_output(), "状況を注視する");

  EXPECT_EQ(QualityRegressionInputCase::descriptor()->FindFieldByName(
                "normalized_whole_output"),
            nullptr);
  EXPECT_EQ(QualityRegressionAnswerCase::descriptor()->FindFieldByName(
                "reading"),
            nullptr);
  absl::StatusOr<std::string> input_binary =
      SerializeDeterministically(corpora->development_inputs);
  absl::StatusOr<std::string> answer_binary =
      SerializeDeterministically(corpora->development_answers);
  ASSERT_TRUE(input_binary.ok()) << input_binary.status();
  ASSERT_TRUE(answer_binary.ok()) << answer_binary.status();
  EXPECT_EQ(input_binary->find("状況を注視する"), std::string::npos);
  EXPECT_NE(answer_binary->find("状況を注視する"), std::string::npos);

  absl::StatusOr<QualityRegressionCorpora> repeated =
      ImportQualityRegressionCorpora(*development_source, *holdout_source,
                                     config, config_sha256);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(*SerializeDeterministically(corpora->development_inputs),
            *SerializeDeterministically(repeated->development_inputs));
  EXPECT_EQ(*SerializeDeterministically(corpora->development_answers),
            *SerializeDeterministically(repeated->development_answers));
  EXPECT_EQ(*SerializeDeterministically(corpora->holdout_inputs),
            *SerializeDeterministically(repeated->holdout_inputs));
  EXPECT_EQ(*SerializeDeterministically(corpora->holdout_answers),
            *SerializeDeterministically(repeated->holdout_answers));
}

TEST(QualityRegressionImporterTest,
     PinsWindowsNormalizationAfterProductionTsvParsing) {
  const std::string development =
      "dev\tよみ〜−\t波〜−\tConversion Expected\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const QualityRegressionImporterConfig config =
      MakeConfig(development, holdout);
  const std::string config_sha256 = Sha256Bytes("synthetic config");

  absl::StatusOr<QualityRegressionCorpora> corpora =
      ImportQualityRegressionCorpora(development, holdout, config,
                                     config_sha256);
  ASSERT_TRUE(corpora.ok()) << corpora.status();
  ASSERT_EQ(corpora->development_answers.cases_size(), 1);
  EXPECT_EQ(corpora->development_answers.cases(0).normalized_whole_output(),
            "波～－");
  EXPECT_EQ(corpora->development_inputs.cases(0).reading(), "よみ〜−");
  absl::StatusOr<std::string> input_binary =
      SerializeDeterministically(corpora->development_inputs);
  absl::StatusOr<std::string> answer_binary =
      SerializeDeterministically(corpora->development_answers);
  ASSERT_TRUE(input_binary.ok());
  ASSERT_TRUE(answer_binary.ok());
  EXPECT_EQ(input_binary->find("波～－"), std::string::npos);
  EXPECT_NE(answer_binary->find("波～－"), std::string::npos);
}

TEST(QualityRegressionImporterTest,
     PublicCorpusValidatorsRejectSemanticAndRecursiveUnknownPoison) {
  const std::string development =
      "dev\tかいはつ\t開発\tConversion Expected\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const QualityRegressionImporterConfig config =
      MakeConfig(development, holdout);
  absl::StatusOr<QualityRegressionCorpora> corpora =
      ImportQualityRegressionCorpora(development, holdout, config,
                                     Sha256Bytes("synthetic config"));
  ASSERT_TRUE(corpora.ok()) << corpora.status();
  EXPECT_TRUE(
      ValidateQualityRegressionInputCorpus(corpora->development_inputs).ok());
  EXPECT_TRUE(ValidateQualityRegressionAnswerCorpus(
                  corpora->development_answers)
                  .ok());

  QualityRegressionInputCorpus invalid_input = corpora->development_inputs;
  invalid_input.mutable_identity()->set_import_config_sha256("not-a-hash");
  EXPECT_EQ(ValidateQualityRegressionInputCorpus(invalid_input).code(),
            absl::StatusCode::kInvalidArgument);

  invalid_input = corpora->development_inputs;
  invalid_input.mutable_cases(0)->set_source_line(0);
  EXPECT_EQ(ValidateQualityRegressionInputCorpus(invalid_input).code(),
            absl::StatusCode::kInvalidArgument);

  invalid_input = corpora->development_inputs;
  invalid_input.mutable_cases(0)->set_reading(std::string("\xff", 1));
  EXPECT_EQ(ValidateQualityRegressionInputCorpus(invalid_input).code(),
            absl::StatusCode::kInvalidArgument);

  QualityRegressionInputCase poisoned_case = invalid_input.cases(0);
  poisoned_case.set_reading("かいはつ");
  std::string poisoned_bytes;
  ASSERT_TRUE(poisoned_case.SerializeToString(&poisoned_bytes));
  poisoned_bytes.append("\xa0\x06\x01", 3);
  ASSERT_TRUE(poisoned_case.ParseFromString(poisoned_bytes));
  invalid_input = corpora->development_inputs;
  *invalid_input.mutable_cases(0) = poisoned_case;
  EXPECT_EQ(ValidateQualityRegressionInputCorpus(invalid_input).code(),
            absl::StatusCode::kInvalidArgument);

  QualityRegressionAnswerCorpus invalid_answer =
      corpora->development_answers;
  invalid_answer.mutable_identity()->set_role(
      QUALITY_REGRESSION_CORPUS_ROLE_UNSPECIFIED);
  EXPECT_EQ(ValidateQualityRegressionAnswerCorpus(invalid_answer).code(),
            absl::StatusCode::kInvalidArgument);

  invalid_answer = corpora->development_answers;
  invalid_answer.mutable_cases(0)->clear_normalized_whole_output();
  EXPECT_EQ(ValidateQualityRegressionAnswerCorpus(invalid_answer).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionImporterTest,
     RejectsHashCountPartitionOrderAndCrossPartitionOverlap) {
  const std::string development =
      "dev\tかいはつ\t開発\tConversion Expected\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const std::string config_sha256 = Sha256Bytes("synthetic config");
  QualityRegressionImporterConfig config = MakeConfig(development, holdout);

  EXPECT_EQ(ImportQualityRegressionCorpora(development + " ", holdout, config,
                                          config_sha256)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  QualityRegressionImporterConfig wrong_count = config;
  wrong_count.mutable_partitions(0)->set_expected_parsed_row_count(2);
  EXPECT_EQ(ImportQualityRegressionCorpora(development, holdout, wrong_count,
                                          config_sha256)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  QualityRegressionImporterConfig reordered = config;
  reordered.mutable_partitions()->SwapElements(0, 1);
  EXPECT_EQ(ValidateQualityRegressionImporterConfig(reordered).code(),
            absl::StatusCode::kInvalidArgument);

  const std::string overlapping_holdout =
      "holdout\tかいはつ\t別解\tConversion Expected\n";
  QualityRegressionImporterConfig overlap =
      MakeConfig(development, overlapping_holdout);
  EXPECT_EQ(ImportQualityRegressionCorpora(development, overlapping_holdout,
                                          overlap, config_sha256)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  const std::string answer_overlapping_holdout =
      "holdout\tべつよみ\t開発\tConversion Expected\n";
  overlap = MakeConfig(development, answer_overlapping_holdout);
  EXPECT_EQ(ImportQualityRegressionCorpora(development,
                                          answer_overlapping_holdout, overlap,
                                          config_sha256)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  EXPECT_EQ(ImportQualityRegressionCorpora(development, holdout, config,
                                          "not-a-sha256")
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionImporterTest, AuthenticatesBothSourcesBeforeParsing) {
  const std::string development_parse_poison = "invalid\trow\tonly\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const QualityRegressionImporterConfig config =
      MakeConfig(development_parse_poison, holdout);

  const absl::Status status = ImportQualityRegressionCorpora(
                                  development_parse_poison, holdout + " ",
                                  config, Sha256Bytes("synthetic config"))
                                  .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(status.message(), "quality-regression raw source SHA256 mismatch");
}

TEST(QualityRegressionImporterTest, RejectsEmptyRawFieldBeforeTsvParsing) {
  const std::string development =
      "dev\tかいはつ\t開発\tConversion Expected\t\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const QualityRegressionImporterConfig config =
      MakeConfig(development, holdout);

  const absl::Status status =
      ImportQualityRegressionCorpora(development, holdout, config,
                                     Sha256Bytes("synthetic config"))
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "quality-regression line 1 contains an empty field");
}

TEST(QualityRegressionImporterTest, RejectsDisabledParsedRow) {
  const std::string development =
      "DISABLED_dev\tかいはつ\t開発\tConversion Expected\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  const QualityRegressionImporterConfig config =
      MakeConfig(development, holdout);

  const absl::Status status =
      ImportQualityRegressionCorpora(development, holdout, config,
                                     Sha256Bytes("synthetic config"))
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "quality-regression line 1 is disabled");
}

TEST(QualityRegressionImporterTest, RejectsUnknownConfigFieldsRecursively) {
  const std::string development =
      "dev\tかいはつ\t開発\tConversion Expected\n";
  const std::string holdout =
      "holdout\tほじ\t保持\tConversion Expected\n";
  QualityRegressionImporterConfig config = MakeConfig(development, holdout);
  std::string bytes;
  ASSERT_TRUE(config.SerializeToString(&bytes));
  bytes.append("\xa0\x06\x01", 3);
  QualityRegressionImporterConfig with_unknown;
  ASSERT_TRUE(with_unknown.ParseFromString(bytes));
  EXPECT_EQ(ValidateQualityRegressionImporterConfig(with_unknown).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionImporterTest, ProductionConfigPinsSourceProvenance) {
  absl::StatusOr<std::string> config_bytes = ReadSource(
      {"engine", "evaluation", "quality_regression_importer_config.textproto"});
  ASSERT_TRUE(config_bytes.ok()) << config_bytes.status();
  QualityRegressionImporterConfig config;
  ASSERT_TRUE(ParseTextproto(*config_bytes, &config).ok());
  ASSERT_EQ(config.partitions_size(), 2);
  EXPECT_EQ(config.partitions(0).source().source_sha256(),
            "053a43e72d174da43b351278cadc471a"
            "be08cea9fe98829dafe3d152580fcff0");
  EXPECT_EQ(config.partitions(1).source().source_sha256(),
            "72c838c5422f04c8246114074ea9c82e"
            "577155e8d7ef653a21c2dbb75bb3dff3");
  EXPECT_EQ(config.partitions(0).source().source_revision(),
            "851c3fe33060d2a6090363e4d7ec44fafde2c03d");
  EXPECT_EQ(config.partitions(0).source().license_identifier(),
            "BSD-3-Clause");
}

}  // namespace
}  // namespace mozc::engine::evaluation
