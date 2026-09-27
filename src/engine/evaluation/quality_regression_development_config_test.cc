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

#include "engine/evaluation/quality_regression_development_config.h"

#include <array>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/file_util.h"
#include "base/protobuf/descriptor.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kFrozenSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kManifestSha256A[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kManifestSha256B[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kManifestSha256C[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

template <typename Message>
void AddUnknownField(Message* message) {
  std::string bytes;
  ASSERT_TRUE(message->SerializeToString(&bytes));
  bytes.append("\xa0\x06\x01", 3);
  ASSERT_TRUE(message->ParseFromString(bytes));
}

DevelopmentExecutionConfig MakeConfig() {
  DevelopmentExecutionConfig config;
  config.set_schema_version(1);
  config.set_frozen_corpus_schema_version(1);
  config.set_native_coverage_schema_version(1);
  config.set_frozen_corpus_sha256(kFrozenSha256);
  config.set_expected_case_count(2);
  config.set_model_manifest_schema_version(5);
  config.set_coverage_definition_version(1);
  config.set_permutation_definition_version(1);

  DevelopmentObjectiveSpec* objective = config.add_objectives();
  objective->set_objective(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITH_TARGET);
  objective->set_manifest_filename("structured-with-target.textproto");
  objective->set_manifest_sha256(kManifestSha256A);
  objective->set_selection_eligible(false);

  objective = config.add_objectives();
  objective->set_objective(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITHOUT_TARGET);
  objective->set_manifest_filename("structured-without-target.textproto");
  objective->set_manifest_sha256(kManifestSha256B);
  objective->set_selection_eligible(true);

  objective = config.add_objectives();
  objective->set_objective(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          NATURAL_TEXT_ONLY);
  objective->set_manifest_filename("natural-text-only.textproto");
  objective->set_manifest_sha256(kManifestSha256C);
  objective->set_selection_eligible(true);
  return config;
}

::mozc::engine::CandidateRankerModelManifest MakeManifest() {
  ::mozc::engine::CandidateRankerModelManifest manifest;
  manifest.set_schema_version(5);
  auto* scoring_template = manifest.mutable_scoring_template();
  scoring_template->set_version("continuation-test-v2");
  scoring_template->set_bos_token_id(1);
  scoring_template->set_terminal_token_id(2);
  scoring_template->set_mode_prefix("mode=");
  scoring_template->set_suggestion_mode("suggestion");
  scoring_template->set_prediction_mode("prediction");
  scoring_template->set_conversion_mode("conversion");
  scoring_template->set_field_separator("\n");
  scoring_template->set_reading_prefix("reading=");
  scoring_template->set_segment_key_prefix("segment=");
  scoring_template->set_text_prefix("text=");
  scoring_template->set_text_normalization(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::NONE);
  scoring_template->set_context_policy(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          MOZC_BASELINE_SUBSTITUTION);
  scoring_template->set_record_layout(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITH_TARGET);

  auto* policy = manifest.mutable_bounded_execution_policy();
  policy->set_candidate_window_policy(
      ::mozc::engine::CandidateRankerModelManifest::BoundedExecutionPolicy::
          MOZC_ORDER_UNPROTECTED_PREFIX);
  policy->set_capacity_reduction_policy(
      ::mozc::engine::CandidateRankerModelManifest::BoundedExecutionPolicy::
          LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT);
  policy->set_decode_packing_policy(
      ::mozc::engine::CandidateRankerModelManifest::BoundedExecutionPolicy::
          SEGMENT_ID_ORDER_GREEDY_WHOLE_SEGMENTS);

  auto* limits = manifest.mutable_limits();
  limits->set_max_selected_candidates_per_segment(26);
  limits->set_max_segments_per_decode(8);
  limits->set_max_sequences_per_decode(64);
  limits->set_per_record_byte_limit(32768);
  limits->set_per_record_token_limit(64);
  limits->set_per_decode_input_node_limit(4096);
  limits->set_per_decode_output_row_limit(512);
  limits->set_max_decode_output_logit_bytes(67108864);
  limits->set_max_request_segments(64);
  limits->set_max_request_candidates(2048);
  limits->set_max_request_string_bytes(1048576);

  auto* runtime = manifest.mutable_runtime();
  runtime->set_execution_device(
      ::mozc::engine::CandidateRankerModelManifest::Runtime::CPU);
  runtime->set_kv_storage(
      ::mozc::engine::CandidateRankerModelManifest::Runtime::UNIFIED);
  runtime->set_micro_batch_token_capacity(64);
  runtime->set_decode_thread_count(2);
  runtime->set_batch_thread_count(4);
  runtime->set_use_memory_map(true);
  runtime->set_use_memory_lock(false);
  runtime->set_check_tensors(true);

  auto* artifacts = manifest.mutable_artifacts();
  artifacts->set_source_model("synthetic/model");
  artifacts->set_source_revision(
      "0123456789abcdef0123456789abcdef01234567");
  artifacts->set_source_weight_sha256(kFrozenSha256);
  artifacts->set_tokenizer_sha256(kFrozenSha256);
  artifacts->set_gguf_file_name("synthetic-f16.gguf");
  artifacts->set_gguf_sha256(kFrozenSha256);
  artifacts->set_quantization("F16");
  artifacts->add_model_license_references("MODEL_LICENSE.txt");
  artifacts->set_runtime_name("synthetic-runtime");
  artifacts->set_runtime_revision(
      "fedcba9876543210fedcba9876543210fedcba98");
  artifacts->set_runtime_source_archive_sha256(kFrozenSha256);
  artifacts->add_runtime_license_references("RUNTIME_LICENSE.txt");
  return manifest;
}

std::array<::mozc::engine::CandidateRankerModelManifest, 3> MakeManifests() {
  std::array<::mozc::engine::CandidateRankerModelManifest, 3> manifests = {
      MakeManifest(), MakeManifest(), MakeManifest()};
  manifests[1].mutable_scoring_template()->set_record_layout(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITHOUT_TARGET);
  manifests[2].mutable_scoring_template()->set_record_layout(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          NATURAL_TEXT_ONLY);
  return manifests;
}

TEST(QualityRegressionDevelopmentConfigTest, PinsWireContract) {
  EXPECT_EQ(DEVELOPMENT_NATIVE_COVERAGE_DISPOSITION_UNSPECIFIED, 0);
  EXPECT_EQ(DEVELOPMENT_NATIVE_COVERAGE_SELECTED, 1);
  EXPECT_EQ(DEVELOPMENT_NATIVE_COVERAGE_OMITTED_CAPACITY, 2);
  EXPECT_EQ(DEVELOPMENT_NUMERIC_PROFILE_UNSPECIFIED, 0);
  EXPECT_EQ(DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED, 1);
  EXPECT_EQ(DEVELOPMENT_EVALUATION_LAYOUT_UNSPECIFIED, 0);
  EXPECT_EQ(DEVELOPMENT_EVALUATION_SHARED_TRIE, 1);

  const protobuf::Descriptor* config =
      DevelopmentExecutionConfig::descriptor();
  ASSERT_NE(config, nullptr);
  ASSERT_EQ(config->field_count(), 9);
  for (int field_index = 0; field_index < config->field_count();
       ++field_index) {
    EXPECT_EQ(config->field(field_index)->number(), field_index + 1);
  }
  const protobuf::Descriptor* objective =
      DevelopmentObjectiveSpec::descriptor();
  ASSERT_NE(objective, nullptr);
  ASSERT_EQ(objective->field_count(), 4);
  for (int field_index = 0; field_index < objective->field_count();
       ++field_index) {
    EXPECT_EQ(objective->field(field_index)->number(), field_index + 1);
  }
  EXPECT_EQ(NativeCoverageSegment::descriptor()->field_count(), 3);
  EXPECT_EQ(NativeCoverageCase::descriptor()->field_count(), 2);
  EXPECT_EQ(NativeCoverageMap::descriptor()->field_count(), 1);
  EXPECT_EQ(ObjectiveNativeCoverage::descriptor()->field_count(), 2);
  EXPECT_EQ(DevelopmentNativeCoverageSuite::descriptor()->field_count(), 5);
}

TEST(QualityRegressionDevelopmentConfigTest,
     ValidatesExactObjectiveOrderAndRecursiveUnknownFields) {
  DevelopmentExecutionConfig config = MakeConfig();
  EXPECT_TRUE(
      ValidateQualityRegressionDevelopmentExecutionConfig(config).ok());

  DevelopmentExecutionConfig invalid = config;
  invalid.mutable_objectives()->SwapElements(0, 1);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  invalid.mutable_objectives(0)->set_selection_eligible(true);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  invalid.mutable_objectives(1)->set_manifest_filename(
      invalid.objectives(0).manifest_filename());
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  invalid.mutable_objectives(2)->set_manifest_filename("directory/file");
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  invalid.set_model_manifest_schema_version(4);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  AddUnknownField(&invalid);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = config;
  AddUnknownField(invalid.mutable_objectives(0));
  EXPECT_EQ(ValidateQualityRegressionDevelopmentExecutionConfig(invalid).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionDevelopmentConfigTest,
     ValidatesParsedManifestsDifferOnlyByRecordLayout) {
  const DevelopmentExecutionConfig config = MakeConfig();
  std::array<::mozc::engine::CandidateRankerModelManifest, 3> manifests =
      MakeManifests();
  const std::array<std::string, 3> hashes = {
      kManifestSha256A, kManifestSha256B, kManifestSha256C};
  EXPECT_TRUE(ValidateQualityRegressionDevelopmentManifests(
                  config, manifests, hashes)
                  .ok());

  manifests[1].mutable_runtime()->set_decode_thread_count(3);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentManifests(
                config, manifests, hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);

  manifests = MakeManifests();
  manifests[1].mutable_scoring_template()->set_record_layout(
      ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
          NATURAL_TEXT_ONLY);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentManifests(
                config, manifests, hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);

  manifests = MakeManifests();
  AddUnknownField(manifests[2].mutable_scoring_template());
  EXPECT_EQ(ValidateQualityRegressionDevelopmentManifests(
                config, manifests, hashes)
                .code(),
            absl::StatusCode::kInvalidArgument);

  const std::array<std::string, 3> wrong_hashes = {
      kManifestSha256A, kManifestSha256A, kManifestSha256C};
  manifests = MakeManifests();
  EXPECT_EQ(ValidateQualityRegressionDevelopmentManifests(
                config, manifests, wrong_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(QualityRegressionDevelopmentConfigTest, CheckedConfigIsPinned) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "quality_regression_development_config.textproto"});
  absl::StatusOr<std::string> bytes = FileUtil::GetContents(path);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->size(), 1001);
  EXPECT_EQ(Sha256Bytes(*bytes),
            "c0bca67765f3ad26bc16495d4dde60daa"
            "07cd1d8cd37d3c12b46999e59f24a9f");

  DevelopmentExecutionConfig config;
  ASSERT_TRUE(ParseTextproto(*bytes, &config).ok());
  ASSERT_TRUE(ValidateQualityRegressionDevelopmentExecutionConfig(config).ok());
  EXPECT_EQ(config.frozen_corpus_sha256(),
            "44c9a08d748f2cef5c8428b1101f490c"
            "2719db34e533ba1cda3cee9b12b41c59");
  EXPECT_EQ(config.expected_case_count(), 219);
  EXPECT_EQ(config.model_manifest_schema_version(), 5);
  EXPECT_EQ(config.coverage_definition_version(), 1);
  EXPECT_EQ(config.permutation_definition_version(), 1);
  ASSERT_EQ(config.objectives_size(), 3);
  EXPECT_FALSE(config.objectives(0).selection_eligible());
  EXPECT_TRUE(config.objectives(1).selection_eligible());
  EXPECT_TRUE(config.objectives(2).selection_eligible());
  EXPECT_EQ(config.objectives(0).manifest_sha256(),
            "2209e8ebcc9876e9427d934f3b9d739a"
            "65fb498d2160d64d742fc719a45271eb");
  EXPECT_EQ(config.objectives(1).manifest_sha256(),
            "66c620673895d5462539b8c2073c8d4f"
            "1615c51c09895d85c9336246c0c08bff");
  EXPECT_EQ(config.objectives(2).manifest_sha256(),
            "be316282bcf1e031b55228a8f99c78dd"
            "a7b19961828c56fd5bf70f90881657f9");
}

}  // namespace
}  // namespace mozc::engine::evaluation
