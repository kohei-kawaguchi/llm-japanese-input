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

#include "engine/evaluation/quality_regression_development_scores.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "base/file_util.h"
#include "base/protobuf/descriptor.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_native_coverage.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

using ::mozc::engine::CandidateRankerCapacityLimit;
using ::mozc::engine::CandidateRankerCapacityResult;
using ::mozc::engine::CandidateRankerCapacityViolation;
using ::mozc::engine::CandidateRankerModelManifest;
using ::mozc::engine::CandidateRankerSegmentCapacityDiagnostic;
using ::mozc::engine::CandidateRankerSegmentCapacityDisposition;

constexpr char kFrozenSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kExecutionSha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kCoverageSha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kScoreConfigSha256[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr char kManifestSha256A[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kManifestSha256B[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr char kManifestSha256C[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr char kPrivacyReading[] =
    "PRIVATE_SCORE_READING_MUST_NOT_ENTER_ARTIFACT";
constexpr char kPrivacyKey[] = "PRIVATE_SCORE_KEY_MUST_NOT_ENTER_ARTIFACT";
constexpr char kPrivacyValue[] =
    "PRIVATE_SCORE_VALUE_MUST_NOT_ENTER_ARTIFACT";

template <typename Message>
void AddUnknownField(Message* message) {
  std::string bytes;
  ASSERT_TRUE(message->SerializeToString(&bytes));
  bytes.append("\xa0\x06\x01", 3);
  ASSERT_TRUE(message->ParseFromString(bytes));
}

void FillCorpusIdentity(QualityRegressionCorpusIdentity* identity) {
  identity->set_role(DEVELOPMENT);
  EvaluationSourceIdentity* source = identity->mutable_source();
  source->set_benchmark_name("Synthetic score core");
  source->set_source_revision("0123456789abcdef0123456789abcdef01234567");
  source->set_source_relative_path("synthetic/development.tsv");
  source->set_source_sha256(kFrozenSha256);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic score corpus.");
  identity->set_parser_definition_version(1);
  identity->set_normalization_definition_version(1);
  identity->set_import_config_sha256(kFrozenSha256);
}

void FillMozcIdentity(EvaluationMozcIdentity* identity) {
  identity->set_source_revision(
      "0123456789abcdef0123456789abcdef01234567");
  identity->set_data_type("synthetic");
  identity->set_data_sha256(kFrozenSha256);
  identity->set_default_desktop_request_sha256(kManifestSha256A);
  identity->set_default_desktop_config_sha256(kManifestSha256B);
  identity->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");
}

void AddCandidate(uint64_t id, bool is_protected,
                  FrozenCandidateRankerSegment* segment) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key(kPrivacyKey);
  candidate->set_value(id == 0 ? kPrivacyValue : "private-other-value");
  candidate->set_cost(static_cast<int32_t>(100 + id));
  candidate->set_attributes(0);
  candidate->set_consumed_key_size(3);
  candidate->set_is_protected(is_protected);
}

FrozenCandidateRankerRequest* InitializeRequest(
    uint64_t sequence, QualityRegressionFrozenCase* frozen_case) {
  FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
  request->mutable_token()->set_session_generation(0);
  request->mutable_token()->set_state_revision(0);
  request->mutable_token()->set_request_sequence(sequence);
  request->set_mode(FrozenCandidateRankerRequest::MODE_CONVERSION);
  request->set_preceding_text("");
  request->set_following_text("");
  request->set_reading(kPrivacyReading);
  request->set_focused_segment_id(0);
  return request;
}

QualityRegressionFrozenCorpus MakeFrozenCorpus() {
  QualityRegressionFrozenCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  corpus.set_input_corpus_sha256(kFrozenSha256);
  FillMozcIdentity(corpus.mutable_mozc());
  corpus.set_freezer_config_sha256(kFrozenSha256);

  QualityRegressionFrozenCase* first = corpus.add_cases();
  first->set_source_line(2);
  first->set_conversion_reading(kPrivacyReading);
  first->set_mozc_baseline_output(
      absl::StrCat(kPrivacyValue, "private-other-value"));
  FrozenCandidateRankerRequest* first_request = InitializeRequest(1, first);
  FrozenCandidateRankerSegment* rankable = first_request->add_segments();
  rankable->set_id(0);
  rankable->set_key(kPrivacyKey);
  AddCandidate(0, true, rankable);
  AddCandidate(1, false, rankable);
  AddCandidate(2, true, rankable);
  AddCandidate(3, false, rankable);
  AddCandidate(4, false, rankable);
  FrozenCandidateRankerSegment* unrankable = first_request->add_segments();
  unrankable->set_id(1);
  unrankable->set_key(kPrivacyKey);
  AddCandidate(5, false, unrankable);

  QualityRegressionFrozenCase* second = corpus.add_cases();
  second->set_source_line(7);
  second->set_conversion_reading(kPrivacyReading);
  second->set_mozc_baseline_output(kPrivacyValue);
  FrozenCandidateRankerRequest* second_request = InitializeRequest(2, second);
  rankable = second_request->add_segments();
  rankable->set_id(0);
  rankable->set_key(kPrivacyKey);
  AddCandidate(0, false, rankable);
  AddCandidate(1, false, rankable);
  return corpus;
}

DevelopmentExecutionConfig MakeExecutionConfig() {
  DevelopmentExecutionConfig config;
  config.set_schema_version(1);
  config.set_frozen_corpus_schema_version(1);
  config.set_native_coverage_schema_version(1);
  config.set_frozen_corpus_sha256(kFrozenSha256);
  config.set_expected_case_count(2);
  config.set_model_manifest_schema_version(5);
  config.set_coverage_definition_version(1);
  config.set_permutation_definition_version(1);
  const std::array<CandidateRankerModelManifest::ScoringTemplate::RecordLayout,
                   3>
      layouts = {
          CandidateRankerModelManifest::ScoringTemplate::
              STRUCTURED_WITH_TARGET,
          CandidateRankerModelManifest::ScoringTemplate::
              STRUCTURED_WITHOUT_TARGET,
          CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY,
      };
  const std::array<std::string, 3> hashes = {
      kManifestSha256A, kManifestSha256B, kManifestSha256C};
  for (size_t index = 0; index < layouts.size(); ++index) {
    DevelopmentObjectiveSpec* objective = config.add_objectives();
    objective->set_objective(layouts[index]);
    objective->set_manifest_filename(
        absl::StrCat("manifest-", index, ".textproto"));
    objective->set_manifest_sha256(hashes[index]);
    objective->set_selection_eligible(index != 0);
  }
  return config;
}

CandidateRankerModelManifest MakeManifest() {
  CandidateRankerModelManifest manifest;
  manifest.set_schema_version(5);
  auto* scoring = manifest.mutable_scoring_template();
  scoring->set_version("score-core-v1");
  scoring->set_bos_token_id(1);
  scoring->set_terminal_token_id(2);
  scoring->set_mode_prefix("mode=");
  scoring->set_suggestion_mode("suggestion");
  scoring->set_prediction_mode("prediction");
  scoring->set_conversion_mode("conversion");
  scoring->set_field_separator("\n");
  scoring->set_reading_prefix("reading=");
  scoring->set_segment_key_prefix("segment=");
  scoring->set_text_prefix("text=");
  scoring->set_text_normalization(
      CandidateRankerModelManifest::ScoringTemplate::NONE);
  scoring->set_context_policy(
      CandidateRankerModelManifest::ScoringTemplate::
          MOZC_BASELINE_SUBSTITUTION);
  scoring->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET);
  auto* policy = manifest.mutable_bounded_execution_policy();
  policy->set_candidate_window_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          MOZC_ORDER_UNPROTECTED_PREFIX);
  policy->set_capacity_reduction_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT);
  policy->set_decode_packing_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
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
  runtime->set_execution_device(CandidateRankerModelManifest::Runtime::CPU);
  runtime->set_kv_storage(CandidateRankerModelManifest::Runtime::UNIFIED);
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

std::array<CandidateRankerModelManifest, 3> MakeManifests() {
  std::array<CandidateRankerModelManifest, 3> manifests = {
      MakeManifest(), MakeManifest(), MakeManifest()};
  manifests[1].mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITHOUT_TARGET);
  manifests[2].mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY);
  return manifests;
}

CandidateRankerCapacityViolation MakeLimiter() {
  return CandidateRankerCapacityViolation{
      .limit = CandidateRankerCapacityLimit::kFullRecordTokens,
      .observed = 101,
      .allowed = 100,
  };
}

CandidateRankerCapacityResult MakeCapacityResult(
    const converter::CandidateRankerRequest& request) {
  CandidateRankerCapacityResult result;
  if (request.token.request_sequence == 1) {
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = 0,
        .unprotected_candidate_count = 3,
        .window_candidate_count = 3,
        .selected_candidate_count = 2,
        .omitted_by_window = 0,
        .omitted_by_capacity = 1,
        .disposition = CandidateRankerSegmentCapacityDisposition::kSelected,
        .first_limiter = MakeLimiter(),
    });
  } else {
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = 0,
        .unprotected_candidate_count = 2,
        .window_candidate_count = 2,
        .selected_candidate_count = 0,
        .omitted_by_window = 0,
        .omitted_by_capacity = 2,
        .disposition =
            CandidateRankerSegmentCapacityDisposition::kOmittedCapacity,
        .first_limiter = MakeLimiter(),
    });
  }
  return result;
}

DevelopmentScoreRunnerConfig MakeScoreConfig() {
  DevelopmentScoreRunnerConfig config;
  config.set_schema_version(1);
  config.set_score_artifact_schema_version(2);
  config.set_execution_config_sha256(kExecutionSha256);
  config.set_frozen_corpus_sha256(kFrozenSha256);
  config.set_native_coverage_suite_sha256(kCoverageSha256);
  config.set_score_definition_version(2);
  config.set_numeric_profile(DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED);
  config.set_layout(DEVELOPMENT_EVALUATION_SHARED_TRIE);
  return config;
}

struct Fixture {
  QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  DevelopmentExecutionConfig execution = MakeExecutionConfig();
  DevelopmentNativeCoverageSuite coverage;
  DevelopmentScoreRunnerConfig score_config = MakeScoreConfig();
  std::array<CandidateRankerModelManifest, 3> manifests = MakeManifests();
  std::array<std::string, 3> manifest_hashes = {
      kManifestSha256A, kManifestSha256B, kManifestSha256C};
};

Fixture MakeFixture() {
  Fixture fixture;
  const auto capacity_callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    return MakeCapacityResult(request);
  };
  absl::StatusOr<DevelopmentNativeCoverageSuite> coverage =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          fixture.frozen, kFrozenSha256, fixture.execution,
          kExecutionSha256, capacity_callback);
  EXPECT_TRUE(coverage.ok()) << coverage.status();
  if (coverage.ok()) {
    fixture.coverage = std::move(*coverage);
  }
  return fixture;
}

enum class ScoreScenario {
  kTie,
  kDistinct,
  kChangedPermutation,
  kNaN,
  kWrongResponse,
  kExtraScore,
  kMissingScore,
  kDuplicateScore,
  kUnselectedScore,
};

QualityRegressionDevelopmentEvaluationResult MakeScoreResult(
    const converter::CandidateRankerRequest& request,
    ScoreScenario scenario) {
  QualityRegressionDevelopmentEvaluationResult result;
  result.response.token = request.token;
  if (request.token.request_sequence != 1) {
    return result;
  }
  std::vector<uint64_t> request_selected_ids;
  for (const converter::CandidateRankerCandidate& candidate :
       request.segments[0].candidates) {
    if (!candidate.is_protected &&
        (candidate.id == 1 || candidate.id == 3)) {
      request_selected_ids.push_back(candidate.id);
    }
  }
  for (uint64_t candidate_id : request_selected_ids) {
    double score = candidate_id == 1 ? -0.0 : 0.0;
    if (scenario == ScoreScenario::kDistinct ||
        scenario == ScoreScenario::kChangedPermutation ||
        scenario == ScoreScenario::kWrongResponse) {
      score = candidate_id == 1 ? -1.0 : -2.0;
    }
    if (scenario == ScoreScenario::kChangedPermutation &&
        request_selected_ids.front() == 3 && candidate_id == 1) {
      score = -3.0;
    }
    if (scenario == ScoreScenario::kNaN && candidate_id == 1) {
      score = std::numeric_limits<double>::quiet_NaN();
    }
    result.candidate_scores.push_back(
        {.segment_id = 0,
         .candidate_id = candidate_id,
         .full_continuation_log_probability = score,
         .scored_continuation_token_count = 1});
  }
  converter::CandidateRankerSegmentOrder order{.segment_id = 0};
  if (scenario == ScoreScenario::kTie || scenario == ScoreScenario::kNaN ||
      scenario == ScoreScenario::kExtraScore) {
    order.candidate_ids = request_selected_ids;
  } else {
    order.candidate_ids = {1, 3};
  }
  if (scenario == ScoreScenario::kWrongResponse) {
    order.candidate_ids = {3, 1};
  }
  result.response.segment_orders.push_back(std::move(order));
  if (scenario == ScoreScenario::kExtraScore) {
    result.candidate_scores.push_back(
        {.segment_id = 0,
         .candidate_id = 4,
         .full_continuation_log_probability = -1.0,
         .scored_continuation_token_count = 1});
  } else if (scenario == ScoreScenario::kMissingScore) {
    result.candidate_scores.pop_back();
  } else if (scenario == ScoreScenario::kDuplicateScore) {
    result.candidate_scores[1] = result.candidate_scores[0];
  } else if (scenario == ScoreScenario::kUnselectedScore) {
    result.candidate_scores[1].candidate_id = 4;
  }
  return result;
}

absl::StatusOr<DevelopmentObjectiveScores> BuildScores(
    const Fixture& fixture, ScoreScenario scenario,
    std::vector<std::string>* events = nullptr) {
  const auto capacity_callback =
      [events](const DevelopmentObjectiveSpec&,
               const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    if (events != nullptr) {
      events->push_back("audit");
    }
    return MakeCapacityResult(request);
  };
  const auto score_callback =
      [events, scenario](const DevelopmentObjectiveSpec&,
                         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<QualityRegressionDevelopmentEvaluationResult> {
    if (events != nullptr) {
      events->push_back("score");
    }
    return MakeScoreResult(request, scenario);
  };
  return BuildQualityRegressionDevelopmentObjectiveScores(
      fixture.score_config, kScoreConfigSha256, fixture.execution,
      kExecutionSha256, fixture.frozen, kFrozenSha256, fixture.coverage,
      kCoverageSha256, fixture.manifests, fixture.manifest_hashes,
      capacity_callback, score_callback);
}

absl::Status ValidateScores(const Fixture& fixture,
                            const DevelopmentObjectiveScores& scores) {
  return ValidateQualityRegressionDevelopmentObjectiveScores(
      scores, fixture.score_config, kScoreConfigSha256, fixture.execution,
      kExecutionSha256, fixture.frozen, kFrozenSha256, fixture.coverage,
      kCoverageSha256, fixture.manifests, fixture.manifest_hashes);
}

TEST(QualityRegressionDevelopmentScoresTest, PinsWireContract) {
  const auto expect_sequential_tags = [](const protobuf::Descriptor* message,
                                         int field_count) {
    ASSERT_NE(message, nullptr);
    ASSERT_EQ(message->field_count(), field_count);
    for (int index = 0; index < field_count; ++index) {
      EXPECT_EQ(message->field(index)->number(), index + 1);
    }
  };
  const protobuf::Descriptor* config =
      DevelopmentScoreRunnerConfig::descriptor();
  expect_sequential_tags(config, 8);
  const protobuf::Descriptor* candidate =
      DevelopmentCandidateScore::descriptor();
  ASSERT_EQ(candidate->field_count(), 5);
  const std::array<int, 5> candidate_tags = {1, 2, 3, 5, 6};
  for (int index = 0; index < candidate->field_count(); ++index) {
    EXPECT_EQ(candidate->field(index)->number(), candidate_tags[index]);
  }
  EXPECT_EQ(candidate->field(3)->type(),
            protobuf::FieldDescriptor::TYPE_FIXED64);
  EXPECT_EQ(candidate->field(4)->type(),
            protobuf::FieldDescriptor::TYPE_UINT32);
  expect_sequential_tags(DevelopmentSegmentScores::descriptor(), 2);
  expect_sequential_tags(DevelopmentCaseScores::descriptor(), 2);
  expect_sequential_tags(DevelopmentObjectiveScoreSet::descriptor(), 2);
  expect_sequential_tags(DevelopmentObjectiveScoresIdentity::descriptor(), 7);
  expect_sequential_tags(DevelopmentObjectiveScores::descriptor(), 3);
}

TEST(QualityRegressionDevelopmentScoresTest,
     ValidatesConfigAndEveryRawIdentityJoin) {
  Fixture fixture = MakeFixture();
  EXPECT_TRUE(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                  fixture.score_config, kScoreConfigSha256,
                  fixture.execution, kExecutionSha256, fixture.frozen,
                  kFrozenSha256, fixture.coverage, kCoverageSha256,
                  fixture.manifests, fixture.manifest_hashes)
                  .ok());

  DevelopmentScoreRunnerConfig invalid = fixture.score_config;
  invalid.set_numeric_profile(DEVELOPMENT_NUMERIC_PROFILE_UNSPECIFIED);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerConfig(invalid)
                .code(),
            absl::StatusCode::kInvalidArgument);
  invalid = fixture.score_config;
  invalid.set_layout(DEVELOPMENT_EVALUATION_LAYOUT_UNSPECIFIED);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerConfig(invalid)
                .code(),
            absl::StatusCode::kInvalidArgument);
  invalid = fixture.score_config;
  AddUnknownField(&invalid);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerConfig(invalid)
                .code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                fixture.score_config, "not-a-sha256", fixture.execution,
                kExecutionSha256, fixture.frozen, kFrozenSha256,
                fixture.coverage, kCoverageSha256, fixture.manifests,
                fixture.manifest_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                fixture.score_config, kScoreConfigSha256, fixture.execution,
                kManifestSha256A, fixture.frozen, kFrozenSha256,
                fixture.coverage, kCoverageSha256, fixture.manifests,
                fixture.manifest_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                fixture.score_config, kScoreConfigSha256, fixture.execution,
                kExecutionSha256, fixture.frozen, kManifestSha256A,
                fixture.coverage, kCoverageSha256, fixture.manifests,
                fixture.manifest_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                fixture.score_config, kScoreConfigSha256, fixture.execution,
                kExecutionSha256, fixture.frozen, kFrozenSha256,
                fixture.coverage, kManifestSha256A, fixture.manifests,
                fixture.manifest_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);

  std::array<std::string, 3> wrong_manifest_hashes =
      fixture.manifest_hashes;
  wrong_manifest_hashes[1] = kManifestSha256A;
  EXPECT_EQ(ValidateQualityRegressionDevelopmentScoreRunnerInputs(
                fixture.score_config, kScoreConfigSha256, fixture.execution,
                kExecutionSha256, fixture.frozen, kFrozenSha256,
                fixture.coverage, kCoverageSha256, fixture.manifests,
                wrong_manifest_hashes)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(QualityRegressionDevelopmentScoresTest,
     AcceptsStableTiePermutationAndCanonicalizesSignedZero) {
  const Fixture fixture = MakeFixture();
  std::vector<std::string> events;
  absl::StatusOr<DevelopmentObjectiveScores> first =
      BuildScores(fixture, ScoreScenario::kTie, &events);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_EQ(events.size(), 18);
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_EQ(events[index], "audit");
  }
  for (size_t index = 6; index < events.size(); ++index) {
    EXPECT_EQ(events[index], "score");
  }
  ASSERT_EQ(first->objectives_size(), 3);
  for (const DevelopmentObjectiveScoreSet& objective : first->objectives()) {
    ASSERT_EQ(objective.cases_size(), 2);
    ASSERT_EQ(objective.cases(0).segments_size(), 1);
    const DevelopmentSegmentScores& segment = objective.cases(0).segments(0);
    ASSERT_EQ(segment.candidates_size(), 2);
    EXPECT_EQ(segment.candidates(0).candidate_id(), 1);
    EXPECT_EQ(segment.candidates(0).original_candidate_index(), 1);
    EXPECT_EQ(segment.candidates(0).mozc_cost(), 101);
    EXPECT_EQ(segment.candidates(0).full_continuation_score_bits(), 0);
    EXPECT_EQ(segment.candidates(0).scored_continuation_token_count(), 1);
    EXPECT_EQ(segment.candidates(1).candidate_id(), 3);
    EXPECT_EQ(segment.candidates(1).original_candidate_index(), 3);
    EXPECT_EQ(segment.candidates(1).mozc_cost(), 103);
    EXPECT_EQ(segment.candidates(1).full_continuation_score_bits(), 0);
    EXPECT_EQ(segment.candidates(1).scored_continuation_token_count(), 1);
    EXPECT_TRUE(objective.cases(1).segments().empty());
  }
  EXPECT_TRUE(ValidateScores(fixture, *first).ok());

  absl::StatusOr<DevelopmentObjectiveScores> second =
      BuildScores(fixture, ScoreScenario::kTie);
  ASSERT_TRUE(second.ok()) << second.status();
  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*first);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*second);
  absl::StatusOr<std::string> review = ReviewTextproto(*first);
  ASSERT_TRUE(first_binary.ok());
  ASSERT_TRUE(second_binary.ok());
  ASSERT_TRUE(review.ok());
  EXPECT_EQ(*first_binary, *second_binary);
  for (const std::string sentinel :
       {kPrivacyReading, kPrivacyKey, kPrivacyValue,
        "private-other-value"}) {
    EXPECT_EQ(first_binary->find(sentinel), std::string::npos);
    EXPECT_EQ(review->find(sentinel), std::string::npos);
  }
  DevelopmentObjectiveScores from_review;
  ASSERT_TRUE(ParseTextproto(*review, &from_review).ok());
  EXPECT_TRUE(ValidateScores(fixture, from_review).ok());
}

TEST(QualityRegressionDevelopmentScoresTest,
     AcceptsDistinctScoresAndRejectsPermutationOrResponseChanges) {
  const Fixture fixture = MakeFixture();
  absl::StatusOr<DevelopmentObjectiveScores> distinct =
      BuildScores(fixture, ScoreScenario::kDistinct);
  ASSERT_TRUE(distinct.ok()) << distinct.status();
  EXPECT_TRUE(ValidateScores(fixture, *distinct).ok());
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kChangedPermutation)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kWrongResponse)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kNaN).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kExtraScore).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kMissingScore).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kDuplicateScore).status().code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(BuildScores(fixture, ScoreScenario::kUnselectedScore)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionDevelopmentScoresTest,
     RejectsArtifactMetadataScoresAndRecursiveUnknownFields) {
  const Fixture fixture = MakeFixture();
  absl::StatusOr<DevelopmentObjectiveScores> built =
      BuildScores(fixture, ScoreScenario::kTie);
  ASSERT_TRUE(built.ok()) << built.status();

  DevelopmentObjectiveScores invalid = *built;
  invalid.mutable_objectives(0)
      ->mutable_cases(0)
      ->mutable_segments(0)
      ->mutable_candidates(0)
      ->set_original_candidate_index(3);
  EXPECT_EQ(ValidateScores(fixture, invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = *built;
  invalid.mutable_objectives(0)
      ->mutable_cases(0)
      ->mutable_segments(0)
      ->mutable_candidates(0)
      ->set_full_continuation_score_bits(std::bit_cast<uint64_t>(
          std::numeric_limits<double>::infinity()));
  EXPECT_EQ(ValidateScores(fixture, invalid).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = *built;
  invalid.mutable_identity()->set_native_coverage_suite_sha256(
      kManifestSha256A);
  EXPECT_EQ(ValidateScores(fixture, invalid).code(),
            absl::StatusCode::kFailedPrecondition);

  invalid = *built;
  AddUnknownField(invalid.mutable_objectives(0)
                      ->mutable_cases(0)
                      ->mutable_segments(0)
                      ->mutable_candidates(0));
  EXPECT_EQ(ValidateScores(fixture, invalid).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionDevelopmentScoresTest, CheckedConfigIsPinned) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "quality_regression_development_score_runner_config.textproto"});
  absl::StatusOr<std::string> bytes = FileUtil::GetContents(path);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->size(), 456);
  EXPECT_EQ(Sha256Bytes(*bytes),
            "dbe549df3b047f7c0efed38a1fa1088"
            "d3a305fd628a9d65fc688f23c4a01dc3b");
  DevelopmentScoreRunnerConfig config;
  ASSERT_TRUE(ParseTextproto(*bytes, &config).ok());
  EXPECT_TRUE(
      ValidateQualityRegressionDevelopmentScoreRunnerConfig(config).ok());
  EXPECT_EQ(config.execution_config_sha256(),
            "c0bca67765f3ad26bc16495d4dde60daa"
            "07cd1d8cd37d3c12b46999e59f24a9f");
  EXPECT_EQ(config.frozen_corpus_sha256(),
            "44c9a08d748f2cef5c8428b1101f490c"
            "2719db34e533ba1cda3cee9b12b41c59");
  EXPECT_EQ(config.native_coverage_suite_sha256(),
            "706bbe8c4c15e1cfb5fbec781afe28a3"
            "661ad3659c60d3dcabd03ba3912aad29");
}

}  // namespace
}  // namespace mozc::engine::evaluation
