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

#include "engine/evaluation/quality_regression_native_coverage.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"
#include "engine/evaluation/quality_regression_development_config.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "testing/gunit.h"

namespace mozc::engine::evaluation {
namespace {

using ::mozc::engine::CandidateRankerCapacityLimit;
using ::mozc::engine::CandidateRankerCapacityResult;
using ::mozc::engine::CandidateRankerCapacityViolation;
using ::mozc::engine::CandidateRankerRequestLimit;
using ::mozc::engine::CandidateRankerRequestLimitViolation;
using ::mozc::engine::CandidateRankerSegmentCapacityDiagnostic;
using ::mozc::engine::CandidateRankerSegmentCapacityDisposition;

constexpr char kFrozenSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kExecutionConfigSha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kInputSha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kFreezerConfigSha256[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr char kSourceRevision[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr char kManifestSha256A[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kManifestSha256B[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr char kManifestSha256C[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr char kPrivacyReading[] =
    "PRIVATE_READING_MUST_NOT_ENTER_NATIVE_COVERAGE";
constexpr char kPrivacyKey[] =
    "PRIVATE_KEY_MUST_NOT_ENTER_NATIVE_COVERAGE";
constexpr char kPrivacyValue[] =
    "PRIVATE_VALUE_MUST_NOT_ENTER_NATIVE_COVERAGE";

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
  source->set_benchmark_name("Synthetic development native coverage");
  source->set_source_revision(kSourceRevision);
  source->set_source_relative_path("synthetic/development.tsv");
  source->set_source_sha256(kInputSha256);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic native coverage corpus.");
  identity->set_parser_definition_version(1);
  identity->set_normalization_definition_version(1);
  identity->set_import_config_sha256(kInputSha256);
}

void FillMozcIdentity(EvaluationMozcIdentity* identity) {
  identity->set_source_revision(kSourceRevision);
  identity->set_data_type("synthetic");
  identity->set_data_sha256(kInputSha256);
  identity->set_default_desktop_request_sha256(kManifestSha256A);
  identity->set_default_desktop_config_sha256(kManifestSha256B);
  identity->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");
}

void AddCandidate(uint64_t id, bool is_protected,
                  FrozenCandidateRankerSegment* segment) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key(kPrivacyKey);
  candidate->set_value(
      id == 0 ? kPrivacyValue : "private-nonbaseline-value");
  candidate->set_cost(static_cast<int32_t>(100 + id));
  candidate->set_attributes(0);
  candidate->set_consumed_key_size(3);
  candidate->set_is_protected(is_protected);
}

FrozenCandidateRankerRequest* InitializeRequest(
    uint64_t request_sequence, QualityRegressionFrozenCase* frozen_case) {
  FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
  request->mutable_token()->set_session_generation(0);
  request->mutable_token()->set_state_revision(0);
  request->mutable_token()->set_request_sequence(request_sequence);
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
  corpus.set_input_corpus_sha256(kInputSha256);
  FillMozcIdentity(corpus.mutable_mozc());
  corpus.set_freezer_config_sha256(kFreezerConfigSha256);

  QualityRegressionFrozenCase* first = corpus.add_cases();
  first->set_source_line(2);
  first->set_conversion_reading(kPrivacyReading);
  first->set_mozc_baseline_output(
      absl::StrCat(kPrivacyValue, "private-nonbaseline-value"));
  FrozenCandidateRankerRequest* first_request = InitializeRequest(1, first);
  FrozenCandidateRankerSegment* first_rankable =
      first_request->add_segments();
  first_rankable->set_id(0);
  first_rankable->set_key(kPrivacyKey);
  AddCandidate(0, true, first_rankable);
  AddCandidate(1, false, first_rankable);
  AddCandidate(2, true, first_rankable);
  AddCandidate(3, false, first_rankable);
  AddCandidate(4, false, first_rankable);
  FrozenCandidateRankerSegment* first_unrankable =
      first_request->add_segments();
  first_unrankable->set_id(1);
  first_unrankable->set_key(kPrivacyKey);
  AddCandidate(5, false, first_unrankable);

  QualityRegressionFrozenCase* second = corpus.add_cases();
  second->set_source_line(7);
  second->set_conversion_reading(kPrivacyReading);
  second->set_mozc_baseline_output(kPrivacyValue);
  FrozenCandidateRankerRequest* second_request = InitializeRequest(2, second);
  FrozenCandidateRankerSegment* second_rankable =
      second_request->add_segments();
  second_rankable->set_id(0);
  second_rankable->set_key(kPrivacyKey);
  AddCandidate(0, false, second_rankable);
  AddCandidate(1, false, second_rankable);
  return corpus;
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

CandidateRankerCapacityViolation MakeLimiter() {
  return CandidateRankerCapacityViolation{
      .limit = CandidateRankerCapacityLimit::kFullRecordTokens,
      .observed = 101,
      .allowed = 100,
  };
}

CandidateRankerCapacityResult MakeCapacityResult(
    const converter::CandidateRankerRequest& request,
    uint64_t first_selected_count) {
  CandidateRankerCapacityResult result;
  if (request.token.request_sequence == 1) {
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = 0,
        .unprotected_candidate_count = 3,
        .window_candidate_count = 3,
        .selected_candidate_count = first_selected_count,
        .omitted_by_window = 0,
        .omitted_by_capacity = 3 - first_selected_count,
        .disposition = CandidateRankerSegmentCapacityDisposition::kSelected,
        .first_limiter = first_selected_count == 3
                             ? std::nullopt
                             : std::optional<
                                   CandidateRankerCapacityViolation>(
                                   MakeLimiter()),
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

absl::Status ValidateSuite(
    const DevelopmentNativeCoverageSuite& suite,
    const QualityRegressionFrozenCorpus& frozen,
    const DevelopmentExecutionConfig& config) {
  return ValidateQualityRegressionDevelopmentNativeCoverageSuite(
      suite, frozen, kFrozenSha256, config, kExecutionConfigSha256);
}

TEST(QualityRegressionNativeCoverageTest,
     BuildsDeterministicOriginalOrderAggregateOnlyArtifact) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const DevelopmentExecutionConfig config = MakeConfig();
  int callback_count = 0;
  std::vector<int> objective_numbers;
  const auto callback =
      [&callback_count, &objective_numbers](
          const DevelopmentObjectiveSpec& objective,
          const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    ++callback_count;
    objective_numbers.push_back(objective.objective());
    EXPECT_EQ(request.reading, kPrivacyReading);
    return MakeCapacityResult(request, 2);
  };
  absl::StatusOr<DevelopmentNativeCoverageSuite> first =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          frozen, kFrozenSha256, config, kExecutionConfigSha256, callback);
  absl::StatusOr<DevelopmentNativeCoverageSuite> second =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          frozen, kFrozenSha256, config, kExecutionConfigSha256, callback);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(callback_count, 12);
  ASSERT_EQ(objective_numbers.size(), 12);
  EXPECT_EQ(objective_numbers[0],
            ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
                STRUCTURED_WITH_TARGET);
  EXPECT_EQ(objective_numbers[2],
            ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
                STRUCTURED_WITHOUT_TARGET);
  EXPECT_EQ(objective_numbers[4],
            ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
                NATURAL_TEXT_ONLY);
  EXPECT_EQ(objective_numbers[6], objective_numbers[0]);
  ASSERT_EQ(first->coverages_size(), 3);
  const NativeCoverageMap& map = first->coverages(0).map();
  ASSERT_EQ(map.cases_size(), 2);
  ASSERT_EQ(map.cases(0).segments_size(), 1);
  const NativeCoverageSegment& selected = map.cases(0).segments(0);
  EXPECT_EQ(selected.segment_id(), 0);
  EXPECT_EQ(selected.disposition(), DEVELOPMENT_NATIVE_COVERAGE_SELECTED);
  ASSERT_EQ(selected.selected_candidate_ids_size(), 2);
  EXPECT_EQ(selected.selected_candidate_ids(0), 1);
  EXPECT_EQ(selected.selected_candidate_ids(1), 3);
  ASSERT_EQ(map.cases(1).segments_size(), 1);
  EXPECT_EQ(map.cases(1).segments(0).disposition(),
            DEVELOPMENT_NATIVE_COVERAGE_OMITTED_CAPACITY);
  EXPECT_TRUE(map.cases(1).segments(0).selected_candidate_ids().empty());
  EXPECT_TRUE(ValidateSuite(*first, frozen, config).ok());
  EXPECT_TRUE(ValidateSuite(*second, frozen, config).ok());

  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*first);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*second);
  absl::StatusOr<std::string> review = ReviewTextproto(*first);
  ASSERT_TRUE(first_binary.ok()) << first_binary.status();
  ASSERT_TRUE(second_binary.ok()) << second_binary.status();
  ASSERT_TRUE(review.ok()) << review.status();
  EXPECT_EQ(*first_binary, *second_binary);
  EXPECT_EQ(Sha256Bytes(*first_binary),
            "5901c5d8442125cad9811accb282111f"
            "2dae6f3fb0b62e858c9d81d21d4bc0d6");
  for (const std::string private_value :
       {kPrivacyReading, kPrivacyKey, kPrivacyValue,
        "private-nonbaseline-value"}) {
    EXPECT_EQ(first_binary->find(private_value), std::string::npos);
    EXPECT_EQ(review->find(private_value), std::string::npos);
  }

  DevelopmentNativeCoverageSuite from_review;
  ASSERT_TRUE(ParseTextproto(*review, &from_review).ok());
  EXPECT_TRUE(ValidateSuite(from_review, frozen, config).ok());
  absl::StatusOr<std::string> review_binary =
      SerializeDeterministically(from_review);
  ASSERT_TRUE(review_binary.ok()) << review_binary.status();
  EXPECT_EQ(*review_binary, *first_binary);
}

TEST(QualityRegressionNativeCoverageTest,
     RejectsObjectiveDifferencesAndInvalidCapacityResults) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const DevelopmentExecutionConfig config = MakeConfig();
  const auto differing_callback =
      [](const DevelopmentObjectiveSpec& objective,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    const uint64_t selected_count =
        objective.objective() ==
                ::mozc::engine::CandidateRankerModelManifest::ScoringTemplate::
                    NATURAL_TEXT_ONLY
            ? 3
            : 2;
    return MakeCapacityResult(request, selected_count);
  };
  EXPECT_EQ(BuildQualityRegressionDevelopmentNativeCoverageSuite(
                frozen, kFrozenSha256, config, kExecutionConfigSha256,
                differing_callback)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  const auto request_limit_callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    CandidateRankerCapacityResult result = MakeCapacityResult(request, 2);
    result.request_limit_violation = CandidateRankerRequestLimitViolation{
        .limit = CandidateRankerRequestLimit::kCandidates,
        .observed = 6,
        .allowed = 5,
    };
    return result;
  };
  EXPECT_EQ(BuildQualityRegressionDevelopmentNativeCoverageSuite(
                frozen, kFrozenSha256, config, kExecutionConfigSha256,
                request_limit_callback)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);

  const auto wrong_segment_callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    CandidateRankerCapacityResult result = MakeCapacityResult(request, 2);
    result.segments[0].segment_id = 9;
    return result;
  };
  EXPECT_EQ(BuildQualityRegressionDevelopmentNativeCoverageSuite(
                frozen, kFrozenSha256, config, kExecutionConfigSha256,
                wrong_segment_callback)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  const auto wrong_count_callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    CandidateRankerCapacityResult result = MakeCapacityResult(request, 2);
    result.segments.clear();
    return result;
  };
  EXPECT_EQ(BuildQualityRegressionDevelopmentNativeCoverageSuite(
                frozen, kFrozenSha256, config, kExecutionConfigSha256,
                wrong_count_callback)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionNativeCoverageTest,
     RejectsMutatedOrRecursivelyUnknownArtifacts) {
  QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const DevelopmentExecutionConfig config = MakeConfig();
  const auto callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    return MakeCapacityResult(request, 2);
  };
  absl::StatusOr<DevelopmentNativeCoverageSuite> built =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          frozen, kFrozenSha256, config, kExecutionConfigSha256, callback);
  ASSERT_TRUE(built.ok()) << built.status();

  DevelopmentNativeCoverageSuite invalid = *built;
  invalid.mutable_coverages(0)
      ->mutable_map()
      ->mutable_cases(0)
      ->mutable_segments(0)
      ->set_selected_candidate_ids(0, 3);
  EXPECT_EQ(ValidateSuite(invalid, frozen, config).code(),
            absl::StatusCode::kInvalidArgument);

  invalid = *built;
  invalid.mutable_coverages(1)
      ->mutable_map()
      ->mutable_cases(0)
      ->mutable_segments(0)
      ->add_selected_candidate_ids(4);
  EXPECT_EQ(ValidateSuite(invalid, frozen, config).code(),
            absl::StatusCode::kFailedPrecondition);

  invalid = *built;
  AddUnknownField(invalid.mutable_coverages(0)
                      ->mutable_map()
                      ->mutable_cases(0)
                      ->mutable_segments(0));
  EXPECT_EQ(ValidateSuite(invalid, frozen, config).code(),
            absl::StatusCode::kInvalidArgument);

  AddUnknownField(frozen.mutable_cases(0)
                      ->mutable_request()
                      ->mutable_segments(0)
                      ->mutable_candidates(0));
  EXPECT_EQ(ValidateSuite(*built, frozen, config).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(QualityRegressionNativeCoverageTest,
     BuildsAndValidatesTheExactSelectedObjectPermutation) {
  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const DevelopmentExecutionConfig config = MakeConfig();
  const auto callback =
      [](const DevelopmentObjectiveSpec&,
         const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    return MakeCapacityResult(request, 2);
  };
  absl::StatusOr<DevelopmentNativeCoverageSuite> suite =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          frozen, kFrozenSha256, config, kExecutionConfigSha256, callback);
  ASSERT_TRUE(suite.ok()) << suite.status();
  absl::StatusOr<converter::CandidateRankerRequest> original =
      ConvertFrozenCandidateRankerRequest(frozen.cases(0).request());
  ASSERT_TRUE(original.ok()) << original.status();
  const NativeCoverageCase& accepted =
      suite->coverages(0).map().cases(0);
  absl::StatusOr<converter::CandidateRankerRequest> permuted =
      BuildQualityRegressionFixedPermutationRequest(accepted, *original);
  ASSERT_TRUE(permuted.ok()) << permuted.status();
  ASSERT_EQ(permuted->segments[0].candidates.size(), 5);
  EXPECT_EQ(permuted->segments[0].candidates[0].id, 0);
  EXPECT_EQ(permuted->segments[0].candidates[1].id, 3);
  EXPECT_EQ(permuted->segments[0].candidates[2].id, 2);
  EXPECT_EQ(permuted->segments[0].candidates[3].id, 1);
  EXPECT_EQ(permuted->segments[0].candidates[4].id, 4);

  const CandidateRankerCapacityResult capacity =
      MakeCapacityResult(*permuted, 2);
  EXPECT_TRUE(
      ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
          accepted, *original, *permuted, capacity)
          .ok());

  converter::CandidateRankerRequest invalid = *permuted;
  std::swap(invalid.segments[0].candidates[3],
            invalid.segments[0].candidates[4]);
  EXPECT_EQ(ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
                accepted, *original, invalid, capacity)
                .code(),
            absl::StatusCode::kInvalidArgument);

  CandidateRankerCapacityResult wrong_capacity = capacity;
  wrong_capacity.segments[0].selected_candidate_count = 3;
  wrong_capacity.segments[0].omitted_by_capacity = 0;
  wrong_capacity.segments[0].first_limiter.reset();
  EXPECT_EQ(ValidateQualityRegressionDevelopmentPermutedNativeCoverageCase(
                accepted, *original, *permuted, wrong_capacity)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace mozc::engine::evaluation
