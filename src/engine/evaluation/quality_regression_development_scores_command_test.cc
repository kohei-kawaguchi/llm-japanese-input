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

#include <windows.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "base/file/temp_dir.h"
#include "base/file_util.h"
#include "base/win32/wide_char.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kHex64A[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kHex64B[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kHex64C[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kRevision[] = "dddddddddddddddddddddddddddddddddddddddd";

absl::StatusOr<uint32_t> RunFreshProcess(
    absl::string_view executable, const std::vector<std::string>& arguments) {
  std::vector<std::string> quoted_arguments;
  quoted_arguments.reserve(arguments.size());
  for (const std::string& argument : arguments) {
    quoted_arguments.push_back(absl::StrCat("\"", argument, "\""));
  }
  std::wstring command_line = win32::Utf8ToWide(absl::StrCat(
      "\"", executable, "\" ", absl::StrJoin(quoted_arguments, " ")));

  STARTUPINFOW startup_info = {};
  startup_info.cb = sizeof(startup_info);
  PROCESS_INFORMATION process_info = {};
  if (::CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                       CREATE_DEFAULT_ERROR_MODE | CREATE_NO_WINDOW, nullptr,
                       nullptr, &startup_info, &process_info) == FALSE) {
    return absl::InternalError(
        absl::StrCat("CreateProcessW failed: ", ::GetLastError()));
  }
  ::CloseHandle(process_info.hThread);
  const DWORD wait_result =
      ::WaitForSingleObject(process_info.hProcess, INFINITE);
  if (wait_result != WAIT_OBJECT_0) {
    const DWORD error = ::GetLastError();
    ::CloseHandle(process_info.hProcess);
    return absl::InternalError(
        absl::StrCat("WaitForSingleObject failed: ", error));
  }
  DWORD exit_code = 0;
  if (::GetExitCodeProcess(process_info.hProcess, &exit_code) == FALSE) {
    const DWORD error = ::GetLastError();
    ::CloseHandle(process_info.hProcess);
    return absl::InternalError(
        absl::StrCat("GetExitCodeProcess failed: ", error));
  }
  ::CloseHandle(process_info.hProcess);
  return static_cast<uint32_t>(exit_code);
}

std::vector<std::string> MakeArguments(absl::string_view score_config_path,
                                       absl::string_view execution_config_path,
                                       absl::string_view frozen_corpus_path,
                                       absl::string_view native_coverage_path,
                                       absl::string_view manifest_directory,
                                       absl::string_view model_directory,
                                       absl::string_view binary_path,
                                       absl::string_view textproto_path) {
  return {
      absl::StrCat("--score_config=", score_config_path),
      absl::StrCat("--execution_config=", execution_config_path),
      absl::StrCat("--frozen_corpus=", frozen_corpus_path),
      absl::StrCat("--native_coverage=", native_coverage_path),
      absl::StrCat("--manifest_directory=", manifest_directory),
      absl::StrCat("--model_directory=", model_directory),
      absl::StrCat("--output_binary=", binary_path),
      absl::StrCat("--output_textproto=", textproto_path),
  };
}

void ExpectNoOutput(absl::string_view binary_path,
                    absl::string_view textproto_path) {
  EXPECT_FALSE(FileUtil::FileExists(binary_path).ok());
  EXPECT_FALSE(FileUtil::FileExists(textproto_path).ok());
}

void FillCorpusIdentity(QualityRegressionCorpusIdentity* identity) {
  identity->set_role(DEVELOPMENT);
  EvaluationSourceIdentity* source = identity->mutable_source();
  source->set_benchmark_name("Synthetic score command test");
  source->set_source_revision(kRevision);
  source->set_source_relative_path("synthetic/development.tsv");
  source->set_source_sha256(kHex64A);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic command-test corpus.");
  identity->set_parser_definition_version(1);
  identity->set_normalization_definition_version(1);
  identity->set_import_config_sha256(kHex64A);
}

QualityRegressionFrozenCorpus MakeFrozenCorpus() {
  QualityRegressionFrozenCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  corpus.set_input_corpus_sha256(kHex64A);
  EvaluationMozcIdentity* mozc = corpus.mutable_mozc();
  mozc->set_source_revision(kRevision);
  mozc->set_data_type("synthetic");
  mozc->set_data_sha256(kHex64A);
  mozc->set_default_desktop_request_sha256(kHex64B);
  mozc->set_default_desktop_config_sha256(kHex64C);
  mozc->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");
  corpus.set_freezer_config_sha256(kHex64B);

  QualityRegressionFrozenCase* frozen_case = corpus.add_cases();
  frozen_case->set_source_line(3);
  frozen_case->set_conversion_reading("synthetic-reading");
  frozen_case->set_mozc_baseline_output("synthetic-output");
  FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
  request->mutable_token()->set_session_generation(0);
  request->mutable_token()->set_state_revision(0);
  request->mutable_token()->set_request_sequence(1);
  request->set_mode(FrozenCandidateRankerRequest::MODE_CONVERSION);
  request->set_preceding_text("");
  request->set_following_text("");
  request->set_reading("synthetic-reading");
  request->set_focused_segment_id(0);
  FrozenCandidateRankerSegment* segment = request->add_segments();
  segment->set_id(0);
  segment->set_key("synthetic-key");
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(0);
  candidate->set_key("synthetic-key");
  candidate->set_value("synthetic-output");
  candidate->set_cost(123);
  candidate->set_attributes(0);
  candidate->set_consumed_key_size(3);
  candidate->set_is_protected(false);
  return corpus;
}

CandidateRankerModelManifest MakeManifest() {
  CandidateRankerModelManifest manifest;
  manifest.set_schema_version(5);
  CandidateRankerModelManifest::ScoringTemplate* scoring =
      manifest.mutable_scoring_template();
  scoring->set_version("score-command-v1");
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
  scoring->set_context_policy(CandidateRankerModelManifest::ScoringTemplate::
                                  MOZC_BASELINE_SUBSTITUTION);
  scoring->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET);
  CandidateRankerModelManifest::BoundedExecutionPolicy* policy =
      manifest.mutable_bounded_execution_policy();
  policy->set_candidate_window_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          MOZC_ORDER_UNPROTECTED_PREFIX);
  policy->set_capacity_reduction_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT);
  policy->set_decode_packing_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          SEGMENT_ID_ORDER_GREEDY_WHOLE_SEGMENTS);
  CandidateRankerModelManifest::Limits* limits = manifest.mutable_limits();
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
  CandidateRankerModelManifest::Runtime* runtime = manifest.mutable_runtime();
  runtime->set_execution_device(CandidateRankerModelManifest::Runtime::CPU);
  runtime->set_kv_storage(CandidateRankerModelManifest::Runtime::UNIFIED);
  runtime->set_micro_batch_token_capacity(64);
  runtime->set_decode_thread_count(2);
  runtime->set_batch_thread_count(4);
  runtime->set_use_memory_map(true);
  runtime->set_use_memory_lock(false);
  runtime->set_check_tensors(true);
  CandidateRankerModelManifest::Artifacts* artifacts =
      manifest.mutable_artifacts();
  artifacts->set_source_model("synthetic/model");
  artifacts->set_source_revision(kRevision);
  artifacts->set_source_weight_sha256(kHex64A);
  artifacts->set_tokenizer_sha256(kHex64A);
  artifacts->set_gguf_file_name("absent-f16.gguf");
  artifacts->set_gguf_sha256(kHex64A);
  artifacts->set_quantization("F16");
  artifacts->add_model_license_references("MODEL_LICENSE.txt");
  artifacts->set_runtime_name("synthetic-runtime");
  artifacts->set_runtime_revision(kRevision);
  artifacts->set_runtime_source_archive_sha256(kHex64A);
  artifacts->add_runtime_license_references("RUNTIME_LICENSE.txt");
  return manifest;
}

std::array<CandidateRankerModelManifest, 3> MakeManifests() {
  std::array<CandidateRankerModelManifest, 3> manifests = {
      MakeManifest(), MakeManifest(), MakeManifest()};
  manifests[1].mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITHOUT_TARGET);
  manifests[2].mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY);
  return manifests;
}

DevelopmentExecutionConfig MakeExecutionConfig(
    absl::string_view frozen_sha256,
    const std::array<std::string, 3>& manifest_sha256s) {
  DevelopmentExecutionConfig config;
  config.set_schema_version(1);
  config.set_frozen_corpus_schema_version(1);
  config.set_native_coverage_schema_version(1);
  config.set_frozen_corpus_sha256(frozen_sha256);
  config.set_expected_case_count(1);
  config.set_model_manifest_schema_version(5);
  config.set_coverage_definition_version(1);
  config.set_permutation_definition_version(1);
  const std::array<CandidateRankerModelManifest::ScoringTemplate::RecordLayout,
                   3>
      layouts = {
          CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET,
          CandidateRankerModelManifest::ScoringTemplate::
              STRUCTURED_WITHOUT_TARGET,
          CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY,
      };
  const std::array<std::string, 3> filenames = {
      "structured-with-target.textproto", "structured-without-target.textproto",
      "natural-text-only.textproto"};
  for (size_t index = 0; index < layouts.size(); ++index) {
    DevelopmentObjectiveSpec* objective = config.add_objectives();
    objective->set_objective(layouts[index]);
    objective->set_manifest_filename(filenames[index]);
    objective->set_manifest_sha256(manifest_sha256s[index]);
    objective->set_selection_eligible(index != 0);
  }
  return config;
}

DevelopmentNativeCoverageSuite MakeCoverage(
    const DevelopmentExecutionConfig& execution_config,
    absl::string_view execution_config_sha256,
    absl::string_view frozen_corpus_sha256) {
  DevelopmentNativeCoverageSuite coverage;
  coverage.set_schema_version(1);
  coverage.set_execution_config_sha256(execution_config_sha256);
  coverage.set_frozen_corpus_sha256(frozen_corpus_sha256);
  coverage.set_coverage_definition_version(1);
  for (const DevelopmentObjectiveSpec& objective :
       execution_config.objectives()) {
    ObjectiveNativeCoverage* objective_coverage = coverage.add_coverages();
    objective_coverage->set_objective(objective.objective());
    objective_coverage->mutable_map()->add_cases()->set_source_line(3);
  }
  return coverage;
}

DevelopmentScoreRunnerConfig MakeScoreConfig(
    absl::string_view execution_config_sha256,
    absl::string_view frozen_corpus_sha256,
    absl::string_view native_coverage_sha256) {
  DevelopmentScoreRunnerConfig config;
  config.set_schema_version(1);
  config.set_score_artifact_schema_version(2);
  config.set_execution_config_sha256(execution_config_sha256);
  config.set_frozen_corpus_sha256(frozen_corpus_sha256);
  config.set_native_coverage_suite_sha256(native_coverage_sha256);
  config.set_score_definition_version(2);
  config.set_numeric_profile(DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED);
  config.set_layout(DEVELOPMENT_EVALUATION_SHARED_TRIE);
  return config;
}

TEST(QualityRegressionDevelopmentScoresCommandTest,
     PreModelGatesWriteNothingWithoutGguf) {
  const std::string executable = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "run_quality_regression_development_scores.exe"});
  TempDirectory temp_dir = testing::MakeTempDirectoryOrDie();

  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  absl::StatusOr<std::string> frozen_binary =
      SerializeDeterministically(frozen);
  ASSERT_TRUE(frozen_binary.ok()) << frozen_binary.status();
  const std::string frozen_sha256 = Sha256Bytes(*frozen_binary);

  const std::array<CandidateRankerModelManifest, 3> manifests = MakeManifests();
  std::array<std::string, 3> manifest_texts;
  std::array<std::string, 3> manifest_sha256s;
  for (size_t index = 0; index < manifests.size(); ++index) {
    absl::StatusOr<std::string> text = ReviewTextproto(manifests[index]);
    ASSERT_TRUE(text.ok()) << text.status();
    manifest_texts[index] = std::move(*text);
    manifest_sha256s[index] = Sha256Bytes(manifest_texts[index]);
  }

  const DevelopmentExecutionConfig execution_config =
      MakeExecutionConfig(frozen_sha256, manifest_sha256s);
  absl::StatusOr<std::string> execution_config_text =
      ReviewTextproto(execution_config);
  ASSERT_TRUE(execution_config_text.ok()) << execution_config_text.status();
  const std::string execution_config_sha256 =
      Sha256Bytes(*execution_config_text);

  const DevelopmentNativeCoverageSuite coverage =
      MakeCoverage(execution_config, execution_config_sha256, frozen_sha256);
  absl::StatusOr<std::string> coverage_binary =
      SerializeDeterministically(coverage);
  ASSERT_TRUE(coverage_binary.ok()) << coverage_binary.status();
  const std::string coverage_sha256 = Sha256Bytes(*coverage_binary);

  const DevelopmentScoreRunnerConfig score_config =
      MakeScoreConfig(execution_config_sha256, frozen_sha256, coverage_sha256);
  absl::StatusOr<std::string> score_config_text = ReviewTextproto(score_config);
  ASSERT_TRUE(score_config_text.ok()) << score_config_text.status();

  const std::string score_config_path =
      FileUtil::JoinPath({temp_dir.path(), "score-config.textproto"});
  const std::string execution_config_path =
      FileUtil::JoinPath({temp_dir.path(), "execution-config.textproto"});
  const std::string frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "frozen.pb"});
  const std::string coverage_path =
      FileUtil::JoinPath({temp_dir.path(), "coverage.pb"});
  ASSERT_TRUE(
      FileUtil::SetContents(score_config_path, *score_config_text).ok());
  ASSERT_TRUE(
      FileUtil::SetContents(execution_config_path, *execution_config_text)
          .ok());
  ASSERT_TRUE(FileUtil::SetContents(frozen_path, *frozen_binary).ok());
  ASSERT_TRUE(FileUtil::SetContents(coverage_path, *coverage_binary).ok());
  for (int index = 0; index < execution_config.objectives_size(); ++index) {
    ASSERT_TRUE(
        FileUtil::SetContents(
            FileUtil::JoinPath(
                {temp_dir.path(),
                 execution_config.objectives(index).manifest_filename()}),
            manifest_texts[index])
            .ok());
  }

  const std::string no_model_binary =
      FileUtil::JoinPath({temp_dir.path(), "no-model.pb"});
  const std::string no_model_text =
      FileUtil::JoinPath({temp_dir.path(), "no-model.textproto"});
  const std::vector<std::string> valid_arguments = MakeArguments(
      score_config_path, execution_config_path, frozen_path, coverage_path,
      temp_dir.path(), temp_dir.path(), no_model_binary, no_model_text);
  ASSERT_EQ(valid_arguments.size(), 8);
  absl::StatusOr<uint32_t> no_model_exit =
      RunFreshProcess(executable, valid_arguments);
  ASSERT_TRUE(no_model_exit.ok()) << no_model_exit.status();
  EXPECT_EQ(*no_model_exit, static_cast<uint32_t>(absl::StatusCode::kNotFound));
  ExpectNoOutput(no_model_binary, no_model_text);

  QualityRegressionFrozenCorpus changed_frozen = frozen;
  changed_frozen.set_freezer_config_sha256(kHex64C);
  absl::StatusOr<std::string> changed_frozen_binary =
      SerializeDeterministically(changed_frozen);
  ASSERT_TRUE(changed_frozen_binary.ok()) << changed_frozen_binary.status();
  const std::string changed_frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "changed-frozen.pb"});
  ASSERT_TRUE(
      FileUtil::SetContents(changed_frozen_path, *changed_frozen_binary).ok());
  const std::string hash_binary =
      FileUtil::JoinPath({temp_dir.path(), "hash-rejected.pb"});
  const std::string hash_text =
      FileUtil::JoinPath({temp_dir.path(), "hash-rejected.textproto"});
  absl::StatusOr<uint32_t> hash_exit = RunFreshProcess(
      executable,
      MakeArguments(score_config_path, execution_config_path,
                    changed_frozen_path, coverage_path, temp_dir.path(),
                    temp_dir.path(), hash_binary, hash_text));
  ASSERT_TRUE(hash_exit.ok()) << hash_exit.status();
  EXPECT_EQ(*hash_exit,
            static_cast<uint32_t>(absl::StatusCode::kFailedPrecondition));
  ExpectNoOutput(hash_binary, hash_text);

  ASSERT_TRUE(FileUtil::SetContents(
                  FileUtil::JoinPath(
                      {temp_dir.path(),
                       execution_config.objectives(0).manifest_filename()}),
                  "invalid manifest")
                  .ok());
  const std::string third_manifest_path = FileUtil::JoinPath(
      {temp_dir.path(), execution_config.objectives(2).manifest_filename()});
  ASSERT_TRUE(FileUtil::Unlink(third_manifest_path).ok());
  const std::string raw_order_binary =
      FileUtil::JoinPath({temp_dir.path(), "raw-order-rejected.pb"});
  const std::string raw_order_text =
      FileUtil::JoinPath({temp_dir.path(), "raw-order-rejected.textproto"});
  absl::StatusOr<uint32_t> raw_order_exit = RunFreshProcess(
      executable,
      MakeArguments(score_config_path, execution_config_path, frozen_path,
                    coverage_path, temp_dir.path(), temp_dir.path(),
                    raw_order_binary, raw_order_text));
  ASSERT_TRUE(raw_order_exit.ok()) << raw_order_exit.status();
  EXPECT_EQ(*raw_order_exit,
            static_cast<uint32_t>(absl::StatusCode::kNotFound));
  ExpectNoOutput(raw_order_binary, raw_order_text);
  ASSERT_TRUE(
      FileUtil::SetContents(third_manifest_path, manifest_texts[2]).ok());

  const std::string parse_binary =
      FileUtil::JoinPath({temp_dir.path(), "parse-rejected.pb"});
  const std::string parse_text =
      FileUtil::JoinPath({temp_dir.path(), "parse-rejected.textproto"});
  absl::StatusOr<uint32_t> parse_exit = RunFreshProcess(
      executable, MakeArguments(score_config_path, execution_config_path,
                                frozen_path, coverage_path, temp_dir.path(),
                                temp_dir.path(), parse_binary, parse_text));
  ASSERT_TRUE(parse_exit.ok()) << parse_exit.status();
  EXPECT_EQ(*parse_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(parse_binary, parse_text);

  CandidateRankerModelManifest changed_manifest = manifests[0];
  changed_manifest.mutable_scoring_template()->set_version(
      "changed-score-command-v1");
  absl::StatusOr<std::string> changed_manifest_text =
      ReviewTextproto(changed_manifest);
  ASSERT_TRUE(changed_manifest_text.ok()) << changed_manifest_text.status();
  ASSERT_TRUE(FileUtil::SetContents(
                  FileUtil::JoinPath(
                      {temp_dir.path(),
                       execution_config.objectives(0).manifest_filename()}),
                  *changed_manifest_text)
                  .ok());
  const std::string manifest_hash_binary =
      FileUtil::JoinPath({temp_dir.path(), "manifest-hash-rejected.pb"});
  const std::string manifest_hash_text =
      FileUtil::JoinPath({temp_dir.path(), "manifest-hash-rejected.textproto"});
  absl::StatusOr<uint32_t> manifest_hash_exit = RunFreshProcess(
      executable,
      MakeArguments(score_config_path, execution_config_path, frozen_path,
                    coverage_path, temp_dir.path(), temp_dir.path(),
                    manifest_hash_binary, manifest_hash_text));
  ASSERT_TRUE(manifest_hash_exit.ok()) << manifest_hash_exit.status();
  EXPECT_EQ(*manifest_hash_exit,
            static_cast<uint32_t>(absl::StatusCode::kFailedPrecondition));
  ExpectNoOutput(manifest_hash_binary, manifest_hash_text);

  const std::string argv_binary =
      FileUtil::JoinPath({temp_dir.path(), "argv-rejected.pb"});
  const std::string argv_text =
      FileUtil::JoinPath({temp_dir.path(), "argv-rejected.textproto"});
  std::vector<std::string> forbidden_arguments = MakeArguments(
      score_config_path, execution_config_path, frozen_path, coverage_path,
      temp_dir.path(), temp_dir.path(), argv_binary, argv_text);
  forbidden_arguments.push_back("--answer_corpus=forbidden");
  absl::StatusOr<uint32_t> forbidden_exit =
      RunFreshProcess(executable, forbidden_arguments);
  ASSERT_TRUE(forbidden_exit.ok()) << forbidden_exit.status();
  EXPECT_EQ(*forbidden_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(argv_binary, argv_text);

  std::vector<std::string> duplicate_arguments = MakeArguments(
      score_config_path, execution_config_path, frozen_path, coverage_path,
      temp_dir.path(), temp_dir.path(), argv_binary, argv_text);
  duplicate_arguments.back() = duplicate_arguments.front();
  absl::StatusOr<uint32_t> duplicate_exit =
      RunFreshProcess(executable, duplicate_arguments);
  ASSERT_TRUE(duplicate_exit.ok()) << duplicate_exit.status();
  EXPECT_EQ(*duplicate_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(argv_binary, argv_text);

  std::vector<std::string> empty_arguments = MakeArguments(
      score_config_path, execution_config_path, frozen_path, coverage_path,
      temp_dir.path(), temp_dir.path(), argv_binary, argv_text);
  empty_arguments.back() = "--output_textproto=";
  absl::StatusOr<uint32_t> empty_exit =
      RunFreshProcess(executable, empty_arguments);
  ASSERT_TRUE(empty_exit.ok()) << empty_exit.status();
  EXPECT_EQ(*empty_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(argv_binary, argv_text);

  std::vector<std::string> split_arguments = MakeArguments(
      score_config_path, execution_config_path, frozen_path, coverage_path,
      temp_dir.path(), temp_dir.path(), argv_binary, argv_text);
  split_arguments.front() = "--score_config";
  absl::StatusOr<uint32_t> split_exit =
      RunFreshProcess(executable, split_arguments);
  ASSERT_TRUE(split_exit.ok()) << split_exit.status();
  EXPECT_EQ(*split_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(argv_binary, argv_text);
}

}  // namespace
}  // namespace mozc::engine::evaluation
