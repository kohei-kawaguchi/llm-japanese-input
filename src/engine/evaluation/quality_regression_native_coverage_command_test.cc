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
constexpr char kRevision[] =
    "dddddddddddddddddddddddddddddddddddddddd";

absl::StatusOr<uint32_t> RunFreshProcess(
    absl::string_view executable,
    const std::vector<std::string>& arguments) {
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

std::vector<std::string> MakeArguments(
    absl::string_view config_path, absl::string_view frozen_path,
    absl::string_view manifest_directory, absl::string_view model_directory,
    absl::string_view binary_path, absl::string_view textproto_path) {
  return {
      absl::StrCat("--config=", config_path),
      absl::StrCat("--frozen_corpus=", frozen_path),
      absl::StrCat("--manifest_directory=", manifest_directory),
      absl::StrCat("--model_directory=", model_directory),
      absl::StrCat("--output_binary=", binary_path),
      absl::StrCat("--output_textproto=", textproto_path),
  };
}

void FillCorpusIdentity(QualityRegressionCorpusIdentity* identity) {
  identity->set_role(DEVELOPMENT);
  EvaluationSourceIdentity* source = identity->mutable_source();
  source->set_benchmark_name("Synthetic native-coverage command test");
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

DevelopmentExecutionConfig MakeConfig(
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

  DevelopmentObjectiveSpec* objective = config.add_objectives();
  objective->set_objective(
      CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET);
  objective->set_manifest_filename("structured-with-target.textproto");
  objective->set_manifest_sha256(manifest_sha256s[0]);
  objective->set_selection_eligible(false);
  objective = config.add_objectives();
  objective->set_objective(CandidateRankerModelManifest::ScoringTemplate::
                               STRUCTURED_WITHOUT_TARGET);
  objective->set_manifest_filename("structured-without-target.textproto");
  objective->set_manifest_sha256(manifest_sha256s[1]);
  objective->set_selection_eligible(true);
  objective = config.add_objectives();
  objective->set_objective(
      CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY);
  objective->set_manifest_filename("natural-text-only.textproto");
  objective->set_manifest_sha256(manifest_sha256s[2]);
  objective->set_selection_eligible(true);
  return config;
}

void ExpectNoOutput(absl::string_view binary_path,
                    absl::string_view textproto_path) {
  EXPECT_FALSE(FileUtil::FileExists(binary_path).ok());
  EXPECT_FALSE(FileUtil::FileExists(textproto_path).ok());
}

TEST(QualityRegressionNativeCoverageCommandTest,
     PreModelFailuresWriteNothingAndAnswerInputIsRejected) {
  const std::string executable = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "audit_quality_regression_development_native_coverage.exe"});
  TempDirectory temp_dir = testing::MakeTempDirectoryOrDie();

  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  absl::StatusOr<std::string> frozen_binary =
      SerializeDeterministically(frozen);
  ASSERT_TRUE(frozen_binary.ok()) << frozen_binary.status();
  const std::string frozen_sha256 = Sha256Bytes(*frozen_binary);
  const std::array<std::string, 3> invalid_manifests = {
      "invalid manifest one", "invalid manifest two",
      "invalid manifest three"};
  const std::array<std::string, 3> manifest_sha256s = {
      Sha256Bytes(invalid_manifests[0]), Sha256Bytes(invalid_manifests[1]),
      Sha256Bytes(invalid_manifests[2])};
  const DevelopmentExecutionConfig config =
      MakeConfig(frozen_sha256, manifest_sha256s);
  absl::StatusOr<std::string> config_text = ReviewTextproto(config);
  ASSERT_TRUE(config_text.ok()) << config_text.status();

  const std::string config_path =
      FileUtil::JoinPath({temp_dir.path(), "config.textproto"});
  const std::string frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "frozen.pb"});
  ASSERT_TRUE(FileUtil::SetContents(config_path, *config_text).ok());
  ASSERT_TRUE(FileUtil::SetContents(frozen_path, *frozen_binary).ok());
  ASSERT_TRUE(FileUtil::SetContents(
                  FileUtil::JoinPath({temp_dir.path(),
                                      "structured-with-target.textproto"}),
                  invalid_manifests[0])
                  .ok());
  ASSERT_TRUE(FileUtil::SetContents(
                  FileUtil::JoinPath(
                      {temp_dir.path(),
                       "structured-without-target.textproto"}),
                  invalid_manifests[1])
                  .ok());
  ASSERT_TRUE(FileUtil::SetContents(
                  FileUtil::JoinPath(
                      {temp_dir.path(), "natural-text-only.textproto"}),
                  invalid_manifests[2])
                  .ok());

  const std::string parse_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "parse_rejected.pb"});
  const std::string parse_text_path =
      FileUtil::JoinPath({temp_dir.path(), "parse_rejected.textproto"});
  const std::vector<std::string> parse_arguments = MakeArguments(
      config_path, frozen_path, temp_dir.path(), temp_dir.path(),
      parse_binary_path, parse_text_path);
  ASSERT_EQ(parse_arguments.size(), 6);
  absl::StatusOr<uint32_t> parse_exit =
      RunFreshProcess(executable, parse_arguments);
  ASSERT_TRUE(parse_exit.ok()) << parse_exit.status();
  EXPECT_EQ(*parse_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(parse_binary_path, parse_text_path);

  QualityRegressionFrozenCorpus changed_frozen = frozen;
  changed_frozen.set_freezer_config_sha256(kHex64C);
  absl::StatusOr<std::string> changed_frozen_binary =
      SerializeDeterministically(changed_frozen);
  ASSERT_TRUE(changed_frozen_binary.ok()) << changed_frozen_binary.status();
  const std::string changed_frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "changed_frozen.pb"});
  ASSERT_TRUE(
      FileUtil::SetContents(changed_frozen_path, *changed_frozen_binary).ok());
  const std::string hash_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "hash_rejected.pb"});
  const std::string hash_text_path =
      FileUtil::JoinPath({temp_dir.path(), "hash_rejected.textproto"});
  absl::StatusOr<uint32_t> hash_exit = RunFreshProcess(
      executable,
      MakeArguments(config_path, changed_frozen_path, temp_dir.path(),
                    temp_dir.path(), hash_binary_path, hash_text_path));
  ASSERT_TRUE(hash_exit.ok()) << hash_exit.status();
  EXPECT_EQ(*hash_exit,
            static_cast<uint32_t>(absl::StatusCode::kFailedPrecondition));
  ExpectNoOutput(hash_binary_path, hash_text_path);

  const std::string answer_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "answer_rejected.pb"});
  const std::string answer_text_path =
      FileUtil::JoinPath({temp_dir.path(), "answer_rejected.textproto"});
  std::vector<std::string> answer_arguments = MakeArguments(
      FileUtil::JoinPath({temp_dir.path(), "missing-config.textproto"}),
      FileUtil::JoinPath({temp_dir.path(), "missing-frozen.pb"}),
      FileUtil::JoinPath({temp_dir.path(), "missing-manifests"}),
      FileUtil::JoinPath({temp_dir.path(), "missing-model"}),
      answer_binary_path, answer_text_path);
  answer_arguments.push_back("--answer_corpus=forbidden");
  absl::StatusOr<uint32_t> answer_exit =
      RunFreshProcess(executable, answer_arguments);
  ASSERT_TRUE(answer_exit.ok()) << answer_exit.status();
  EXPECT_EQ(*answer_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(answer_binary_path, answer_text_path);

  const std::string unexpected_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "unexpected_rejected.pb"});
  const std::string unexpected_text_path = FileUtil::JoinPath(
      {temp_dir.path(), "unexpected_rejected.textproto"});
  std::vector<std::string> unexpected_arguments = MakeArguments(
      config_path, frozen_path, temp_dir.path(), temp_dir.path(),
      unexpected_binary_path, unexpected_text_path);
  unexpected_arguments.push_back("--unexpected=forbidden");
  absl::StatusOr<uint32_t> unexpected_exit =
      RunFreshProcess(executable, unexpected_arguments);
  ASSERT_TRUE(unexpected_exit.ok()) << unexpected_exit.status();
  EXPECT_EQ(*unexpected_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(unexpected_binary_path, unexpected_text_path);

  const std::string duplicate_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "duplicate_rejected.pb"});
  const std::string duplicate_text_path =
      FileUtil::JoinPath({temp_dir.path(), "duplicate_rejected.textproto"});
  std::vector<std::string> duplicate_arguments = MakeArguments(
      config_path, frozen_path, temp_dir.path(), temp_dir.path(),
      duplicate_binary_path, duplicate_text_path);
  duplicate_arguments.back() = duplicate_arguments.front();
  absl::StatusOr<uint32_t> duplicate_exit =
      RunFreshProcess(executable, duplicate_arguments);
  ASSERT_TRUE(duplicate_exit.ok()) << duplicate_exit.status();
  EXPECT_EQ(*duplicate_exit,
            static_cast<uint32_t>(absl::StatusCode::kInvalidArgument));
  ExpectNoOutput(duplicate_binary_path, duplicate_text_path);
}

}  // namespace
}  // namespace mozc::engine::evaluation
