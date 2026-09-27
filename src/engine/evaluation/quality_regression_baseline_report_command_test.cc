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
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/quality_regression_baseline_report.h"
#include "engine/evaluation/quality_regression_baseline_report.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kInputCorpusSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kFreezerConfigSha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kAlternateFreezerConfigSha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kSourceRevision[] =
    "dddddddddddddddddddddddddddddddddddddddd";
constexpr char kSourceSha256[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr char kRequestSha256[] =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
constexpr char kConfigSha256[] =
    "1111111111111111111111111111111111111111111111111111111111111111";

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
    absl::string_view answer_path, absl::string_view binary_path,
    absl::string_view textproto_path) {
  return {
      absl::StrCat("--config=", config_path),
      absl::StrCat("--frozen_corpus=", frozen_path),
      absl::StrCat("--answer_corpus=", answer_path),
      absl::StrCat("--output_binary=", binary_path),
      absl::StrCat("--output_textproto=", textproto_path),
  };
}

void FillCorpusIdentity(QualityRegressionCorpusIdentity* identity) {
  identity->set_role(DEVELOPMENT);
  EvaluationSourceIdentity* source = identity->mutable_source();
  source->set_benchmark_name("Synthetic development command test");
  source->set_source_revision(kSourceRevision);
  source->set_source_relative_path("synthetic/development.tsv");
  source->set_source_sha256(kSourceSha256);
  source->set_creator("Synthetic creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("BSD-3-Clause");
  source->set_license_url("https://example.test/license");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic command-test corpus.");
  identity->set_parser_definition_version(1);
  identity->set_normalization_definition_version(1);
  identity->set_import_config_sha256(kSourceSha256);
}

void FillMozcIdentity(EvaluationMozcIdentity* identity) {
  identity->set_source_revision(kSourceRevision);
  identity->set_data_type("synthetic");
  identity->set_data_sha256(kSourceSha256);
  identity->set_default_desktop_request_sha256(kRequestSha256);
  identity->set_default_desktop_config_sha256(kConfigSha256);
  identity->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");
}

QualityRegressionFrozenCorpus MakeFrozenCorpus() {
  QualityRegressionFrozenCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  corpus.set_input_corpus_sha256(kInputCorpusSha256);
  FillMozcIdentity(corpus.mutable_mozc());
  corpus.set_freezer_config_sha256(kFreezerConfigSha256);

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

QualityRegressionAnswerCorpus MakeAnswerCorpus() {
  QualityRegressionAnswerCorpus corpus;
  corpus.set_schema_version(1);
  FillCorpusIdentity(corpus.mutable_identity());
  QualityRegressionAnswerCase* answer_case = corpus.add_cases();
  answer_case->set_source_line(3);
  answer_case->set_normalized_whole_output("synthetic-output");
  return corpus;
}

QualityRegressionDevelopmentBaselineReportConfig MakeConfig(
    absl::string_view frozen_sha256, absl::string_view answer_sha256) {
  QualityRegressionDevelopmentBaselineReportConfig config;
  config.set_schema_version(1);
  config.set_frozen_corpus_schema_version(1);
  config.set_answer_corpus_schema_version(1);
  config.set_report_schema_version(1);
  FillCorpusIdentity(config.mutable_expected_corpus_identity());
  config.set_frozen_corpus_sha256(frozen_sha256);
  config.set_answer_corpus_sha256(answer_sha256);
  config.set_expected_case_count(1);
  config.set_expected_correct_count(1);
  return config;
}

void AssertOutputValid(
    absl::string_view binary, absl::string_view review_textproto,
    const QualityRegressionFrozenCorpus& frozen,
    absl::string_view frozen_sha256,
    const QualityRegressionAnswerCorpus& answers,
    absl::string_view answer_sha256,
    const QualityRegressionDevelopmentBaselineReportConfig& config,
    absl::string_view config_sha256) {
  QualityRegressionDevelopmentBaselineReport from_binary;
  ASSERT_TRUE(from_binary.ParseFromString(binary));
  ASSERT_TRUE(ValidateQualityRegressionDevelopmentBaselineReport(
                  from_binary, frozen, frozen_sha256, answers, answer_sha256,
                  config, config_sha256)
                  .ok());

  QualityRegressionDevelopmentBaselineReport from_text;
  ASSERT_TRUE(ParseTextproto(review_textproto, &from_text).ok());
  ASSERT_TRUE(ValidateQualityRegressionDevelopmentBaselineReport(
                  from_text, frozen, frozen_sha256, answers, answer_sha256,
                  config, config_sha256)
                  .ok());

  absl::StatusOr<std::string> text_binary =
      SerializeDeterministically(from_text);
  ASSERT_TRUE(text_binary.ok()) << text_binary.status();
  EXPECT_EQ(*text_binary, binary);
  absl::StatusOr<std::string> canonical_review = ReviewTextproto(from_binary);
  ASSERT_TRUE(canonical_review.ok()) << canonical_review.status();
  EXPECT_EQ(*canonical_review, review_textproto);
}

TEST(QualityRegressionBaselineReportCommandTest,
     FreshProcessesAreDeterministicAndFailuresWriteNothing) {
  const std::string executable = testing::GetSourceFileOrDie(
      {"engine", "evaluation",
       "report_quality_regression_development_baseline.exe"});
  TempDirectory temp_dir = testing::MakeTempDirectoryOrDie();

  const QualityRegressionFrozenCorpus frozen = MakeFrozenCorpus();
  const QualityRegressionAnswerCorpus answers = MakeAnswerCorpus();
  absl::StatusOr<std::string> frozen_binary =
      SerializeDeterministically(frozen);
  absl::StatusOr<std::string> answer_binary =
      SerializeDeterministically(answers);
  ASSERT_TRUE(frozen_binary.ok()) << frozen_binary.status();
  ASSERT_TRUE(answer_binary.ok()) << answer_binary.status();
  const std::string frozen_sha256 = Sha256Bytes(*frozen_binary);
  const std::string answer_sha256 = Sha256Bytes(*answer_binary);
  const QualityRegressionDevelopmentBaselineReportConfig config =
      MakeConfig(frozen_sha256, answer_sha256);
  absl::StatusOr<std::string> config_text = ReviewTextproto(config);
  ASSERT_TRUE(config_text.ok()) << config_text.status();
  const std::string config_sha256 = Sha256Bytes(*config_text);

  const std::string config_path =
      FileUtil::JoinPath({temp_dir.path(), "config.textproto"});
  const std::string frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "frozen.pb"});
  const std::string answer_path =
      FileUtil::JoinPath({temp_dir.path(), "answers.pb"});
  ASSERT_TRUE(FileUtil::SetContents(config_path, *config_text).ok());
  ASSERT_TRUE(FileUtil::SetContents(frozen_path, *frozen_binary).ok());
  ASSERT_TRUE(FileUtil::SetContents(answer_path, *answer_binary).ok());

  const std::string first_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "first.pb"});
  const std::string first_text_path =
      FileUtil::JoinPath({temp_dir.path(), "first.textproto"});
  const std::string second_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "second.pb"});
  const std::string second_text_path =
      FileUtil::JoinPath({temp_dir.path(), "second.textproto"});

  const std::vector<std::string> first_arguments = MakeArguments(
      config_path, frozen_path, answer_path, first_binary_path,
      first_text_path);
  const std::vector<std::string> second_arguments = MakeArguments(
      config_path, frozen_path, answer_path, second_binary_path,
      second_text_path);
  ASSERT_EQ(first_arguments.size(), 5);
  ASSERT_EQ(second_arguments.size(), 5);
  absl::StatusOr<uint32_t> first_exit =
      RunFreshProcess(executable, first_arguments);
  absl::StatusOr<uint32_t> second_exit =
      RunFreshProcess(executable, second_arguments);
  ASSERT_TRUE(first_exit.ok()) << first_exit.status();
  ASSERT_TRUE(second_exit.ok()) << second_exit.status();
  ASSERT_EQ(*first_exit, 0);
  ASSERT_EQ(*second_exit, 0);

  absl::StatusOr<std::string> first_binary =
      FileUtil::GetContents(first_binary_path);
  absl::StatusOr<std::string> first_text =
      FileUtil::GetContents(first_text_path);
  absl::StatusOr<std::string> second_binary =
      FileUtil::GetContents(second_binary_path);
  absl::StatusOr<std::string> second_text =
      FileUtil::GetContents(second_text_path);
  ASSERT_TRUE(first_binary.ok()) << first_binary.status();
  ASSERT_TRUE(first_text.ok()) << first_text.status();
  ASSERT_TRUE(second_binary.ok()) << second_binary.status();
  ASSERT_TRUE(second_text.ok()) << second_text.status();
  EXPECT_EQ(*first_binary, *second_binary);
  EXPECT_EQ(*first_text, *second_text);
  AssertOutputValid(*first_binary, *first_text, frozen, frozen_sha256, answers,
                    answer_sha256, config, config_sha256);
  AssertOutputValid(*second_binary, *second_text, frozen, frozen_sha256,
                    answers, answer_sha256, config, config_sha256);

  QualityRegressionFrozenCorpus mismatched_frozen = frozen;
  mismatched_frozen.set_freezer_config_sha256(
      kAlternateFreezerConfigSha256);
  absl::StatusOr<std::string> mismatched_frozen_binary =
      SerializeDeterministically(mismatched_frozen);
  ASSERT_TRUE(mismatched_frozen_binary.ok())
      << mismatched_frozen_binary.status();
  ASSERT_NE(Sha256Bytes(*mismatched_frozen_binary), frozen_sha256);
  const std::string mismatched_frozen_path =
      FileUtil::JoinPath({temp_dir.path(), "mismatched_frozen.pb"});
  ASSERT_TRUE(FileUtil::SetContents(mismatched_frozen_path,
                                    *mismatched_frozen_binary)
                  .ok());
  const std::string hash_rejected_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "hash_rejected.pb"});
  const std::string hash_rejected_text_path =
      FileUtil::JoinPath({temp_dir.path(), "hash_rejected.textproto"});
  const std::vector<std::string> hash_rejected_arguments = MakeArguments(
      config_path, mismatched_frozen_path, answer_path,
      hash_rejected_binary_path, hash_rejected_text_path);
  ASSERT_EQ(hash_rejected_arguments.size(), 5);
  absl::StatusOr<uint32_t> hash_rejected_exit =
      RunFreshProcess(executable, hash_rejected_arguments);
  ASSERT_TRUE(hash_rejected_exit.ok()) << hash_rejected_exit.status();
  EXPECT_NE(*hash_rejected_exit, 0);
  EXPECT_FALSE(FileUtil::FileExists(hash_rejected_binary_path).ok());
  EXPECT_FALSE(FileUtil::FileExists(hash_rejected_text_path).ok());

  QualityRegressionDevelopmentBaselineReportConfig role_mismatched_config =
      config;
  role_mismatched_config.mutable_expected_corpus_identity()->set_role(HOLDOUT);
  absl::StatusOr<std::string> role_mismatched_config_text =
      ReviewTextproto(role_mismatched_config);
  ASSERT_TRUE(role_mismatched_config_text.ok())
      << role_mismatched_config_text.status();
  const std::string role_mismatched_config_path =
      FileUtil::JoinPath({temp_dir.path(), "role_mismatched.textproto"});
  ASSERT_TRUE(FileUtil::SetContents(role_mismatched_config_path,
                                    *role_mismatched_config_text)
                  .ok());
  const std::string role_rejected_binary_path =
      FileUtil::JoinPath({temp_dir.path(), "role_rejected.pb"});
  const std::string role_rejected_text_path =
      FileUtil::JoinPath({temp_dir.path(), "role_rejected.textproto"});
  const std::vector<std::string> role_rejected_arguments = MakeArguments(
      role_mismatched_config_path, frozen_path, answer_path,
      role_rejected_binary_path, role_rejected_text_path);
  ASSERT_EQ(role_rejected_arguments.size(), 5);
  absl::StatusOr<uint32_t> role_rejected_exit =
      RunFreshProcess(executable, role_rejected_arguments);
  ASSERT_TRUE(role_rejected_exit.ok()) << role_rejected_exit.status();
  EXPECT_NE(*role_rejected_exit, 0);
  EXPECT_FALSE(FileUtil::FileExists(role_rejected_binary_path).ok());
  EXPECT_FALSE(FileUtil::FileExists(role_rejected_text_path).ok());
}

}  // namespace
}  // namespace mozc::engine::evaluation
