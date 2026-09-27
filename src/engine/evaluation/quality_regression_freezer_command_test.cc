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
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kDevelopmentInputSha256[] =
    "7f06025eacc5d43d7a56aa8b92f79a982ed54699960a73b764fb322c0d84272f";
constexpr char kHoldoutInputSha256[] =
    "97e5b7eb6a6c7f831f5306725a552eee6d398def178fb5c38601e962b5084087";

struct Partition {
  absl::string_view name;
  absl::string_view config_filename;
  absl::string_view input_filename;
  QualityRegressionCorpusRole role;
  absl::string_view input_sha256;
  int expected_case_count;
};

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
    absl::string_view config_path, absl::string_view input_path,
    absl::string_view data_path, absl::string_view binary_path,
    absl::string_view textproto_path) {
  return {
      absl::StrCat("--config=", config_path),
      absl::StrCat("--input_corpus=", input_path),
      absl::StrCat("--data_file=", data_path),
      absl::StrCat("--output_binary=", binary_path),
      absl::StrCat("--output_textproto=", textproto_path),
  };
}

void AssertOutputValid(absl::string_view binary,
                       absl::string_view review_textproto,
                       const QualityRegressionFreezerConfig& config,
                       absl::string_view config_sha256) {
  QualityRegressionFrozenCorpus from_binary;
  ASSERT_TRUE(from_binary.ParseFromString(binary));
  ASSERT_TRUE(ValidateQualityRegressionFrozenCorpus(
                  from_binary, config, config_sha256)
                  .ok());

  QualityRegressionFrozenCorpus from_text;
  ASSERT_TRUE(ParseTextproto(review_textproto, &from_text).ok());
  ASSERT_TRUE(ValidateQualityRegressionFrozenCorpus(from_text, config,
                                                    config_sha256)
                  .ok());

  absl::StatusOr<std::string> text_binary =
      SerializeDeterministically(from_text);
  ASSERT_TRUE(text_binary.ok()) << text_binary.status();
  EXPECT_EQ(*text_binary, binary);
  absl::StatusOr<std::string> canonical_review = ReviewTextproto(from_binary);
  ASSERT_TRUE(canonical_review.ok()) << canonical_review.status();
  EXPECT_EQ(*canonical_review, review_textproto);
}

TEST(QualityRegressionFreezerCommandTest,
     FreshProcessesFreezeBothInputOnlyPartitionsDeterministically) {
  const std::string executable = testing::GetSourceFileOrDie(
      {"engine", "evaluation", "freeze_quality_regression.exe"});
  const std::string data_path = testing::GetSourceFileOrDie(
      {"data_manager", "oss", "mozc.data"});
  TempDirectory temp_dir = testing::MakeTempDirectoryOrDie();

  constexpr Partition kPartitions[] = {
      {
          .name = "development",
          .config_filename =
              "quality_regression_development_freezer_config.textproto",
          .input_filename =
              "quality_regression_development_input_corpus.textproto",
          .role = DEVELOPMENT,
          .input_sha256 = kDevelopmentInputSha256,
          .expected_case_count = 219,
      },
      {
          .name = "holdout",
          .config_filename =
              "quality_regression_holdout_freezer_config.textproto",
          .input_filename =
              "quality_regression_holdout_input_corpus.textproto",
          .role = HOLDOUT,
          .input_sha256 = kHoldoutInputSha256,
          .expected_case_count = 72,
      },
  };

  for (const Partition& partition : kPartitions) {
    SCOPED_TRACE(partition.name);
    const std::string config_path = testing::GetSourceFileOrDie(
        {"engine", "evaluation", partition.config_filename});
    const std::string input_text_path = testing::GetSourceFileOrDie(
        {"engine", "evaluation", "testdata", partition.input_filename});

    absl::StatusOr<std::string> raw_config =
        FileUtil::GetContents(config_path);
    ASSERT_TRUE(raw_config.ok()) << raw_config.status();
    QualityRegressionFreezerConfig config;
    ASSERT_TRUE(ParseTextproto(*raw_config, &config).ok());
    ASSERT_TRUE(ValidateQualityRegressionFreezerConfig(config).ok());
    ASSERT_EQ(config.expected_corpus_identity().role(), partition.role);
    ASSERT_EQ(config.expected_case_count(), partition.expected_case_count);

    absl::StatusOr<std::string> input_text =
        FileUtil::GetContents(input_text_path);
    ASSERT_TRUE(input_text.ok()) << input_text.status();
    QualityRegressionInputCorpus input;
    ASSERT_TRUE(ParseTextproto(*input_text, &input).ok());
    ASSERT_TRUE(ValidateQualityRegressionInputCorpus(input).ok());
    absl::StatusOr<std::string> input_binary =
        SerializeDeterministically(input);
    ASSERT_TRUE(input_binary.ok()) << input_binary.status();
    ASSERT_EQ(Sha256Bytes(*input_binary), partition.input_sha256);
    ASSERT_EQ(config.input_corpus_sha256(), partition.input_sha256);
    ASSERT_TRUE(ValidateQualityRegressionFreezerInput(
                    input, partition.input_sha256, config)
                    .ok());

    const std::string input_path = FileUtil::JoinPath(
        {temp_dir.path(), absl::StrCat(partition.name, "_input.pb")});
    ASSERT_TRUE(FileUtil::SetContents(input_path, *input_binary).ok());
    const std::string first_binary_path = FileUtil::JoinPath(
        {temp_dir.path(), absl::StrCat(partition.name, "_first.pb")});
    const std::string first_text_path = FileUtil::JoinPath(
        {temp_dir.path(), absl::StrCat(partition.name, "_first.textproto")});
    const std::string second_binary_path = FileUtil::JoinPath(
        {temp_dir.path(), absl::StrCat(partition.name, "_second.pb")});
    const std::string second_text_path = FileUtil::JoinPath(
        {temp_dir.path(), absl::StrCat(partition.name, "_second.textproto")});

    absl::StatusOr<uint32_t> first_exit = RunFreshProcess(
        executable, MakeArguments(config_path, input_path, data_path,
                                  first_binary_path, first_text_path));
    ASSERT_TRUE(first_exit.ok()) << first_exit.status();
    ASSERT_EQ(*first_exit, 0);
    absl::StatusOr<uint32_t> second_exit = RunFreshProcess(
        executable, MakeArguments(config_path, input_path, data_path,
                                  second_binary_path, second_text_path));
    ASSERT_TRUE(second_exit.ok()) << second_exit.status();
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

    const std::string config_sha256 = Sha256Bytes(*raw_config);
    AssertOutputValid(*first_binary, *first_text, config, config_sha256);
    AssertOutputValid(*second_binary, *second_text, config, config_sha256);

    if (partition.role == DEVELOPMENT) {
      const std::string rejected_binary_path = FileUtil::JoinPath(
          {temp_dir.path(), "rejected_unknown_flag.pb"});
      const std::string rejected_text_path = FileUtil::JoinPath(
          {temp_dir.path(), "rejected_unknown_flag.textproto"});
      std::vector<std::string> rejected_arguments = MakeArguments(
          config_path, input_path, data_path, rejected_binary_path,
          rejected_text_path);
      rejected_arguments.push_back("--answer_corpus=forbidden");
      absl::StatusOr<uint32_t> rejected_exit =
          RunFreshProcess(executable, rejected_arguments);
      ASSERT_TRUE(rejected_exit.ok()) << rejected_exit.status();
      EXPECT_NE(*rejected_exit, 0);
      EXPECT_FALSE(FileUtil::FileExists(rejected_binary_path).ok());
      EXPECT_FALSE(FileUtil::FileExists(rejected_text_path).ok());
    }
  }
}

}  // namespace
}  // namespace mozc::engine::evaluation
