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

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/evaluation_freezer.h"
#include "engine/evaluation/evaluation_freezer_runtime.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_freezer.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "",
          "Checked quality-regression freezer textproto");
ABSL_FLAG(std::string, input_corpus, "",
          "Deterministic quality-regression input corpus");
ABSL_FLAG(std::string, data_file, "", "Pinned OSS mozc.data path");
ABSL_FLAG(std::string, output_binary, "", "Frozen corpus binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Frozen corpus review textproto output");

namespace mozc::engine::evaluation {
namespace {

absl::Status RejectAnswerCorpusFlag(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const absl::string_view argument(argv[i]);
    if (argument == "--answer_corpus" ||
        absl::StartsWith(argument, "--answer_corpus=") ||
        argument == "-answer_corpus" ||
        absl::StartsWith(argument, "-answer_corpus=")) {
      return absl::InvalidArgumentError(
          "freeze_quality_regression accepts no answer corpus");
    }
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string input_corpus_path = absl::GetFlag(FLAGS_input_corpus);
  const std::string data_file_path = absl::GetFlag(FLAGS_data_file);
  const std::string output_binary_path = absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);
  if (config_path.empty() || input_corpus_path.empty() ||
      data_file_path.empty() || output_binary_path.empty() ||
      output_textproto_path.empty()) {
    return absl::InvalidArgumentError(
        "config, input_corpus, data_file, output_binary, and output_textproto "
        "are required");
  }

  absl::StatusOr<std::string> config_bytes = FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  const std::string config_sha256 = Sha256Bytes(*config_bytes);
  QualityRegressionFreezerConfig freezer_config;
  absl::Status status = ParseTextproto(*config_bytes, &freezer_config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionFreezerConfig(freezer_config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> input_bytes =
      FileUtil::GetContents(input_corpus_path);
  if (!input_bytes.ok()) {
    return input_bytes.status();
  }
  const std::string input_sha256 = Sha256Bytes(*input_bytes);
  if (input_sha256 != freezer_config.input_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression input corpus SHA256 mismatch");
  }
  QualityRegressionInputCorpus input_corpus;
  if (!input_corpus.ParseFromString(*input_bytes)) {
    return absl::InvalidArgumentError(
        "quality-regression input corpus binary is invalid");
  }
  status = ValidateQualityRegressionFreezerInput(
      input_corpus, input_sha256, freezer_config);
  if (!status.ok()) {
    return status;
  }

  const commands::Request desktop_request =
      MakeQualityRegressionDefaultDesktopRequest();
  const config::Config desktop_config =
      MakeQualityRegressionDefaultDesktopConfig();
  absl::StatusOr<std::string> request_sha256 =
      DeterministicMessageSha256(desktop_request);
  if (!request_sha256.ok()) {
    return request_sha256.status();
  }
  absl::StatusOr<std::string> desktop_config_sha256 =
      DeterministicMessageSha256(desktop_config);
  if (!desktop_config_sha256.ok()) {
    return desktop_config_sha256.status();
  }
  if (*request_sha256 !=
          freezer_config.mozc().default_desktop_request_sha256() ||
      *desktop_config_sha256 !=
          freezer_config.mozc().default_desktop_config_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression desktop Request or Config SHA256 mismatch");
  }

  absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>> runtime =
      EvaluationFreezerRuntime::Create(
          data_file_path, freezer_config.mozc().data_sha256(),
          freezer_config.mozc().data_type(),
          freezer_config.mozc().evaluation_clock_utc_rfc3339());
  if (!runtime.ok()) {
    return runtime.status();
  }
  if ((*runtime)->data_sha256() != freezer_config.mozc().data_sha256() ||
      (*runtime)->data_type() != freezer_config.mozc().data_type() ||
      (*runtime)->evaluation_clock_utc_rfc3339() !=
          freezer_config.mozc().evaluation_clock_utc_rfc3339()) {
    return absl::InternalError(
        "quality-regression runtime identity does not match freezer config");
  }

  absl::StatusOr<QualityRegressionFrozenCorpus> frozen =
      FreezeQualityRegressionCorpus(
          input_corpus, input_sha256, freezer_config, config_sha256,
          (*runtime)->converter());
  if (!frozen.ok()) {
    return frozen.status();
  }
  absl::StatusOr<std::string> frozen_binary =
      SerializeDeterministically(*frozen);
  absl::StatusOr<std::string> frozen_text = ReviewTextproto(*frozen);
  if (!frozen_binary.ok()) {
    return frozen_binary.status();
  }
  if (!frozen_text.ok()) {
    return frozen_text.status();
  }

  status = FileUtil::SetContents(output_binary_path, *frozen_binary);
  if (!status.ok()) {
    return status;
  }
  return FileUtil::SetContents(output_textproto_path, *frozen_text);
}

}  // namespace
}  // namespace mozc::engine::evaluation

namespace {

int MainUtf8(int argc, char** argv) {
  const absl::Status argument_status =
      mozc::engine::evaluation::RejectAnswerCorpusFlag(argc, argv);
  if (!argument_status.ok()) {
    std::cerr << argument_status << '\n';
    return static_cast<int>(argument_status.code());
  }
  mozc::InitMozc(argv[0], &argc, &argv);
  const absl::Status status = mozc::engine::evaluation::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return static_cast<int>(status.code());
  }
  return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  mozc::win32::Utf8ConsoleArgv utf8_argv(argc, argv);
  return MainUtf8(utf8_argv.argc(), utf8_argv.argv());
}
#else
int main(int argc, char** argv) { return MainUtf8(argc, argv); }
#endif
