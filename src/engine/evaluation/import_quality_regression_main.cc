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
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_importer.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "",
          "Checked quality-regression importer textproto");
ABSL_FLAG(std::string, development_source, "", "Checked OSS TSV source");
ABSL_FLAG(std::string, holdout_source, "", "Checked holdout TSV source");
ABSL_FLAG(std::string, output_directory, "",
          "Existing output directory for deterministic corpora");

namespace mozc::engine::evaluation {
namespace {

absl::Status WriteArtifact(absl::string_view directory,
                           absl::string_view filename,
                           absl::string_view contents) {
  return FileUtil::SetContents(FileUtil::JoinPath(directory, filename),
                               contents);
}

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string development_path =
      absl::GetFlag(FLAGS_development_source);
  const std::string holdout_path = absl::GetFlag(FLAGS_holdout_source);
  const std::string output_directory =
      absl::GetFlag(FLAGS_output_directory);
  if (config_path.empty() || development_path.empty() || holdout_path.empty() ||
      output_directory.empty()) {
    return absl::InvalidArgumentError(
        "config, development_source, holdout_source, and output_directory are "
        "required");
  }
  absl::Status status = FileUtil::DirectoryExists(output_directory);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> config_bytes =
      FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  const std::string config_sha256 = Sha256Bytes(*config_bytes);
  QualityRegressionImporterConfig config;
  status = ParseTextproto(*config_bytes, &config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionImporterConfig(config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> development_source =
      FileUtil::GetContents(development_path);
  if (!development_source.ok()) {
    return development_source.status();
  }
  absl::StatusOr<std::string> holdout_source =
      FileUtil::GetContents(holdout_path);
  if (!holdout_source.ok()) {
    return holdout_source.status();
  }
  absl::StatusOr<QualityRegressionCorpora> corpora =
      ImportQualityRegressionCorpora(*development_source, *holdout_source,
                                     config, config_sha256);
  if (!corpora.ok()) {
    return corpora.status();
  }

  absl::StatusOr<std::string> development_input_binary =
      SerializeDeterministically(corpora->development_inputs);
  absl::StatusOr<std::string> development_input_text =
      ReviewTextproto(corpora->development_inputs);
  absl::StatusOr<std::string> development_answer_binary =
      SerializeDeterministically(corpora->development_answers);
  absl::StatusOr<std::string> development_answer_text =
      ReviewTextproto(corpora->development_answers);
  absl::StatusOr<std::string> holdout_input_binary =
      SerializeDeterministically(corpora->holdout_inputs);
  absl::StatusOr<std::string> holdout_input_text =
      ReviewTextproto(corpora->holdout_inputs);
  absl::StatusOr<std::string> holdout_answer_binary =
      SerializeDeterministically(corpora->holdout_answers);
  absl::StatusOr<std::string> holdout_answer_text =
      ReviewTextproto(corpora->holdout_answers);
  if (!development_input_binary.ok()) {
    return development_input_binary.status();
  }
  if (!development_input_text.ok()) {
    return development_input_text.status();
  }
  if (!development_answer_binary.ok()) {
    return development_answer_binary.status();
  }
  if (!development_answer_text.ok()) {
    return development_answer_text.status();
  }
  if (!holdout_input_binary.ok()) {
    return holdout_input_binary.status();
  }
  if (!holdout_input_text.ok()) {
    return holdout_input_text.status();
  }
  if (!holdout_answer_binary.ok()) {
    return holdout_answer_binary.status();
  }
  if (!holdout_answer_text.ok()) {
    return holdout_answer_text.status();
  }

  status = WriteArtifact(output_directory,
                         "quality_regression_development_input_corpus.pb",
                         *development_input_binary);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(
      output_directory,
      "quality_regression_development_input_corpus.textproto",
      *development_input_text);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(output_directory,
                         "quality_regression_development_answer_corpus.pb",
                         *development_answer_binary);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(
      output_directory,
      "quality_regression_development_answer_corpus.textproto",
      *development_answer_text);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(output_directory,
                         "quality_regression_holdout_input_corpus.pb",
                         *holdout_input_binary);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(output_directory,
                         "quality_regression_holdout_input_corpus.textproto",
                         *holdout_input_text);
  if (!status.ok()) {
    return status;
  }
  status = WriteArtifact(output_directory,
                         "quality_regression_holdout_answer_corpus.pb",
                         *holdout_answer_binary);
  if (!status.ok()) {
    return status;
  }
  return WriteArtifact(
      output_directory,
      "quality_regression_holdout_answer_corpus.textproto",
      *holdout_answer_text);
}

}  // namespace
}  // namespace mozc::engine::evaluation

namespace {

int MainUtf8(int argc, char** argv) {
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
