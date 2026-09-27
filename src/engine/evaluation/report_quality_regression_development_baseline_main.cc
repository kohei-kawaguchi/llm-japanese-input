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
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/quality_regression_baseline_report.h"
#include "engine/evaluation/quality_regression_baseline_report.pb.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "",
          "Checked development baseline-report textproto");
ABSL_FLAG(std::string, frozen_corpus, "",
          "Accepted development frozen corpus binary");
ABSL_FLAG(std::string, answer_corpus, "",
          "Accepted development answer corpus binary");
ABSL_FLAG(std::string, output_binary, "", "Baseline report binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Baseline report review textproto output");

namespace mozc::engine::evaluation {
namespace {

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string frozen_corpus_path = absl::GetFlag(FLAGS_frozen_corpus);
  const std::string answer_corpus_path = absl::GetFlag(FLAGS_answer_corpus);
  const std::string output_binary_path = absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);
  if (config_path.empty() || frozen_corpus_path.empty() ||
      answer_corpus_path.empty() || output_binary_path.empty() ||
      output_textproto_path.empty()) {
    return absl::InvalidArgumentError(
        "config, frozen_corpus, answer_corpus, output_binary, and "
        "output_textproto are required");
  }

  absl::StatusOr<std::string> config_bytes = FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  const std::string config_sha256 = Sha256Bytes(*config_bytes);
  QualityRegressionDevelopmentBaselineReportConfig config;
  absl::Status status = ParseTextproto(*config_bytes, &config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionDevelopmentBaselineReportConfig(config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> frozen_bytes =
      FileUtil::GetContents(frozen_corpus_path);
  if (!frozen_bytes.ok()) {
    return frozen_bytes.status();
  }
  const std::string frozen_sha256 = Sha256Bytes(*frozen_bytes);
  absl::StatusOr<std::string> answer_bytes =
      FileUtil::GetContents(answer_corpus_path);
  if (!answer_bytes.ok()) {
    return answer_bytes.status();
  }
  const std::string answer_sha256 = Sha256Bytes(*answer_bytes);
  if (frozen_sha256 != config.frozen_corpus_sha256() ||
      answer_sha256 != config.answer_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression baseline input SHA256 mismatch");
  }

  QualityRegressionFrozenCorpus frozen_corpus;
  if (!frozen_corpus.ParseFromString(*frozen_bytes)) {
    return absl::InvalidArgumentError(
        "quality-regression frozen corpus binary is invalid");
  }
  QualityRegressionAnswerCorpus answer_corpus;
  if (!answer_corpus.ParseFromString(*answer_bytes)) {
    return absl::InvalidArgumentError(
        "quality-regression answer corpus binary is invalid");
  }

  absl::StatusOr<QualityRegressionDevelopmentBaselineReport> report =
      BuildQualityRegressionDevelopmentBaselineReport(
          frozen_corpus, frozen_sha256, answer_corpus, answer_sha256, config,
          config_sha256);
  if (!report.ok()) {
    return report.status();
  }
  absl::StatusOr<std::string> report_binary =
      SerializeDeterministically(*report);
  absl::StatusOr<std::string> report_text = ReviewTextproto(*report);
  if (!report_binary.ok()) {
    return report_binary.status();
  }
  if (!report_text.ok()) {
    return report_text.status();
  }

  status = FileUtil::SetContents(output_binary_path, *report_binary);
  if (!status.ok()) {
    return status;
  }
  return FileUtil::SetContents(output_textproto_path, *report_text);
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
