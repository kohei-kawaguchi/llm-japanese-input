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
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_corpus.pb.h"
#include "engine/evaluation/ajimee_freezer.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/evaluation_freezer_runtime.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "", "Checked AJIMEE freezer textproto");
ABSL_FLAG(std::string, input_corpus, "", "Deterministic AJIMEE input corpus");
ABSL_FLAG(std::string, data_file, "", "Pinned OSS mozc.data path");
ABSL_FLAG(std::string, output_binary, "", "Frozen corpus binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Frozen corpus review textproto output");

namespace mozc::engine::evaluation {
namespace {

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
        "config, input_corpus, data_file, output_binary, and "
        "output_textproto are required");
  }

  absl::StatusOr<std::string> config_bytes = FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  AjimeeFreezerConfig freezer_config;
  absl::Status status = ParseTextproto(*config_bytes, &freezer_config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeFreezerConfig(freezer_config);
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
    return absl::FailedPreconditionError("AJIMEE input corpus SHA256 mismatch");
  }
  AjimeeInputCorpus input_corpus;
  if (!input_corpus.ParseFromString(*input_bytes) ||
      !input_corpus.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE input corpus binary is invalid");
  }
  status =
      ValidateAjimeeFreezerInput(input_corpus, input_sha256, freezer_config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>> runtime =
      EvaluationFreezerRuntime::Create(
          data_file_path, freezer_config.mozc_data_sha256(),
          freezer_config.mozc_data_type(),
          freezer_config.evaluation_clock_utc_rfc3339());
  if (!runtime.ok()) {
    return runtime.status();
  }

  const commands::Request desktop_request = MakeAjimeeDefaultDesktopRequest();
  const config::Config desktop_config = MakeAjimeeDefaultDesktopConfig();
  absl::StatusOr<AjimeeFrozenCorpus> frozen = FreezeAjimeeCorpus(
      input_corpus, input_sha256, freezer_config, desktop_request,
      desktop_config, (*runtime)->converter());
  if (!frozen.ok()) {
    return frozen.status();
  }
  absl::StatusOr<std::string> frozen_binary =
      SerializeDeterministically(*frozen);
  if (!frozen_binary.ok()) {
    return frozen_binary.status();
  }
  absl::StatusOr<std::string> frozen_text = ReviewTextproto(*frozen);
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
