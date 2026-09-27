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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/evaluation_freezer.h"
#include "engine/evaluation/evaluation_freezer_runtime.h"
#include "engine/evaluation/evaluation_protobuf_util.h"
#include "engine/evaluation/prediction_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_freezer.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, freezer_config, "",
          "Checked quality-regression freezer textproto for the Mozc identity");
ABSL_FLAG(std::string, data_file, "", "Pinned OSS mozc.data path");
ABSL_FLAG(std::string, input_tsv, "",
          "UTF-8 lines of source line and typed reading prefix");
ABSL_FLAG(std::string, output_binary, "", "Prediction frozen corpus output");

namespace mozc::engine::evaluation {
namespace {

constexpr uint32_t kPredictionFrozenCorpusSchemaVersion = 1;

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_freezer_config);
  const std::string data_file_path = absl::GetFlag(FLAGS_data_file);
  const std::string input_path = absl::GetFlag(FLAGS_input_tsv);
  const std::string output_path = absl::GetFlag(FLAGS_output_binary);
  if (config_path.empty() || data_file_path.empty() || input_path.empty() ||
      output_path.empty()) {
    return absl::InvalidArgumentError(
        "freezer_config, data_file, input_tsv, and output_binary are required");
  }
  absl::StatusOr<std::string> config_bytes = FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  QualityRegressionFreezerConfig freezer_config;
  absl::Status status = ParseTextproto(*config_bytes, &freezer_config);
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<std::string> input = FileUtil::GetContents(input_path);
  if (!input.ok()) {
    return input.status();
  }

  absl::StatusOr<std::unique_ptr<EvaluationFreezerRuntime>> runtime =
      EvaluationFreezerRuntime::Create(
          data_file_path, freezer_config.mozc().data_sha256(),
          freezer_config.mozc().data_type(),
          freezer_config.mozc().evaluation_clock_utc_rfc3339());
  if (!runtime.ok()) {
    return runtime.status();
  }
  absl::StatusOr<std::unique_ptr<EvaluationFreezerSession>> session =
      EvaluationFreezerSession::Create(
          freezer_config.mozc().evaluation_clock_utc_rfc3339(),
          MakeQualityRegressionDefaultDesktopRequest(),
          MakeQualityRegressionDefaultDesktopConfig(), (*runtime)->converter());
  if (!session.ok()) {
    return session.status();
  }

  PredictionFrozenCorpus corpus;
  corpus.set_schema_version(kPredictionFrozenCorpusSchemaVersion);
  uint64_t request_sequence = 0;
  for (absl::string_view line :
       absl::StrSplit(*input, '\n', absl::SkipEmpty())) {
    const std::vector<absl::string_view> fields = absl::StrSplit(line, '\t');
    uint64_t source_line = 0;
    if (fields.size() != 2 || !absl::SimpleAtoi(fields[0], &source_line) ||
        fields[1].empty()) {
      return absl::InvalidArgumentError("input_tsv line is malformed");
    }
    absl::StatusOr<EvaluationFreezeCaseResult> result = (*session)->FreezeCase({
        .request_sequence = ++request_sequence,
        .preedit_text = fields[1],
        .reading_source = FrozenReadingSource::kPreeditText,
        .mode = converter::CandidateRankerMode::kPrediction,
    });
    if (!result.ok()) {
      return result.status();
    }
    PredictionFrozenCase& frozen_case = *corpus.add_cases();
    frozen_case.set_source_line(source_line);
    frozen_case.set_preedit(fields[1]);
    frozen_case.set_mozc_baseline_output(result->baseline_output);
    *frozen_case.mutable_request() = std::move(result->request);
  }
  absl::StatusOr<std::string> binary = SerializeDeterministically(corpus);
  if (!binary.ok()) {
    return binary.status();
  }
  return FileUtil::SetContents(output_path, *binary);
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
