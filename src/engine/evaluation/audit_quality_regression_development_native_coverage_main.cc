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

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_development_config.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_frozen_corpus_util.h"
#include "engine/evaluation/quality_regression_native_coverage.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "engine/llama_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "",
          "Checked quality-regression development execution config");
ABSL_FLAG(std::string, frozen_corpus, "",
          "Deterministic development frozen corpus");
ABSL_FLAG(std::string, manifest_directory, "",
          "Directory containing the three checked model manifests");
ABSL_FLAG(std::string, model_directory, "", "Directory containing the GGUF");
ABSL_FLAG(std::string, output_binary, "", "Native-coverage binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Native-coverage review textproto output");

namespace mozc::engine::evaluation {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

absl::Status ValidateCommandLine(int argc, char** argv) {
  constexpr std::array<absl::string_view, 6> kPrefixes = {
      "--config=",          "--frozen_corpus=", "--manifest_directory=",
      "--model_directory=", "--output_binary=", "--output_textproto=",
  };
  if (argc != kPrefixes.size() + 1) {
    return absl::InvalidArgumentError(
        "native coverage requires exactly six flags");
  }
  std::array<bool, kPrefixes.size()> seen = {};
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const absl::string_view argument = argv[argument_index];
    bool matched = false;
    for (size_t prefix_index = 0; prefix_index < kPrefixes.size();
         ++prefix_index) {
      if (absl::StartsWith(argument, kPrefixes[prefix_index]) &&
          argument.size() > kPrefixes[prefix_index].size() &&
          !seen[prefix_index]) {
        seen[prefix_index] = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return absl::InvalidArgumentError(
          "native coverage command line has an unknown, empty, or duplicate "
          "flag");
    }
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string frozen_corpus_path =
      absl::GetFlag(FLAGS_frozen_corpus);
  const std::string manifest_directory =
      absl::GetFlag(FLAGS_manifest_directory);
  const std::string model_directory =
      absl::GetFlag(FLAGS_model_directory);
  const std::string output_binary_path =
      absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);
  if (config_path.empty() || frozen_corpus_path.empty() ||
      manifest_directory.empty() || model_directory.empty() ||
      output_binary_path.empty() || output_textproto_path.empty()) {
    return absl::InvalidArgumentError(
        "config, frozen_corpus, manifest_directory, model_directory, "
        "output_binary, and output_textproto are required");
  }

  absl::StatusOr<std::string> config_bytes =
      FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  const std::string execution_config_sha256 = Sha256Bytes(*config_bytes);
  DevelopmentExecutionConfig config;
  absl::Status status = ParseTextproto(*config_bytes, &config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionDevelopmentExecutionConfig(config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> frozen_bytes =
      FileUtil::GetContents(frozen_corpus_path);
  if (!frozen_bytes.ok()) {
    return frozen_bytes.status();
  }
  const std::string frozen_corpus_sha256 = Sha256Bytes(*frozen_bytes);
  if (frozen_corpus_sha256 != config.frozen_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "quality-regression development frozen corpus SHA256 mismatch");
  }
  QualityRegressionFrozenCorpus frozen_corpus;
  if (!frozen_corpus.ParseFromString(*frozen_bytes) ||
      !frozen_corpus.IsInitialized()) {
    return absl::InvalidArgumentError(
        "quality-regression development frozen corpus binary is invalid");
  }
  status = ValidateQualityRegressionFrozenCorpusStructure(frozen_corpus);
  if (!status.ok()) {
    return status;
  }
  if (frozen_corpus.schema_version() !=
          config.frozen_corpus_schema_version() ||
      frozen_corpus.identity().role() != DEVELOPMENT ||
      frozen_corpus.cases_size() != config.expected_case_count()) {
    return absl::FailedPreconditionError(
        "quality-regression development frozen corpus identity mismatch");
  }

  std::vector<std::string> manifest_paths;
  std::vector<std::string> manifest_bytes;
  std::vector<std::string> manifest_sha256s;
  std::vector<CandidateRankerModelManifest> manifests;
  manifest_paths.reserve(config.objectives_size());
  manifest_bytes.reserve(config.objectives_size());
  manifest_sha256s.reserve(config.objectives_size());
  manifests.reserve(config.objectives_size());
  for (const DevelopmentObjectiveSpec& objective : config.objectives()) {
    manifest_paths.push_back(FileUtil::JoinPath(
        manifest_directory, objective.manifest_filename()));
    absl::StatusOr<std::string> bytes =
        FileUtil::GetContents(manifest_paths.back());
    if (!bytes.ok()) {
      return bytes.status();
    }
    const std::string sha256 = Sha256Bytes(*bytes);
    if (sha256 != objective.manifest_sha256()) {
      return absl::FailedPreconditionError(
          "quality-regression development manifest SHA256 mismatch");
    }
    CandidateRankerModelManifest manifest;
    status = ParseTextproto(*bytes, &manifest);
    if (!status.ok()) {
      return status;
    }
    manifest_bytes.push_back(std::move(*bytes));
    manifest_sha256s.push_back(sha256);
    manifests.push_back(std::move(manifest));
  }
  status = ValidateQualityRegressionDevelopmentManifests(
      config, manifests, manifest_sha256s);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>> auditors;
  auditors.reserve(config.objectives_size());
  for (int objective_index = 0;
       objective_index < config.objectives_size(); ++objective_index) {
    const LlamaCandidateRankerEvaluationIdentity expected_identity =
        BuildLlamaCandidateRankerEvaluationIdentity(
            manifests[objective_index], manifest_bytes[objective_index]);
    absl::StatusOr<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>>
        auditor = LlamaCandidateRankerCapacityAuditor::Create(
            manifest_paths[objective_index], model_directory,
            expected_identity);
    if (!auditor.ok()) {
      return auditor.status();
    }
    auditors.push_back(std::move(*auditor));
  }

  NeverCancelled cancellation;
  const auto capacity_callback =
      [&config, &auditors, &cancellation](
          const DevelopmentObjectiveSpec& objective,
          const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    for (int objective_index = 0;
         objective_index < config.objectives_size(); ++objective_index) {
      if (objective.objective() ==
          config.objectives(objective_index).objective()) {
        return auditors[objective_index]->Audit(request, cancellation);
      }
    }
    return absl::InvalidArgumentError(
        "quality-regression development objective is invalid");
  };
  absl::StatusOr<DevelopmentNativeCoverageSuite> suite =
      BuildQualityRegressionDevelopmentNativeCoverageSuite(
          frozen_corpus, frozen_corpus_sha256, config,
          execution_config_sha256, capacity_callback);
  auditors.clear();
  if (!suite.ok()) {
    return suite.status();
  }

  absl::StatusOr<std::string> binary = SerializeDeterministically(*suite);
  if (!binary.ok()) {
    return binary.status();
  }
  absl::StatusOr<std::string> review_text = ReviewTextproto(*suite);
  if (!review_text.ok()) {
    return review_text.status();
  }
  status = FileUtil::SetContents(output_binary_path, *binary);
  if (!status.ok()) {
    return status;
  }
  return FileUtil::SetContents(output_textproto_path, *review_text);
}

}  // namespace
}  // namespace mozc::engine::evaluation

namespace {

int MainUtf8(int argc, char** argv) {
  const absl::Status argument_status =
      mozc::engine::evaluation::ValidateCommandLine(argc, argv);
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
