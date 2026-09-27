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
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/quality_regression_development_config.h"
#include "engine/evaluation/quality_regression_development_scores.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/evaluation/quality_regression_native_coverage.h"
#include "engine/evaluation/quality_regression_objective_evaluation.pb.h"
#include "engine/llama_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, score_config, "",
          "Checked quality-regression development score config");
ABSL_FLAG(std::string, execution_config, "",
          "Checked quality-regression development execution config");
ABSL_FLAG(std::string, frozen_corpus, "",
          "Deterministic development frozen corpus");
ABSL_FLAG(std::string, native_coverage, "",
          "Accepted development native-coverage suite");
ABSL_FLAG(std::string, manifest_directory, "",
          "Directory containing the three checked model manifests");
ABSL_FLAG(std::string, model_directory, "", "Directory containing the GGUF");
ABSL_FLAG(std::string, output_binary, "", "Development-score binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Development-score review textproto output");

namespace mozc::engine::evaluation {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

absl::Status ValidateCommandLine(int argc, char** argv) {
  constexpr std::array<absl::string_view, 8> kPrefixes = {
      "--score_config=",    "--execution_config=",   "--frozen_corpus=",
      "--native_coverage=", "--manifest_directory=", "--model_directory=",
      "--output_binary=",   "--output_textproto=",
  };
  if (argc != kPrefixes.size() + 1) {
    return absl::InvalidArgumentError(
        "development scores require exactly eight flags");
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
          "development score command line has an unknown, empty, duplicate, "
          "or split-form flag");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> FindObjectiveIndex(
    const DevelopmentExecutionConfig& config,
    const DevelopmentObjectiveSpec& objective) {
  for (int objective_index = 0; objective_index < config.objectives_size();
       ++objective_index) {
    if (objective.objective() ==
        config.objectives(objective_index).objective()) {
      return static_cast<size_t>(objective_index);
    }
  }
  return absl::InvalidArgumentError(
      "quality-regression development objective is invalid");
}

absl::Status Run() {
  const std::string score_config_path = absl::GetFlag(FLAGS_score_config);
  const std::string execution_config_path =
      absl::GetFlag(FLAGS_execution_config);
  const std::string frozen_corpus_path = absl::GetFlag(FLAGS_frozen_corpus);
  const std::string native_coverage_path = absl::GetFlag(FLAGS_native_coverage);
  const std::string manifest_directory =
      absl::GetFlag(FLAGS_manifest_directory);
  const std::string model_directory = absl::GetFlag(FLAGS_model_directory);
  const std::string output_binary_path = absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);

  absl::StatusOr<std::string> score_config_bytes =
      FileUtil::GetContents(score_config_path);
  if (!score_config_bytes.ok()) {
    return score_config_bytes.status();
  }
  const std::string score_config_sha256 = Sha256Bytes(*score_config_bytes);
  DevelopmentScoreRunnerConfig score_config;
  absl::Status status = ParseTextproto(*score_config_bytes, &score_config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateQualityRegressionDevelopmentScoreRunnerConfig(score_config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> execution_config_bytes =
      FileUtil::GetContents(execution_config_path);
  if (!execution_config_bytes.ok()) {
    return execution_config_bytes.status();
  }
  const std::string execution_config_sha256 =
      Sha256Bytes(*execution_config_bytes);
  DevelopmentExecutionConfig execution_config;
  status = ParseTextproto(*execution_config_bytes, &execution_config);
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateQualityRegressionDevelopmentExecutionConfig(execution_config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> frozen_corpus_bytes =
      FileUtil::GetContents(frozen_corpus_path);
  if (!frozen_corpus_bytes.ok()) {
    return frozen_corpus_bytes.status();
  }
  const std::string frozen_corpus_sha256 = Sha256Bytes(*frozen_corpus_bytes);
  QualityRegressionFrozenCorpus frozen_corpus;
  if (!frozen_corpus.ParseFromString(*frozen_corpus_bytes) ||
      !frozen_corpus.IsInitialized()) {
    return absl::InvalidArgumentError(
        "quality-regression development frozen corpus binary is invalid");
  }

  absl::StatusOr<std::string> native_coverage_bytes =
      FileUtil::GetContents(native_coverage_path);
  if (!native_coverage_bytes.ok()) {
    return native_coverage_bytes.status();
  }
  const std::string native_coverage_sha256 =
      Sha256Bytes(*native_coverage_bytes);
  DevelopmentNativeCoverageSuite native_coverage;
  if (!native_coverage.ParseFromString(*native_coverage_bytes) ||
      !native_coverage.IsInitialized()) {
    return absl::InvalidArgumentError(
        "quality-regression development native coverage binary is invalid");
  }

  std::vector<std::string> manifest_paths;
  std::vector<std::string> manifest_bytes;
  std::vector<std::string> manifest_sha256s;
  std::vector<CandidateRankerModelManifest> manifests;
  manifest_paths.reserve(execution_config.objectives_size());
  manifest_bytes.reserve(execution_config.objectives_size());
  manifest_sha256s.reserve(execution_config.objectives_size());
  manifests.reserve(execution_config.objectives_size());
  for (const DevelopmentObjectiveSpec& objective :
       execution_config.objectives()) {
    manifest_paths.push_back(
        FileUtil::JoinPath(manifest_directory, objective.manifest_filename()));
    absl::StatusOr<std::string> bytes =
        FileUtil::GetContents(manifest_paths.back());
    if (!bytes.ok()) {
      return bytes.status();
    }
    manifest_sha256s.push_back(Sha256Bytes(*bytes));
    manifest_bytes.push_back(std::move(*bytes));
  }
  for (const std::string& bytes : manifest_bytes) {
    CandidateRankerModelManifest manifest;
    status = ParseTextproto(bytes, &manifest);
    if (!status.ok()) {
      return status;
    }
    manifests.push_back(std::move(manifest));
  }

  status = ValidateQualityRegressionDevelopmentScoreRunnerInputs(
      score_config, score_config_sha256, execution_config,
      execution_config_sha256, frozen_corpus, frozen_corpus_sha256,
      native_coverage, native_coverage_sha256, manifests, manifest_sha256s);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>> auditors;
  auditors.reserve(execution_config.objectives_size());
  for (int objective_index = 0;
       objective_index < execution_config.objectives_size();
       ++objective_index) {
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
      [&execution_config, &auditors, &cancellation](
          const DevelopmentObjectiveSpec& objective,
          const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    absl::StatusOr<size_t> objective_index =
        FindObjectiveIndex(execution_config, objective);
    if (!objective_index.ok()) {
      return objective_index.status();
    }
    return auditors[*objective_index]->Audit(request, cancellation);
  };

  std::shared_ptr<LlamaCandidateRanker> ranker;
  size_t ranker_objective_index = manifests.size();
  const auto score_callback =
      [&execution_config, &manifest_paths, &manifest_bytes, &manifests,
       &model_directory, &auditors, &ranker, &ranker_objective_index,
       &cancellation](const DevelopmentObjectiveSpec& objective,
                      const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<QualityRegressionDevelopmentEvaluationResult> {
    auditors.clear();
    absl::StatusOr<size_t> objective_index =
        FindObjectiveIndex(execution_config, objective);
    if (!objective_index.ok()) {
      return objective_index.status();
    }
    if (ranker_objective_index != *objective_index) {
      ranker.reset();
      const LlamaCandidateRankerEvaluationIdentity expected_identity =
          BuildLlamaCandidateRankerEvaluationIdentity(
              manifests[*objective_index], manifest_bytes[*objective_index]);
      absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> created =
          LlamaCandidateRanker::Create(manifest_paths[*objective_index],
                                       model_directory, expected_identity);
      if (!created.ok()) {
        return created.status();
      }
      ranker = std::move(*created);
      ranker_objective_index = *objective_index;
    }

    absl::StatusOr<LlamaCandidateRankerEvaluationResult> result =
        ranker->ScoreForEvaluation(
            request, cancellation,
            LlamaCandidateRankerEvaluationLayout::kSharedTrie,
            LlamaCandidateRankerEvaluationProfile::kConfigured);
    if (!result.ok()) {
      return result.status();
    }
    QualityRegressionDevelopmentEvaluationResult adapted;
    adapted.response = std::move(result->response);
    adapted.candidate_scores.reserve(result->candidate_scores.size());
    for (const LlamaCandidateRankerCandidateScore& candidate :
         result->candidate_scores) {
      adapted.candidate_scores.push_back(
          {.segment_id = candidate.segment_id,
           .candidate_id = candidate.candidate_id,
           .full_continuation_log_probability =
               candidate.full_continuation_log_probability,
           .scored_continuation_token_count =
               candidate.scored_continuation_token_count});
    }
    return adapted;
  };

  absl::StatusOr<DevelopmentObjectiveScores> scores =
      BuildQualityRegressionDevelopmentObjectiveScores(
          score_config, score_config_sha256, execution_config,
          execution_config_sha256, frozen_corpus, frozen_corpus_sha256,
          native_coverage, native_coverage_sha256, manifests, manifest_sha256s,
          capacity_callback, score_callback);
  auditors.clear();
  ranker.reset();
  if (!scores.ok()) {
    return scores.status();
  }

  absl::StatusOr<std::string> binary = SerializeDeterministically(*scores);
  if (!binary.ok()) {
    return binary.status();
  }
  absl::StatusOr<std::string> review_text = ReviewTextproto(*scores);
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
