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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/llama_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, manifest_path, "", "Checked model manifest path");
ABSL_FLAG(std::string, model_directory, "", "Directory containing the GGUF");
ABSL_FLAG(std::string, candidates_path, "",
          "UTF-8 file with one candidate value per line");
ABSL_FLAG(std::string, mode, "conversion",
          "Ranking mode: suggestion, prediction, or conversion");
ABSL_FLAG(std::string, reading, "", "Complete composition reading");
ABSL_FLAG(std::string, segment_key, "", "Focused segment reading");
ABSL_FLAG(std::string, preceding_text, "", "Text before the composition");
ABSL_FLAG(std::string, following_text, "", "Text after the composition");
ABSL_FLAG(double, tolerance, 0.0001,
          "Absolute shared-versus-independent score tolerance");
ABSL_FLAG(bool, check_context_sensitivity, false,
          "Compare the reference independent scores under alternate context");
ABSL_FLAG(std::string, alternate_preceding_text, "",
          "Alternate text before the composition");
ABSL_FLAG(std::string, alternate_following_text, "",
          "Alternate text after the composition");

namespace mozc {
namespace engine {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

absl::StatusOr<converter::CandidateRankerMode> ParseMode(
    const std::string& mode) {
  if (mode == "suggestion") {
    return converter::CandidateRankerMode::kSuggestion;
  }
  if (mode == "prediction") {
    return converter::CandidateRankerMode::kPrediction;
  }
  if (mode == "conversion") {
    return converter::CandidateRankerMode::kConversion;
  }
  return absl::InvalidArgumentError("ranking mode is invalid");
}

const LlamaCandidateRankerCandidateScore* FindCandidateScore(
    const LlamaCandidateRankerEvaluationResult& result,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id) {
  for (const LlamaCandidateRankerCandidateScore& score :
       result.candidate_scores) {
    if (score.segment_id == segment_id && score.candidate_id == candidate_id) {
      return &score;
    }
  }
  return nullptr;
}

converter::CandidateRankerCandidateId Winner(
    const LlamaCandidateRankerEvaluationResult& result) {
  if (result.response.segment_orders.size() != 1 ||
      result.response.segment_orders.front().candidate_ids.empty()) {
    return 0;
  }
  return result.response.segment_orders.front().candidate_ids.front();
}

std::optional<converter::CandidateRankerCandidateId> UniqueWinner(
    const LlamaCandidateRankerEvaluationResult& result) {
  double maximum_score = -std::numeric_limits<double>::infinity();
  converter::CandidateRankerCandidateId winner = 0;
  bool unique = false;
  for (const LlamaCandidateRankerCandidateScore& candidate :
       result.candidate_scores) {
    if (candidate.full_continuation_log_probability > maximum_score) {
      maximum_score = candidate.full_continuation_log_probability;
      winner = candidate.candidate_id;
      unique = true;
    } else if (candidate.full_continuation_log_probability == maximum_score) {
      unique = false;
    }
  }
  return unique ? std::optional(winner) : std::nullopt;
}

absl::string_view ContributionName(
    LlamaCandidateRankerEdgeContribution contribution) {
  switch (contribution) {
    case LlamaCandidateRankerEdgeContribution::kCandidateOrBoundary:
      return "candidate_or_boundary";
    case LlamaCandidateRankerEdgeContribution::kFollowingContext:
      return "following_context";
    case LlamaCandidateRankerEdgeContribution::kTerminal:
      return "terminal";
  }
  return "invalid";
}

void PrintResult(absl::string_view name,
                 const LlamaCandidateRankerEvaluationResult& result) {
  std::cout << "run\t" << name << "\tinput_nodes\t"
            << result.input_node_count << "\toutput_rows\t"
            << result.output_row_count << "\twinner\t" << Winner(result)
            << '\n';
  for (const LlamaCandidateRankerCandidateScore& candidate :
       result.candidate_scores) {
    std::cout << "candidate\t" << name << '\t' << candidate.segment_id << '\t'
              << candidate.candidate_id << '\t'
              << candidate.common_prefix_token_count << '\t'
              << candidate.full_continuation_log_probability << '\n';
    std::cout << "tokens\t" << name << '\t' << candidate.segment_id << '\t'
              << candidate.candidate_id;
    for (const int32_t token_id : candidate.input_token_ids) {
      std::cout << '\t' << token_id;
    }
    std::cout << '\n';
    for (const LlamaCandidateRankerEdgeScore& edge : candidate.edges) {
      std::cout << "edge\t" << name << '\t' << candidate.segment_id << '\t'
                << candidate.candidate_id << '\t' << edge.target_position
                << '\t' << edge.target_token_id << '\t'
                << static_cast<int>(edge.is_terminal) << '\t'
                << ContributionName(edge.contribution) << '\t'
                << edge.log_probability << '\n';
    }
  }
}

bool CompareSharedAndIndependent(
    absl::string_view profile,
    const LlamaCandidateRankerEvaluationResult& shared,
    const LlamaCandidateRankerEvaluationResult& independent,
    double tolerance) {
  bool passed = shared.candidate_scores.size() ==
                independent.candidate_scores.size();
  for (const LlamaCandidateRankerCandidateScore& shared_candidate :
       shared.candidate_scores) {
    const LlamaCandidateRankerCandidateScore* independent_candidate =
        FindCandidateScore(independent, shared_candidate.segment_id,
                           shared_candidate.candidate_id);
    if (independent_candidate == nullptr) {
      passed = false;
      std::cout << "compare\t" << profile << '\t'
                << shared_candidate.segment_id << '\t'
                << shared_candidate.candidate_id
                << "\tmissing_independent_candidate\t0\n";
      continue;
    }
    bool candidate_passed =
        shared_candidate.common_prefix_token_count ==
            independent_candidate->common_prefix_token_count &&
        shared_candidate.input_token_ids ==
            independent_candidate->input_token_ids &&
        shared_candidate.edges.size() == independent_candidate->edges.size();
    double maximum_edge_delta = 0.0;
    const size_t comparable_edges =
        std::min(shared_candidate.edges.size(),
                 independent_candidate->edges.size());
    for (size_t i = 0; i < comparable_edges; ++i) {
      const LlamaCandidateRankerEdgeScore& shared_edge =
          shared_candidate.edges[i];
      const LlamaCandidateRankerEdgeScore& independent_edge =
          independent_candidate->edges[i];
      const double edge_delta = std::abs(shared_edge.log_probability -
                                         independent_edge.log_probability);
      maximum_edge_delta = std::max(maximum_edge_delta, edge_delta);
      candidate_passed =
          candidate_passed &&
          shared_edge.target_position == independent_edge.target_position &&
          shared_edge.target_token_id == independent_edge.target_token_id &&
          shared_edge.is_terminal == independent_edge.is_terminal &&
          shared_edge.contribution == independent_edge.contribution &&
          edge_delta <= tolerance;
    }
    const double score_delta =
        std::abs(shared_candidate.full_continuation_log_probability -
                 independent_candidate->full_continuation_log_probability);
    candidate_passed =
        candidate_passed && score_delta <= tolerance &&
        shared_candidate.scored_continuation_token_count ==
            independent_candidate->scored_continuation_token_count;
    passed = passed && candidate_passed;
    std::cout << "compare\t" << profile << '\t'
              << shared_candidate.segment_id << '\t'
              << shared_candidate.candidate_id << '\t' << score_delta << '\t'
              << maximum_edge_delta << '\t'
              << static_cast<int>(candidate_passed) << '\n';
  }
  passed = passed && Winner(shared) == Winner(independent);
  std::cout << "equivalence\t" << profile << '\t'
            << static_cast<int>(passed) << '\n';
  return passed;
}

bool ComparePermutation(
    absl::string_view rotation,
    const LlamaCandidateRankerEvaluationResult& base,
    const LlamaCandidateRankerEvaluationResult& permuted) {
  bool passed = base.candidate_scores.size() == permuted.candidate_scores.size();
  for (const LlamaCandidateRankerCandidateScore& base_candidate :
       base.candidate_scores) {
    const LlamaCandidateRankerCandidateScore* permuted_candidate =
        FindCandidateScore(permuted, base_candidate.segment_id,
                           base_candidate.candidate_id);
    bool candidate_passed = permuted_candidate != nullptr;
    if (permuted_candidate != nullptr) {
      candidate_passed =
          base_candidate.full_continuation_log_probability ==
              permuted_candidate->full_continuation_log_probability &&
          base_candidate.scored_continuation_token_count ==
              permuted_candidate->scored_continuation_token_count &&
          base_candidate.common_prefix_token_count ==
              permuted_candidate->common_prefix_token_count &&
          base_candidate.input_token_ids ==
              permuted_candidate->input_token_ids &&
          base_candidate.edges.size() == permuted_candidate->edges.size();
      for (size_t i = 0;
           candidate_passed && i < base_candidate.edges.size(); ++i) {
        const LlamaCandidateRankerEdgeScore& base_edge =
            base_candidate.edges[i];
        const LlamaCandidateRankerEdgeScore& permuted_edge =
            permuted_candidate->edges[i];
        candidate_passed =
            base_edge.target_position == permuted_edge.target_position &&
            base_edge.target_token_id == permuted_edge.target_token_id &&
            base_edge.is_terminal == permuted_edge.is_terminal &&
            base_edge.contribution == permuted_edge.contribution &&
            base_edge.log_probability == permuted_edge.log_probability;
      }
    }
    passed = passed && candidate_passed;
    std::cout << "permutation_candidate\t" << rotation << '\t'
              << base_candidate.segment_id << '\t'
              << base_candidate.candidate_id << '\t'
              << static_cast<int>(candidate_passed) << '\n';
  }
  passed = passed && UniqueWinner(base) == UniqueWinner(permuted);
  std::cout << "permutation\t" << rotation << '\t'
            << static_cast<int>(passed) << '\n';
  return passed;
}

bool CompareProductionResponse(
    const converter::CandidateRankerResponse& production,
    const converter::CandidateRankerResponse& evaluated) {
  bool passed = production.token == evaluated.token &&
                production.segment_orders.size() ==
                    evaluated.segment_orders.size();
  const size_t comparable_segments =
      std::min(production.segment_orders.size(),
               evaluated.segment_orders.size());
  for (size_t i = 0; i < comparable_segments; ++i) {
    passed = passed &&
             production.segment_orders[i].segment_id ==
                 evaluated.segment_orders[i].segment_id &&
             production.segment_orders[i].candidate_ids ==
                 evaluated.segment_orders[i].candidate_ids;
  }
  std::cout << "production_response\t" << static_cast<int>(passed) << '\n';
  return passed;
}

bool CompareContextMargins(
    const LlamaCandidateRankerEvaluationResult& base,
    const LlamaCandidateRankerEvaluationResult& alternate, double tolerance) {
  bool complete = base.candidate_scores.size() == alternate.candidate_scores.size();
  bool changed = false;
  for (size_t i = 0; i < base.candidate_scores.size(); ++i) {
    for (size_t j = i + 1; j < base.candidate_scores.size(); ++j) {
      const LlamaCandidateRankerCandidateScore& first =
          base.candidate_scores[i];
      const LlamaCandidateRankerCandidateScore& second =
          base.candidate_scores[j];
      const LlamaCandidateRankerCandidateScore* alternate_first =
          FindCandidateScore(alternate, first.segment_id, first.candidate_id);
      const LlamaCandidateRankerCandidateScore* alternate_second =
          FindCandidateScore(alternate, second.segment_id, second.candidate_id);
      if (alternate_first == nullptr || alternate_second == nullptr) {
        complete = false;
        continue;
      }
      const double base_margin = first.full_continuation_log_probability -
                                 second.full_continuation_log_probability;
      const double alternate_margin =
          alternate_first->full_continuation_log_probability -
          alternate_second->full_continuation_log_probability;
      const double margin_change = alternate_margin - base_margin;
      changed = changed || std::abs(margin_change) > tolerance;
      std::cout << "context_margin\t" << first.candidate_id << '\t'
                << second.candidate_id << '\t' << base_margin << '\t'
                << alternate_margin << '\t' << margin_change << '\n';
    }
  }
  const bool passed = complete && changed;
  std::cout << "context_sensitivity\t" << static_cast<int>(passed) << '\n';
  return passed;
}

absl::StatusOr<LlamaCandidateRankerEvaluationResult> Score(
    LlamaCandidateRanker* ranker,
    const converter::CandidateRankerRequest& request,
    LlamaCandidateRankerEvaluationLayout layout,
    LlamaCandidateRankerEvaluationProfile profile,
    const converter::CandidateRankerCancellation& cancellation) {
  return ranker->ScoreForEvaluation(request, cancellation, layout, profile);
}

int Run() {
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest_path);
  const std::string model_directory = absl::GetFlag(FLAGS_model_directory);
  const std::string candidates_path = absl::GetFlag(FLAGS_candidates_path);
  const std::string reading = absl::GetFlag(FLAGS_reading);
  const std::string segment_key = absl::GetFlag(FLAGS_segment_key);
  const double tolerance = absl::GetFlag(FLAGS_tolerance);
  if (manifest_path.empty() || model_directory.empty() ||
      candidates_path.empty() || reading.empty() || segment_key.empty() ||
      !std::isfinite(tolerance) || tolerance < 0.0) {
    std::cerr << "manifest_path, model_directory, candidates_path, reading, "
                 "segment_key, and a nonnegative finite tolerance are required\n";
    return 2;
  }
  const absl::StatusOr<converter::CandidateRankerMode> mode =
      ParseMode(absl::GetFlag(FLAGS_mode));
  if (!mode.ok()) {
    std::cerr << mode.status() << '\n';
    return 2;
  }
  const absl::StatusOr<std::string> candidate_contents =
      FileUtil::GetContents(candidates_path);
  if (!candidate_contents.ok()) {
    std::cerr << "candidate file is unavailable\n";
    return 2;
  }
  std::vector<std::string> candidate_values =
      absl::StrSplit(*candidate_contents, '\n', absl::SkipEmpty());
  for (std::string& value : candidate_values) {
    if (!value.empty() && value.back() == '\r') {
      value.pop_back();
    }
  }
  if (candidate_values.size() < 2) {
    std::cerr << "candidate file must contain at least two values\n";
    return 2;
  }
  absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> ranker =
      LlamaCandidateRanker::Create(manifest_path, model_directory);
  if (!ranker.ok()) {
    std::cerr << "backend creation failed: " << ranker.status() << '\n';
    return 1;
  }
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  WriteLlamaCandidateRankerEvaluationIdentity(
      std::cout, (*ranker)->EvaluationIdentity());
  WriteLlamaCandidateRankerEvaluationNumericProfile(
      std::cout, LlamaCandidateRankerEvaluationProfile::kConfigured);
  WriteLlamaCandidateRankerEvaluationNumericProfile(
      std::cout,
      LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled);
  converter::CandidateRankerRequest request;
  request.token = converter::CandidateRankerToken{
      .session_generation = 1,
      .state_revision = 1,
      .request_sequence = 1,
  };
  request.mode = *mode;
  request.preceding_text = absl::GetFlag(FLAGS_preceding_text);
  request.following_text = absl::GetFlag(FLAGS_following_text);
  request.reading = reading;
  request.focused_segment_id = 1;
  converter::CandidateRankerSegment& segment = request.segments.emplace_back();
  segment.id = 1;
  segment.key = segment_key;
  for (size_t i = 0; i < candidate_values.size(); ++i) {
    segment.candidates.push_back(converter::CandidateRankerCandidate{
        .id = static_cast<uint64_t>(i + 1),
        .key = segment_key,
        .value = candidate_values[i],
    });
  }

  NeverCancelled cancellation;
  const auto shared_configured = Score(
      ranker->get(), request, LlamaCandidateRankerEvaluationLayout::kSharedTrie,
      LlamaCandidateRankerEvaluationProfile::kConfigured, cancellation);
  const auto independent_configured =
      Score(ranker->get(), request,
            LlamaCandidateRankerEvaluationLayout::
                kIndependentCompleteSequences,
            LlamaCandidateRankerEvaluationProfile::kConfigured, cancellation);
  const auto shared_reference = Score(
      ranker->get(), request, LlamaCandidateRankerEvaluationLayout::kSharedTrie,
      LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled,
      cancellation);
  const auto independent_reference =
      Score(ranker->get(), request,
            LlamaCandidateRankerEvaluationLayout::
                kIndependentCompleteSequences,
            LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled,
            cancellation);
  if (!shared_configured.ok() || !independent_configured.ok() ||
      !shared_reference.ok() || !independent_reference.ok()) {
    std::cerr << "score run failed\n";
    if (!shared_configured.ok()) {
      std::cerr << "shared_configured: " << shared_configured.status() << '\n';
    }
    if (!independent_configured.ok()) {
      std::cerr << "independent_configured: "
                << independent_configured.status() << '\n';
    }
    if (!shared_reference.ok()) {
      std::cerr << "shared_reference: " << shared_reference.status() << '\n';
    }
    if (!independent_reference.ok()) {
      std::cerr << "independent_reference: " << independent_reference.status()
                << '\n';
    }
    return 1;
  }
  const absl::StatusOr<converter::CandidateRankerResponse>
      production_response = (*ranker)->Rank(request, cancellation);
  if (!production_response.ok()) {
    std::cerr << "production ranking failed: " << production_response.status()
              << '\n';
    return 1;
  }

  PrintResult("shared_configured", *shared_configured);
  PrintResult("independent_configured", *independent_configured);
  PrintResult("shared_reference", *shared_reference);
  PrintResult("independent_reference", *independent_reference);
  bool passed = CompareSharedAndIndependent(
      "configured", *shared_configured, *independent_configured, tolerance);
  passed = CompareProductionResponse(*production_response,
                                     shared_configured->response) &&
           passed;
  passed = CompareSharedAndIndependent("reference", *shared_reference,
                                       *independent_reference, tolerance) &&
           passed;

  for (size_t rotation = 1; rotation < candidate_values.size(); ++rotation) {
    converter::CandidateRankerRequest permuted_request = request;
    std::rotate(permuted_request.segments.front().candidates.begin(),
                permuted_request.segments.front().candidates.begin() +
                    rotation,
                permuted_request.segments.front().candidates.end());
    const auto permuted_shared = Score(
        ranker->get(), permuted_request,
        LlamaCandidateRankerEvaluationLayout::kSharedTrie,
        LlamaCandidateRankerEvaluationProfile::kConfigured, cancellation);
    if (!permuted_shared.ok()) {
      std::cerr << "permuted shared score failed: "
                << permuted_shared.status() << '\n';
      return 1;
    }
    passed = ComparePermutation(std::to_string(rotation), *shared_configured,
                                *permuted_shared) &&
             passed;
  }

  if (absl::GetFlag(FLAGS_check_context_sensitivity)) {
    converter::CandidateRankerRequest alternate_request = request;
    alternate_request.preceding_text =
        absl::GetFlag(FLAGS_alternate_preceding_text);
    alternate_request.following_text =
        absl::GetFlag(FLAGS_alternate_following_text);
    const auto alternate_reference =
        Score(ranker->get(), alternate_request,
              LlamaCandidateRankerEvaluationLayout::
                  kIndependentCompleteSequences,
              LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled,
              cancellation);
    if (!alternate_reference.ok()) {
      std::cerr << "alternate reference score failed: "
                << alternate_reference.status() << '\n';
      return 1;
    }
    PrintResult("independent_reference_alternate", *alternate_reference);
    passed = CompareContextMargins(*independent_reference,
                                   *alternate_reference, tolerance) &&
             passed;
  }
  std::cout << "probe\t" << static_cast<int>(passed) << '\n';
  return passed ? 0 : 1;
}

}  // namespace
}  // namespace engine
}  // namespace mozc

namespace {

int MainUtf8(int argc, char** argv) {
  mozc::InitMozc(argv[0], &argc, &argv);
  return mozc::engine::Run();
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
