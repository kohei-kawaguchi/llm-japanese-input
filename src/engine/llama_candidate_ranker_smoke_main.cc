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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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
ABSL_FLAG(std::string, reading, "", "Complete composition reading");
ABSL_FLAG(std::string, segment_key, "", "Focused segment reading");
ABSL_FLAG(std::string, preceding_text, "", "Text before the composition");
ABSL_FLAG(std::string, following_text, "", "Text after the composition");
ABSL_FLAG(uint32_t, repeat, 1, "Number of deterministic warm ranking runs");

namespace mozc {
namespace engine {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

int Run() {
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest_path);
  const std::string model_directory = absl::GetFlag(FLAGS_model_directory);
  const std::string candidates_path = absl::GetFlag(FLAGS_candidates_path);
  const std::string reading = absl::GetFlag(FLAGS_reading);
  const std::string segment_key = absl::GetFlag(FLAGS_segment_key);
  const uint32_t repeat = absl::GetFlag(FLAGS_repeat);
  if (manifest_path.empty() || model_directory.empty() ||
      candidates_path.empty() || reading.empty() || segment_key.empty() ||
      repeat == 0) {
    std::cerr << "manifest_path, model_directory, candidates_path, reading, "
                 "segment_key, and positive repeat are required\n";
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
  const absl::Time load_start = absl::Now();
  absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> ranker =
      LlamaCandidateRanker::Create(manifest_path, model_directory);
  if (!ranker.ok()) {
    std::cerr << "backend creation failed with status code "
              << static_cast<int>(ranker.status().code()) << '\n';
    return 1;
  }
  WriteLlamaCandidateRankerEvaluationIdentity(
      std::cout, (*ranker)->EvaluationIdentity());
  WriteLlamaCandidateRankerEvaluationNumericProfile(
      std::cout, LlamaCandidateRankerEvaluationProfile::kConfigured);
  std::cout << "load_ms\t"
            << absl::ToDoubleMilliseconds(absl::Now() - load_start) << '\n';

  converter::CandidateRankerRequest request;
  request.token = converter::CandidateRankerToken{
      .session_generation = 1,
      .state_revision = 1,
      .request_sequence = 1,
  };
  request.mode = converter::CandidateRankerMode::kConversion;
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
  std::vector<converter::CandidateRankerCandidateId> first_order;
  for (uint32_t iteration = 0; iteration < repeat; ++iteration) {
    const absl::Time rank_start = absl::Now();
    absl::StatusOr<converter::CandidateRankerResponse> response =
        (*ranker)->Rank(request, cancellation);
    const double elapsed_millis =
        absl::ToDoubleMilliseconds(absl::Now() - rank_start);
    if (!response.ok()) {
      std::cerr << "ranking failed with status code "
                << static_cast<int>(response.status().code()) << '\n';
      return 1;
    }
    if (response->segment_orders.size() != 1) {
      std::cerr << "ranking returned an unexpected segment count\n";
      return 1;
    }
    const std::vector<converter::CandidateRankerCandidateId>& order =
        response->segment_orders.front().candidate_ids;
    if (iteration == 0) {
      first_order = order;
    } else if (order != first_order) {
      std::cerr << "ranking order changed between identical warm runs\n";
      return 1;
    }
    std::cout << "rank_ms\t" << iteration << '\t' << elapsed_millis << '\n';
  }

  for (size_t position = 0; position < first_order.size(); ++position) {
    const uint64_t candidate_id = first_order[position];
    if (candidate_id == 0 || candidate_id > candidate_values.size()) {
      std::cerr << "ranking returned an invalid candidate identifier\n";
      return 1;
    }
    std::cout << "candidate\t" << position << '\t' << candidate_id << '\t'
              << candidate_values[candidate_id - 1] << '\n';
  }
  return 0;
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
