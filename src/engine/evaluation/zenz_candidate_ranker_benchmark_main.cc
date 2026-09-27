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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"
#include "engine/evaluation/quality_regression_frozen_corpus.pb.h"
#include "engine/zenz_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, model_path, "", "Runtime zenz GGUF path");
ABSL_FLAG(std::string, corpus_path, "", "Frozen corpus binary path");
ABSL_FLAG(std::string, corpus_kind, "",
          "quality_regression or ajimee frozen corpus");
ABSL_FLAG(std::string, output_path, "", "JSON lines output path");
ABSL_FLAG(int32_t, top_k, 5, "Candidates scored per segment");
ABSL_FLAG(std::string, right_window, "next", "full, next, or none");
ABSL_FLAG(double, copy_penalty, 100.0, "Reading copy penalty");
ABSL_FLAG(double, order_prior, 1.0, "Mozc order prior per index");
ABSL_FLAG(int32_t, thread_count, 8, "Decode threads");
ABSL_FLAG(int32_t, context_size, 512, "Context and batch token capacity");

namespace mozc {
namespace engine {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

struct BenchmarkCase {
  uint64_t source = 0;
  const evaluation::FrozenCandidateRankerRequest* request = nullptr;
};

absl::StatusOr<ZenzRightWindow> ParseRightWindow(absl::string_view value) {
  if (value == "full") {
    return ZenzRightWindow::kFull;
  }
  if (value == "next") {
    return ZenzRightWindow::kNext;
  }
  if (value == "none") {
    return ZenzRightWindow::kNone;
  }
  return absl::InvalidArgumentError("right_window is unknown");
}

std::string MergedTopOutput(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerMergeOrder& merge_order) {
  std::string output;
  for (size_t i = 0; i < request.segments.size(); ++i) {
    const converter::CandidateRankerCandidateId top =
        merge_order.segment_orders[i].candidate_ids.front();
    for (const converter::CandidateRankerCandidate& candidate :
         request.segments[i].candidates) {
      if (candidate.id == top) {
        output += candidate.value;
      }
    }
  }
  return output;
}

int Run() {
  const absl::StatusOr<ZenzRightWindow> right_window =
      ParseRightWindow(absl::GetFlag(FLAGS_right_window));
  if (!right_window.ok()) {
    std::cerr << right_window.status() << '\n';
    return 2;
  }
  const absl::StatusOr<std::string> contents =
      FileUtil::GetContents(absl::GetFlag(FLAGS_corpus_path));
  if (!contents.ok()) {
    std::cerr << contents.status() << '\n';
    return 2;
  }
  evaluation::QualityRegressionFrozenCorpus quality_regression;
  evaluation::AjimeeFrozenCorpus ajimee;
  std::vector<BenchmarkCase> cases;
  const std::string corpus_kind = absl::GetFlag(FLAGS_corpus_kind);
  if (corpus_kind == "quality_regression") {
    if (!quality_regression.ParseFromString(*contents)) {
      std::cerr << "quality regression corpus is invalid\n";
      return 2;
    }
    for (const auto& item : quality_regression.cases()) {
      cases.push_back({.source = item.source_line(), .request = &item.request()});
    }
  } else if (corpus_kind == "ajimee") {
    if (!ajimee.ParseFromString(*contents)) {
      std::cerr << "AJIMEE corpus is invalid\n";
      return 2;
    }
    for (const auto& item : ajimee.cases()) {
      cases.push_back({.source = item.source_index(), .request = &item.request()});
    }
  } else {
    std::cerr << "corpus_kind must be quality_regression or ajimee\n";
    return 2;
  }

  const absl::Time load_start = absl::Now();
  absl::StatusOr<std::shared_ptr<ZenzCandidateRanker>> ranker =
      ZenzCandidateRanker::Create(ZenzCandidateRankerOptions{
          .model_path = absl::GetFlag(FLAGS_model_path),
          .top_k = absl::GetFlag(FLAGS_top_k),
          .right_window = *right_window,
          .copy_penalty = absl::GetFlag(FLAGS_copy_penalty),
          .order_prior = absl::GetFlag(FLAGS_order_prior),
          .thread_count = absl::GetFlag(FLAGS_thread_count),
          .context_size = absl::GetFlag(FLAGS_context_size),
      });
  if (!ranker.ok()) {
    std::cerr << ranker.status() << '\n';
    return 1;
  }
  std::cerr << "load_ms " << absl::ToDoubleMilliseconds(absl::Now() - load_start)
            << '\n';

  std::ofstream output(absl::GetFlag(FLAGS_output_path), std::ios::binary);
  if (!output.is_open()) {
    std::cerr << "output_path cannot be opened\n";
    return 2;
  }
  NeverCancelled cancellation;
  for (const BenchmarkCase& item : cases) {
    const absl::StatusOr<converter::CandidateRankerRequest> request =
        evaluation::ConvertFrozenCandidateRankerRequest(*item.request);
    if (!request.ok()) {
      std::cerr << request.status() << '\n';
      return 1;
    }
    const absl::Time started = absl::Now();
    const absl::StatusOr<converter::CandidateRankerResponse> response =
        (*ranker)->Rank(*request, cancellation);
    const double milliseconds =
        absl::ToDoubleMilliseconds(absl::Now() - started);
    if (!response.ok()) {
      std::cerr << item.source << ' ' << response.status() << '\n';
      return 1;
    }
    const absl::StatusOr<converter::CandidateRankerMergeOrder> merge_order =
        converter::BuildCandidateRankerMergeOrder(*request, *response,
                                                  request->token);
    if (!merge_order.ok()) {
      std::cerr << item.source << ' ' << merge_order.status() << '\n';
      return 1;
    }
    output << "{\"source\": " << item.source << ", \"output\": \""
           << absl::Utf8SafeCEscape(MergedTopOutput(*request, *merge_order))
           << "\", \"milliseconds\": " << milliseconds << "}\n";
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
