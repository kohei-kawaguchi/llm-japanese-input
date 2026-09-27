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

#include "engine/zenz_candidate_ranker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/strings/japanese.h"
#include "base/system_util.h"
#include "converter/candidate_ranker.h"
#include "ggml.h"
#include "llama.h"

namespace mozc {
namespace engine {
namespace {

constexpr absl::string_view kLeftContextTag = "\uEE02";
constexpr absl::string_view kRightContextTag = "\uEE07";
constexpr absl::string_view kInputTag = "\uEE00";
constexpr absl::string_view kOutputTag = "\uEE01";
constexpr absl::string_view kEndTokenPiece = "</s>";
constexpr llama_seq_id kPrefixSequence = 0;

void DiscardRuntimeLog(ggml_log_level, const char*, void*) {}

class LlamaProcessLifetime final {
 public:
  LlamaProcessLifetime() {
    llama_log_set(&DiscardRuntimeLog, nullptr);
    ggml_log_set(&DiscardRuntimeLog, nullptr);
    llama_backend_init();
  }

  ~LlamaProcessLifetime() { llama_backend_free(); }
};

void EnsureLlamaProcessLifetime() {
  static const LlamaProcessLifetime lifetime;
  static_cast<void>(lifetime);
}

struct ModelDeleter {
  void operator()(llama_model* model) const { llama_model_free(model); }
};

struct ContextDeleter {
  void operator()(llama_context* context) const { llama_free(context); }
};

class Batch final {
 public:
  Batch(int32_t token_capacity, int32_t sequence_capacity)
      : batch_(llama_batch_init(token_capacity, 0, sequence_capacity)) {}
  ~Batch() { llama_batch_free(batch_); }

  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  void Clear() { batch_.n_tokens = 0; }

  int32_t size() const { return batch_.n_tokens; }

  void Add(llama_token token, llama_pos position, llama_seq_id sequence,
           bool logits) {
    const int32_t index = batch_.n_tokens;
    batch_.token[index] = token;
    batch_.pos[index] = position;
    batch_.n_seq_id[index] = 1;
    batch_.seq_id[index][0] = sequence;
    batch_.logits[index] = logits;
    ++batch_.n_tokens;
  }

  llama_batch get() const { return batch_; }

 private:
  llama_batch batch_;
};

std::string NormalizeForModel(absl::string_view text) {
  return absl::StrReplaceAll(text, {{" ", "\u3000"}, {"\n", ""}});
}

std::vector<const converter::CandidateRankerCandidate*> SelectedCandidates(
    const converter::CandidateRankerSegment& segment, int32_t top_k) {
  std::vector<const converter::CandidateRankerCandidate*> selected;
  for (const converter::CandidateRankerCandidate& candidate :
       segment.candidates) {
    if (static_cast<int32_t>(selected.size()) == top_k) {
      break;
    }
    if (!candidate.is_protected) {
      selected.push_back(&candidate);
    }
  }
  return selected;
}

std::pair<std::string, bool> RightWindow(
    const std::vector<std::string>& baseline_values, size_t index,
    ZenzRightWindow right_window) {
  const bool last = index + 1 == baseline_values.size();
  switch (right_window) {
    case ZenzRightWindow::kFull: {
      std::string suffix;
      for (size_t i = index + 1; i < baseline_values.size(); ++i) {
        suffix += baseline_values[i];
      }
      return {suffix, true};
    }
    case ZenzRightWindow::kNext:
      return last ? std::make_pair(std::string(), true)
                  : std::make_pair(baseline_values[index + 1], false);
    case ZenzRightWindow::kNone:
      return {std::string(), false};
  }
  return {std::string(), false};
}

absl::StatusOr<std::vector<llama_token>> Tokenize(const llama_vocab* vocab,
                                                  absl::string_view text,
                                                  bool parse_special) {
  std::vector<llama_token> tokens(text.size() + 1);
  const int32_t count = llama_tokenize(
      vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
      static_cast<int32_t>(tokens.size()), false, parse_special);
  if (count < 0) {
    return absl::InternalError("zenz tokenization exceeded its capacity");
  }
  tokens.resize(count);
  return tokens;
}

double LogProbability(const float* logits, int32_t vocabulary_size,
                      llama_token token) {
  const float maximum = *std::max_element(logits, logits + vocabulary_size);
  double sum = 0.0;
  for (int32_t i = 0; i < vocabulary_size; ++i) {
    sum += std::exp(static_cast<double>(logits[i]) - maximum);
  }
  return static_cast<double>(logits[token]) - maximum - std::log(sum);
}

std::vector<float> CopyLastLogits(llama_context* context,
                                  int32_t vocabulary_size) {
  const float* logits = llama_get_logits_ith(context, -1);
  return std::vector<float>(logits, logits + vocabulary_size);
}

}  // namespace

ZenzCandidateRankerOptions InstalledZenzCandidateRankerOptions(
    std::string model_path) {
  return ZenzCandidateRankerOptions{
      .model_path = std::move(model_path),
      .top_k = 5,
      .right_window = ZenzRightWindow::kNext,
      .copy_penalty = 100.0,
      .order_prior = 1.0,
      .thread_count = 8,
      .context_size = 512,
  };
}

absl::StatusOr<std::shared_ptr<converter::CandidateRankerBackendInterface>>
CreateInstalledZenzCandidateRanker() {
  const std::string model_path = FileUtil::JoinPath(
      SystemUtil::GetServerDirectory(), kInstalledZenzModelFileName);
  if (!FileUtil::FileExists(model_path).ok()) {
    return nullptr;
  }
  absl::StatusOr<std::shared_ptr<ZenzCandidateRanker>> ranker =
      ZenzCandidateRanker::Create(
          InstalledZenzCandidateRankerOptions(model_path));
  if (!ranker.ok()) {
    return ranker.status();
  }
  return std::shared_ptr<converter::CandidateRankerBackendInterface>(
      *std::move(ranker));
}

bool IsZenzRankedMode(converter::CandidateRankerMode mode) {
  return mode == converter::CandidateRankerMode::kConversion;
}

std::string BuildZenzPrompt(const converter::CandidateRankerRequest& request) {
  return NormalizeForModel(absl::StrCat(
      kLeftContextTag, request.preceding_text, kRightContextTag,
      request.following_text, kInputTag,
      japanese::HiraganaToKatakana(request.reading), kOutputTag));
}

std::vector<ZenzScoredOutput> BuildZenzScoredOutputs(
    const converter::CandidateRankerRequest& request, int32_t top_k,
    ZenzRightWindow right_window) {
  std::vector<std::string> baseline_values;
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    baseline_values.push_back(segment.candidates.front().value);
  }
  std::vector<ZenzScoredOutput> outputs;
  for (size_t index = 0; index < request.segments.size(); ++index) {
    const converter::CandidateRankerSegment& segment = request.segments[index];
    std::string prefix;
    for (size_t i = 0; i < index; ++i) {
      prefix += baseline_values[i];
    }
    const auto [suffix, add_end] =
        RightWindow(baseline_values, index, right_window);
    for (const converter::CandidateRankerCandidate* candidate :
         SelectedCandidates(segment, top_k)) {
      outputs.push_back(ZenzScoredOutput{
          .segment_id = segment.id,
          .candidate_id = candidate->id,
          .prefix = NormalizeForModel(prefix),
          .body = NormalizeForModel(absl::StrCat(candidate->value, suffix)),
          .add_end = add_end,
      });
    }
  }
  return outputs;
}

converter::CandidateRankerSegmentOrder OrderZenzSegment(
    const converter::CandidateRankerSegment& segment,
    const std::vector<double>& scores, double copy_penalty,
    double order_prior) {
  const std::vector<const converter::CandidateRankerCandidate*> selected =
      SelectedCandidates(segment, static_cast<int32_t>(scores.size()));
  const std::string katakana_key = japanese::HiraganaToKatakana(segment.key);
  const auto is_copy = [&](absl::string_view value) {
    return value == segment.key || value == katakana_key;
  };
  const bool penalize_copies = !is_copy(segment.candidates.front().value);
  std::vector<std::tuple<double, size_t, converter::CandidateRankerCandidateId>>
      calibrated;
  for (size_t index = 0; index < selected.size(); ++index) {
    double score = scores[index] - order_prior * static_cast<double>(index);
    if (penalize_copies && is_copy(selected[index]->value)) {
      score -= copy_penalty;
    }
    calibrated.emplace_back(-score, index, selected[index]->id);
  }
  std::sort(calibrated.begin(), calibrated.end());
  converter::CandidateRankerSegmentOrder order{.segment_id = segment.id};
  for (const auto& item : calibrated) {
    order.candidate_ids.push_back(std::get<2>(item));
  }
  return order;
}

struct ZenzCandidateRanker::RuntimeState {
  std::unique_ptr<llama_model, ModelDeleter> model;
  std::unique_ptr<llama_context, ContextDeleter> context;
  const llama_vocab* vocab = nullptr;
  llama_token end_token = 0;
  int32_t vocabulary_size = 0;
};

absl::StatusOr<std::shared_ptr<ZenzCandidateRanker>>
ZenzCandidateRanker::Create(const ZenzCandidateRankerOptions& options) {
  if (options.top_k <= 0 || options.thread_count <= 0 ||
      options.context_size <= 0) {
    return absl::InvalidArgumentError("zenz ranker options are incomplete");
  }
  EnsureLlamaProcessLifetime();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto runtime = std::make_unique<RuntimeState>();
  runtime->model.reset(
      llama_model_load_from_file(options.model_path.c_str(), model_params));
  if (runtime->model == nullptr) {
    return absl::NotFoundError("zenz model could not be loaded");
  }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = options.context_size;
  context_params.n_batch = options.context_size;
  context_params.n_ubatch = options.context_size;
  context_params.n_seq_max = options.top_k + 1;
  context_params.n_threads = options.thread_count;
  context_params.n_threads_batch = options.thread_count;
  context_params.kv_unified = true;
  context_params.offload_kqv = false;
  context_params.op_offload = false;
  runtime->context.reset(
      llama_init_from_model(runtime->model.get(), context_params));
  if (runtime->context == nullptr) {
    return absl::InternalError("zenz context could not be created");
  }
  runtime->vocab = llama_model_get_vocab(runtime->model.get());
  runtime->vocabulary_size = llama_vocab_n_tokens(runtime->vocab);
  absl::StatusOr<std::vector<llama_token>> end_tokens =
      Tokenize(runtime->vocab, kEndTokenPiece, true);
  if (!end_tokens.ok()) {
    return end_tokens.status();
  }
  if (end_tokens->size() != 1) {
    return absl::InternalError("zenz end token piece is not a single token");
  }
  runtime->end_token = end_tokens->front();
  return std::shared_ptr<ZenzCandidateRanker>(
      new ZenzCandidateRanker(options, std::move(runtime)));
}

ZenzCandidateRanker::ZenzCandidateRanker(ZenzCandidateRankerOptions options,
                                         std::unique_ptr<RuntimeState> runtime)
    : options_(std::move(options)), runtime_(std::move(runtime)) {}

ZenzCandidateRanker::~ZenzCandidateRanker() = default;

absl::StatusOr<converter::CandidateRankerResponse> ZenzCandidateRanker::Rank(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerCancellation& cancellation) {
  if (!IsZenzRankedMode(request.mode)) {
    return converter::CandidateRankerResponse{.token = request.token};
  }
  llama_context* context = runtime_->context.get();
  llama_memory_t memory = llama_get_memory(context);
  const int32_t vocabulary_size = runtime_->vocabulary_size;
  llama_memory_clear(memory, true);
  Batch batch(options_.context_size, options_.top_k + 1);

  absl::StatusOr<std::vector<llama_token>> prompt_tokens =
      Tokenize(runtime_->vocab, BuildZenzPrompt(request), false);
  if (!prompt_tokens.ok()) {
    return prompt_tokens.status();
  }
  const llama_pos prompt_length =
      static_cast<llama_pos>(prompt_tokens->size());
  if (prompt_length >= options_.context_size) {
    return absl::ResourceExhaustedError("zenz prompt exceeds the context");
  }
  for (llama_pos i = 0; i < prompt_length; ++i) {
    batch.Add((*prompt_tokens)[i], i, kPrefixSequence, i + 1 == prompt_length);
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("zenz ranking was cancelled");
  }
  if (llama_decode(context, batch.get()) != 0) {
    return absl::InternalError("zenz prompt decode failed");
  }
  const std::vector<float> prompt_logits =
      CopyLastLogits(context, vocabulary_size);

  const std::vector<ZenzScoredOutput> outputs =
      BuildZenzScoredOutputs(request, options_.top_k, options_.right_window);
  converter::CandidateRankerResponse response{.token = request.token};
  size_t next_output = 0;
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    std::vector<const ZenzScoredOutput*> segment_outputs;
    while (next_output < outputs.size() &&
           outputs[next_output].segment_id == segment.id) {
      segment_outputs.push_back(&outputs[next_output]);
      ++next_output;
    }
    if (segment_outputs.empty()) {
      continue;
    }
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("zenz ranking was cancelled");
    }
    llama_memory_seq_rm(memory, -1, prompt_length, -1);
    absl::StatusOr<std::vector<llama_token>> prefix_tokens =
        Tokenize(runtime_->vocab, segment_outputs.front()->prefix, false);
    if (!prefix_tokens.ok()) {
      return prefix_tokens.status();
    }
    const llama_pos start =
        prompt_length + static_cast<llama_pos>(prefix_tokens->size());
    if (start >= options_.context_size) {
      return absl::ResourceExhaustedError("zenz prefix exceeds the context");
    }
    std::vector<float> start_logits = prompt_logits;
    if (!prefix_tokens->empty()) {
      batch.Clear();
      for (size_t i = 0; i < prefix_tokens->size(); ++i) {
        batch.Add((*prefix_tokens)[i],
                  prompt_length + static_cast<llama_pos>(i), kPrefixSequence,
                  i + 1 == prefix_tokens->size());
      }
      if (llama_decode(context, batch.get()) != 0) {
        return absl::InternalError("zenz prefix decode failed");
      }
      start_logits = CopyLastLogits(context, vocabulary_size);
    }

    std::vector<std::vector<llama_token>> bodies;
    batch.Clear();
    for (size_t i = 0; i < segment_outputs.size(); ++i) {
      absl::StatusOr<std::vector<llama_token>> body =
          Tokenize(runtime_->vocab, segment_outputs[i]->body, false);
      if (!body.ok()) {
        return body.status();
      }
      if (segment_outputs[i]->add_end) {
        body->push_back(runtime_->end_token);
      }
      if (start + static_cast<llama_pos>(body->size()) >
              options_.context_size ||
          batch.size() + static_cast<int32_t>(body->size()) >
              options_.context_size) {
        return absl::ResourceExhaustedError(
            "zenz candidate batch exceeds the context");
      }
      const llama_seq_id sequence = static_cast<llama_seq_id>(i + 1);
      llama_memory_seq_cp(memory, kPrefixSequence, sequence, -1, -1);
      for (size_t j = 0; j < body->size(); ++j) {
        batch.Add((*body)[j], start + static_cast<llama_pos>(j), sequence,
                  true);
      }
      bodies.push_back(*std::move(body));
    }
    if (llama_decode(context, batch.get()) != 0) {
      return absl::InternalError("zenz candidate decode failed");
    }

    std::vector<double> scores;
    int32_t row = 0;
    for (const std::vector<llama_token>& body : bodies) {
      double score = LogProbability(start_logits.data(), vocabulary_size,
                                    body.front());
      for (size_t j = 1; j < body.size(); ++j) {
        score += LogProbability(
            llama_get_logits_ith(context, row + static_cast<int32_t>(j) - 1),
            vocabulary_size, body[j]);
      }
      scores.push_back(score);
      row += static_cast<int32_t>(body.size());
    }
    for (size_t i = 0; i < segment_outputs.size(); ++i) {
      llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i + 1), -1, -1);
    }
    response.segment_orders.push_back(OrderZenzSegment(
        segment, scores, options_.copy_penalty, options_.order_prior));
  }
  return response;
}

}  // namespace engine
}  // namespace mozc
