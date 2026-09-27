// Copyright 2026 LLM Japanese Input Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the distribution.
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
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

namespace {

void DiscardRuntimeLog(ggml_log_level, const char*, void*) {}

struct ModelDeleter {
  void operator()(llama_model* model) const { llama_model_free(model); }
};

struct ContextDeleter {
  void operator()(llama_context* context) const { llama_free(context); }
};

using ModelPointer = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPointer = std::unique_ptr<llama_context, ContextDeleter>;

class Batch final {
 public:
  explicit Batch(int32_t token_capacity)
      : batch_(llama_batch_init(token_capacity, 0, 1)) {}
  ~Batch() { llama_batch_free(batch_); }

  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  llama_batch* get() { return &batch_; }

 private:
  llama_batch batch_;
};

int Fail(int code, const char* message) {
  std::cerr << message << '\n';
  return code;
}

template <typename T>
bool WriteValue(std::ofstream* output, const T& value) {
  output->write(reinterpret_cast<const char*>(&value), sizeof(value));
  return output->good();
}

int Run(const char* model_path, const char* record_path,
        const char* output_path) {
  std::ifstream input(record_path, std::ios::binary);
  if (!input) {
    return Fail(3, "record file is unavailable");
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Fail(4, "record is too large");
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.check_tensors = true;
  ModelPointer model(llama_model_load_from_file(model_path, model_params));
  if (model == nullptr) {
    return Fail(5, "model could not be loaded");
  }
  const llama_vocab* vocabulary = llama_model_get_vocab(model.get());
  if (vocabulary == nullptr) {
    return Fail(6, "model metadata is invalid");
  }
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocabulary);
  const llama_token bos = llama_vocab_bos(vocabulary);
  const int32_t model_context = llama_model_n_ctx_train(model.get());
  if (vocabulary_size <= 0 || bos < 0 || model_context <= 0) {
    return Fail(6, "model metadata is invalid");
  }

  const int32_t required =
      llama_tokenize(vocabulary, text.data(), static_cast<int32_t>(text.size()),
                     nullptr, 0, false, false);
  if (required == std::numeric_limits<int32_t>::min()) {
    return Fail(7, "token count overflowed");
  }
  const int64_t record_token_count =
      required < 0 ? -static_cast<int64_t>(required) : required;
  if (record_token_count > model_context - 1) {
    return Fail(8, "record exceeds model context");
  }
  std::vector<llama_token> record_tokens(
      static_cast<size_t>(record_token_count));
  if (!record_tokens.empty()) {
    const int32_t actual = llama_tokenize(
        vocabulary, text.data(), static_cast<int32_t>(text.size()),
        record_tokens.data(), static_cast<int32_t>(record_tokens.size()), false,
        false);
    if (actual != static_cast<int32_t>(record_tokens.size())) {
      return Fail(9, "tokenizer returned an inconsistent result");
    }
  }
  std::vector<llama_token> tokens;
  tokens.reserve(record_tokens.size() + 1);
  tokens.push_back(bos);
  tokens.insert(tokens.end(), record_tokens.begin(), record_tokens.end());

  std::vector<uint32_t> output_positions = {
      0,
      static_cast<uint32_t>(tokens.size() / 2),
      static_cast<uint32_t>(tokens.size() - 1),
  };
  std::sort(output_positions.begin(), output_positions.end());
  output_positions.erase(
      std::unique(output_positions.begin(), output_positions.end()),
      output_positions.end());

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(model_context);
  context_params.n_batch = static_cast<uint32_t>(model_context);
  context_params.n_ubatch =
      std::min<uint32_t>(256, static_cast<uint32_t>(model_context));
  context_params.n_seq_max = 1;
  context_params.n_outputs_max = static_cast<uint32_t>(output_positions.size());
  context_params.n_outputs_max_per_seq = 0;
  context_params.n_threads = 1;
  context_params.n_threads_batch = 1;
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.type_k = GGML_TYPE_F32;
  context_params.type_v = GGML_TYPE_F32;
  context_params.offload_kqv = false;
  context_params.op_offload = false;
  context_params.kv_unified = true;
  ContextPointer context(llama_init_from_model(model.get(), context_params));
  if (context == nullptr) {
    return Fail(10, "context could not be initialized");
  }

  Batch batch(static_cast<int32_t>(tokens.size()));
  llama_batch* batch_data = batch.get();
  batch_data->n_tokens = static_cast<int32_t>(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    batch_data->token[i] = tokens[i];
    batch_data->pos[i] = static_cast<llama_pos>(i);
    batch_data->n_seq_id[i] = 1;
    batch_data->seq_id[i][0] = 0;
    batch_data->logits[i] =
        std::binary_search(output_positions.begin(), output_positions.end(),
                           static_cast<uint32_t>(i));
  }
  if (llama_decode(context.get(), *batch_data) != 0) {
    return Fail(11, "decode failed");
  }

  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Fail(12, "output file is unavailable");
  }
  constexpr std::array<char, 8> kMagic = {'L', 'L', 'M', 'J',
                                          'L', 'O', 'G', '1'};
  output.write(kMagic.data(), kMagic.size());
  const uint32_t token_count = static_cast<uint32_t>(tokens.size());
  const uint32_t row_count = static_cast<uint32_t>(output_positions.size());
  const uint32_t vocab_count = static_cast<uint32_t>(vocabulary_size);
  if (!WriteValue(&output, token_count) || !WriteValue(&output, vocab_count) ||
      !WriteValue(&output, row_count)) {
    return Fail(13, "output header could not be written");
  }
  for (const llama_token token : tokens) {
    const int32_t value = static_cast<int32_t>(token);
    if (!WriteValue(&output, value)) {
      return Fail(14, "token IDs could not be written");
    }
  }
  for (const uint32_t position : output_positions) {
    if (!WriteValue(&output, position)) {
      return Fail(15, "output positions could not be written");
    }
  }
  for (const uint32_t position : output_positions) {
    const float* logits =
        llama_get_logits_ith(context.get(), static_cast<int32_t>(position));
    if (logits == nullptr) {
      return Fail(16, "logits are unavailable");
    }
    output.write(reinterpret_cast<const char*>(logits),
                 static_cast<std::streamsize>(vocabulary_size) * sizeof(float));
    if (!output.good()) {
      return Fail(17, "logits could not be written");
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    return Fail(2, "usage: llama_cpp_logits_probe MODEL RECORD OUTPUT");
  }
  llama_log_set(&DiscardRuntimeLog, nullptr);
  ggml_log_set(&DiscardRuntimeLog, nullptr);
  llama_backend_init();
  const int result = Run(argv[1], argv[2], argv[3]);
  llama_backend_free();
  return result;
}
