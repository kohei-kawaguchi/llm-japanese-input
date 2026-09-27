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

#include "engine/llama_candidate_ranker.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <share.h>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "base/file_util.h"
#include "base/protobuf/text_format.h"
#include "base/strings/pfchar.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/candidate_ranker_model.pb.h"
#include "ggml.h"
#include "gguf.h"
#include "google/protobuf/io/tokenizer.h"
#include "llama.h"

extern "C" {
#include "sha256/sha256.h"
}

namespace mozc {
namespace engine {
namespace {

constexpr size_t kHashReadBufferSize = 1024 * 1024;

class SilentTextFormatErrorCollector final
    : public protobuf::io::ErrorCollector {
 public:
  void RecordError(int, protobuf::io::ColumnNumber,
                   absl::string_view) override {}
};

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

struct FileCloser {
  void operator()(FILE* file) const { std::fclose(file); }
};

struct GgufDeleter {
  void operator()(gguf_context* context) const { gguf_free(context); }
};

using ModelPointer = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPointer = std::unique_ptr<llama_context, ContextDeleter>;
using FilePointer = std::unique_ptr<FILE, FileCloser>;
using GgufPointer = std::unique_ptr<gguf_context, GgufDeleter>;

class Batch final {
 public:
  Batch(int32_t token_capacity, int32_t sequence_capacity)
      : batch_(llama_batch_init(token_capacity, 0, sequence_capacity)) {}
  ~Batch() { llama_batch_free(batch_); }

  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  llama_batch* get() { return &batch_; }

 private:
  llama_batch batch_;
};

class ContextMemoryScope final {
 public:
  explicit ContextMemoryScope(llama_context* context)
      : memory_(llama_get_memory(context)) {
    llama_memory_clear(memory_, true);
  }
  ~ContextMemoryScope() { llama_memory_clear(memory_, true); }

  ContextMemoryScope(const ContextMemoryScope&) = delete;
  ContextMemoryScope& operator=(const ContextMemoryScope&) = delete;

 private:
  llama_memory_t memory_;
};

class AbortCallbackScope final {
 public:
  AbortCallbackScope(llama_context* context,
                     const converter::CandidateRankerCancellation& cancellation)
      : context_(context), cancellation_(&cancellation) {
    llama_set_abort_callback(context_, &AbortRequested, this);
  }
  ~AbortCallbackScope() {
    llama_set_abort_callback(context_, nullptr, nullptr);
  }

  AbortCallbackScope(const AbortCallbackScope&) = delete;
  AbortCallbackScope& operator=(const AbortCallbackScope&) = delete;

 private:
  static bool AbortRequested(void* data) {
    return static_cast<AbortCallbackScope*>(data)
        ->cancellation_->IsCancellationRequested();
  }

  llama_context* context_;
  const converter::CandidateRankerCancellation* cancellation_;
};

std::string DigestToHex(const unsigned char* digest, size_t size) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result(size * 2, '\0');
  for (size_t i = 0; i < size; ++i) {
    result[2 * i] = kHexDigits[digest[i] >> 4];
    result[2 * i + 1] = kHexDigits[digest[i] & 0x0f];
  }
  return result;
}

std::string ComputeBytesSha256(absl::string_view contents) {
  sha256_t hash;
  sha256_init(&hash);
  sha256_update(
      &hash, reinterpret_cast<const unsigned char*>(contents.data()),
      contents.size());
  std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
  sha256_final(&hash, digest.data());
  return DigestToHex(digest.data(), digest.size());
}

struct LoadedCandidateRankerManifest {
  CandidateRankerModelManifest manifest;
  LlamaCandidateRankerEvaluationIdentity evaluation_identity;
};

absl::Status ValidateExpectedEvaluationIdentity(
    const LlamaCandidateRankerEvaluationIdentity& actual,
    const LlamaCandidateRankerEvaluationIdentity& expected) {
  if (actual.manifest_sha256 != expected.manifest_sha256 ||
      actual.gguf_file_name != expected.gguf_file_name ||
      actual.gguf_sha256 != expected.gguf_sha256 ||
      actual.tokenizer_sha256 != expected.tokenizer_sha256 ||
      actual.quantization != expected.quantization ||
      actual.source_revision != expected.source_revision ||
      actual.runtime_revision != expected.runtime_revision) {
    return absl::FailedPreconditionError(
        "candidate ranking evaluation identity does not match the expected "
        "identity");
  }
  return absl::OkStatus();
}

absl::StatusOr<LoadedCandidateRankerManifest> LoadManifest(
    absl::string_view manifest_path) {
  const absl::StatusOr<std::string> contents =
      FileUtil::GetContents(manifest_path);
  if (!contents.ok()) {
    return absl::NotFoundError("candidate ranking manifest is unavailable");
  }

  CandidateRankerModelManifest manifest;
  SilentTextFormatErrorCollector error_collector;
  protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&error_collector);
  if (!parser.ParseFromString(*contents, &manifest)) {
    return absl::InvalidArgumentError(
        "candidate ranking manifest text is invalid");
  }
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  LlamaCandidateRankerEvaluationIdentity evaluation_identity =
      BuildLlamaCandidateRankerEvaluationIdentity(manifest, *contents);
  return LoadedCandidateRankerManifest{
      .manifest = std::move(manifest),
      .evaluation_identity = std::move(evaluation_identity),
  };
}

absl::StatusOr<FilePointer> OpenModelFile(absl::string_view path) {
  const pfstring native_path = to_pfstring(path);
#ifdef _WIN32
  FILE* file = _wfsopen(native_path.c_str(), L"rb", _SH_DENYWR);
#else
  FILE* file = std::fopen(native_path.c_str(), "rb");
#endif
  if (file == nullptr) {
    return absl::NotFoundError("candidate ranking model is unavailable");
  }
  return FilePointer(file);
}

absl::StatusOr<std::string> ComputeFileSha256(FILE* file) {
  sha256_t hash;
  sha256_init(&hash);
  std::vector<unsigned char> buffer(kHashReadBufferSize);
  for (;;) {
    const size_t bytes_read = std::fread(buffer.data(), 1, buffer.size(), file);
    if (bytes_read > 0) {
      sha256_update(&hash, buffer.data(), bytes_read);
    }
    if (bytes_read < buffer.size()) {
      if (std::ferror(file) != 0) {
        return absl::DataLossError("candidate ranking model could not be read");
      }
      break;
    }
  }

  std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
  sha256_final(&hash, digest.data());
  return DigestToHex(digest.data(), digest.size());
}

absl::Status RewindModelFile(FILE* file) {
  if (std::fseek(file, 0, SEEK_SET) != 0) {
    return absl::DataLossError("candidate ranking model could not be read");
  }
  return absl::OkStatus();
}

absl::StatusOr<uint32_t> ParseModelContextMetadata(FILE* file) {
  const gguf_init_params params{
      .no_alloc = true,
      .ctx = nullptr,
  };
  const GgufPointer metadata(gguf_init_from_file_ptr(file, params));
  if (metadata == nullptr) {
    return absl::InternalError(
        "candidate ranking GGUF metadata could not be read");
  }

  const int64_t architecture_index =
      gguf_find_key(metadata.get(), "general.architecture");
  if (architecture_index < 0 ||
      gguf_get_kv_type(metadata.get(), architecture_index) !=
          GGUF_TYPE_STRING) {
    return absl::InvalidArgumentError(
        "candidate ranking model architecture metadata is invalid");
  }
  const char* architecture =
      gguf_get_val_str(metadata.get(), architecture_index);
  if (architecture == nullptr || architecture[0] == '\0') {
    return absl::InvalidArgumentError(
        "candidate ranking model architecture metadata is invalid");
  }

  std::string context_key(architecture);
  context_key.append(".context_length");
  const int64_t context_index =
      gguf_find_key(metadata.get(), context_key.c_str());
  if (context_index < 0 ||
      gguf_get_kv_type(metadata.get(), context_index) != GGUF_TYPE_UINT32) {
    return absl::InvalidArgumentError(
        "candidate ranking model context metadata is invalid");
  }
  const uint32_t model_context =
      gguf_get_val_u32(metadata.get(), context_index);
  if (model_context == 0 ||
      model_context >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return absl::InvalidArgumentError(
        "candidate ranking model context metadata is invalid");
  }
  return model_context;
}

absl::StatusOr<uint32_t> ReadModelContextMetadata(FILE* file) {
  const absl::StatusOr<uint32_t> model_context =
      ParseModelContextMetadata(file);
  const absl::Status rewind_status = RewindModelFile(file);
  if (!rewind_status.ok()) {
    return rewind_status;
  }
  return model_context;
}

struct VerifiedCandidateRankerSource {
  LoadedCandidateRankerManifest loaded_manifest;
  FilePointer model_file;
  uint32_t model_context = 0;
};

absl::Status ValidateRuntimeCapacities(
    const CandidateRankerModelManifest& manifest);
absl::Status ValidateModelContextCapacity(
    const CandidateRankerModelManifest& manifest, uint32_t model_context);

absl::StatusOr<VerifiedCandidateRankerSource>
OpenVerifiedCandidateRankerModel(
    LoadedCandidateRankerManifest loaded_manifest,
    absl::string_view model_directory) {
  const std::string model_path = FileUtil::JoinPath(
      model_directory,
      loaded_manifest.manifest.artifacts().gguf_file_name());
  absl::StatusOr<FilePointer> model_file = OpenModelFile(model_path);
  if (!model_file.ok()) {
    return model_file.status();
  }
  const absl::StatusOr<std::string> model_sha256 =
      ComputeFileSha256(model_file->get());
  if (!model_sha256.ok()) {
    return model_sha256.status();
  }
  if (*model_sha256 !=
      loaded_manifest.manifest.artifacts().gguf_sha256()) {
    return absl::DataLossError(
        "candidate ranking model hash does not match the manifest");
  }
  const absl::Status rewind_status = RewindModelFile(model_file->get());
  if (!rewind_status.ok()) {
    return rewind_status;
  }
  const absl::StatusOr<uint32_t> model_context =
      ReadModelContextMetadata(model_file->get());
  if (!model_context.ok()) {
    return model_context.status();
  }
  const absl::Status context_capacity_status = ValidateModelContextCapacity(
      loaded_manifest.manifest, *model_context);
  if (!context_capacity_status.ok()) {
    return context_capacity_status;
  }
  return VerifiedCandidateRankerSource{
      .loaded_manifest = std::move(loaded_manifest),
      .model_file = std::move(*model_file),
      .model_context = *model_context,
  };
}

absl::StatusOr<LoadedCandidateRankerManifest>
LoadRuntimeValidatedCandidateRankerManifest(absl::string_view manifest_path) {
  absl::StatusOr<LoadedCandidateRankerManifest> loaded_manifest =
      LoadManifest(manifest_path);
  if (!loaded_manifest.ok()) {
    return loaded_manifest.status();
  }
  const absl::Status capacity_status =
      ValidateRuntimeCapacities(loaded_manifest->manifest);
  if (!capacity_status.ok()) {
    return capacity_status;
  }
  return loaded_manifest;
}

llama_load_mode LoadMode(const CandidateRankerModelManifest::Runtime& runtime) {
  if (runtime.use_memory_map() && runtime.use_memory_lock()) {
    return LLAMA_LOAD_MODE_MMAP_MLOCK;
  }
  if (runtime.use_memory_map()) {
    return LLAMA_LOAD_MODE_MMAP;
  }
  if (runtime.use_memory_lock()) {
    return LLAMA_LOAD_MODE_MLOCK;
  }
  return LLAMA_LOAD_MODE_NONE;
}

absl::Status ValidateRuntimeCapacities(
    const CandidateRankerModelManifest& manifest) {
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  if (limits.per_record_token_limit() - 1 >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      limits.per_decode_input_node_limit() >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return absl::InvalidArgumentError(
        "candidate ranking token capacities exceed runtime limits");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int32_t>> TokenizeRecord(
    const llama_vocab* vocab, absl::string_view text, int32_t bos_token_id,
    int32_t terminal_token_id, bool reject_terminal,
    const converter::CandidateRankerCancellation& cancellation) {
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return absl::ResourceExhaustedError(
        "candidate ranking record exceeds the text limit");
  }
  std::vector<llama_token> record_tokens(text.size());
  int32_t count =
      llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                     record_tokens.data(),
                     static_cast<int32_t>(record_tokens.size()), false, false);
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (count == std::numeric_limits<int32_t>::min()) {
    return absl::ResourceExhaustedError(
        "candidate ranking record token count overflowed");
  }
  if (count < 0) {
    record_tokens.resize(static_cast<size_t>(-static_cast<int64_t>(count)));
    count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        record_tokens.data(), static_cast<int32_t>(record_tokens.size()), false,
        false);
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    if (count < 0) {
      return absl::InternalError(
          "candidate ranking tokenizer size changed between calls");
    }
  }
  record_tokens.resize(static_cast<size_t>(count));
  std::vector<int32_t> tokens;
  tokens.reserve(record_tokens.size() + 1);
  tokens.push_back(bos_token_id);
  for (const llama_token token : record_tokens) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    if (reject_terminal && token == terminal_token_id) {
      return absl::InvalidArgumentError(
          "candidate ranking terminal token cannot be an input token");
    }
    tokens.push_back(token);
  }
  return tokens;
}

absl::Status ValidateManifestVocabulary(
    const CandidateRankerModelManifest& manifest, const llama_vocab* vocab) {
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
  if (vocabulary_size <= 0) {
    return absl::InternalError("candidate ranking vocabulary is invalid");
  }
  const CandidateRankerModelManifest::ScoringTemplate& scoring_template =
      manifest.scoring_template();
  const uint32_t bos_token_id = scoring_template.bos_token_id();
  const uint32_t terminal_token_id = scoring_template.terminal_token_id();
  if (bos_token_id >= static_cast<uint32_t>(vocabulary_size) ||
      terminal_token_id >= static_cast<uint32_t>(vocabulary_size)) {
    return absl::InvalidArgumentError(
        "candidate ranking scoring tokens are outside the vocabulary");
  }
  const llama_token bos = static_cast<llama_token>(bos_token_id);
  const llama_token terminal = static_cast<llama_token>(terminal_token_id);
  if (bos != llama_vocab_bos(vocab) || !llama_vocab_is_control(vocab, bos)) {
    return absl::InvalidArgumentError(
        "candidate ranking BOS does not match the model control token");
  }
  if (terminal != llama_vocab_eos(vocab) ||
      !llama_vocab_is_eog(vocab, terminal)) {
    return absl::InvalidArgumentError(
        "candidate ranking terminal does not match the model end token");
  }
  return absl::OkStatus();
}

absl::Status ValidateModelContextCapacity(
    const CandidateRankerModelManifest& manifest, uint32_t model_context) {
  if (!manifest.has_limits() ||
      !manifest.limits().has_per_record_token_limit() ||
      manifest.limits().per_record_token_limit() == 0 ||
      model_context == 0 ||
      manifest.limits().per_record_token_limit() - 1 > model_context) {
    return absl::InvalidArgumentError(
        "candidate ranking record limit exceeds the model context");
  }
  return absl::OkStatus();
}

absl::Status ValidateManifestOutputLogitCapacity(
    const CandidateRankerModelManifest& manifest, int32_t vocabulary_size) {
  if (vocabulary_size <= 0) {
    return absl::InternalError("candidate ranking vocabulary is invalid");
  }
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const absl::StatusOr<uint64_t> maximum_output_logit_bytes =
      ComputeCandidateRankerOutputLogitBytes(
          limits.per_decode_output_row_limit(),
          limits.max_sequences_per_decode(),
          static_cast<uint64_t>(vocabulary_size));
  if (!maximum_output_logit_bytes.ok()) {
    return maximum_output_logit_bytes.status();
  }
  if (*maximum_output_logit_bytes > limits.max_decode_output_logit_bytes()) {
    return absl::InvalidArgumentError(
        "candidate ranking output-logit capacity exceeds the byte limit");
  }
  return absl::OkStatus();
}

struct ValidatedCandidateRankerModelMetadata {
  const llama_vocab* vocabulary = nullptr;
  int32_t vocabulary_size = 0;
};

absl::StatusOr<ValidatedCandidateRankerModelMetadata>
ValidateCandidateRankerModelMetadata(
    const CandidateRankerModelManifest& manifest, const llama_model* model,
    uint32_t verified_model_context, bool vocabulary_only) {
  if (!vocabulary_only) {
    const int32_t loaded_context = llama_model_n_ctx_train(model);
    if (loaded_context <= 0 ||
        static_cast<uint32_t>(loaded_context) != verified_model_context) {
      return absl::InvalidArgumentError(
          "candidate ranking loaded model context does not match metadata");
    }
  }
  if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
    return absl::InvalidArgumentError(
        "candidate ranking requires a decoder-only model");
  }
  const llama_vocab* vocabulary = llama_model_get_vocab(model);
  if (vocabulary == nullptr) {
    return absl::InternalError("candidate ranking vocabulary is unavailable");
  }
  const absl::Status vocabulary_status =
      ValidateManifestVocabulary(manifest, vocabulary);
  if (!vocabulary_status.ok()) {
    return vocabulary_status;
  }
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocabulary);
  const absl::Status output_capacity_status =
      ValidateManifestOutputLogitCapacity(manifest, vocabulary_size);
  if (!output_capacity_status.ok()) {
    return output_capacity_status;
  }
  return ValidatedCandidateRankerModelMetadata{
      .vocabulary = vocabulary,
      .vocabulary_size = vocabulary_size,
  };
}

struct EvaluationNumericProfile {
  absl::string_view name;
  enum ggml_type key_type;
  enum ggml_type value_type;
  enum llama_flash_attn_type flash_attention;
  absl::string_view key_type_name;
  absl::string_view value_type_name;
  absl::string_view flash_attention_name;
};

const EvaluationNumericProfile& GetEvaluationNumericProfile(
    LlamaCandidateRankerEvaluationProfile profile) {
  static constexpr EvaluationNumericProfile kConfigured{
      .name = "configured",
      .key_type = GGML_TYPE_F16,
      .value_type = GGML_TYPE_F16,
      .flash_attention = LLAMA_FLASH_ATTN_TYPE_AUTO,
      .key_type_name = "f16",
      .value_type_name = "f16",
      .flash_attention_name = "auto",
  };
  static constexpr EvaluationNumericProfile kF32KvFlashDisabled{
      .name = "f32_kv_flash_disabled",
      .key_type = GGML_TYPE_F32,
      .value_type = GGML_TYPE_F32,
      .flash_attention = LLAMA_FLASH_ATTN_TYPE_DISABLED,
      .key_type_name = "f32",
      .value_type_name = "f32",
      .flash_attention_name = "disabled",
  };
  switch (profile) {
    case LlamaCandidateRankerEvaluationProfile::kConfigured:
      return kConfigured;
    case LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled:
      return kF32KvFlashDisabled;
  }
  std::abort();
}

llama_context_params MakeContextParams(
    const CandidateRankerModelManifest& manifest, uint32_t sequence_capacity,
    uint32_t output_capacity, uint32_t outputs_per_sequence,
    LlamaCandidateRankerEvaluationProfile profile) {
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const CandidateRankerModelManifest::Runtime& runtime = manifest.runtime();
  llama_context_params params = llama_context_default_params();
  params.n_ctx = limits.per_decode_input_node_limit();
  params.n_batch = limits.per_decode_input_node_limit();
  params.n_ubatch = runtime.micro_batch_token_capacity();
  params.n_seq_max = sequence_capacity;
  params.n_outputs_max = output_capacity;
  params.n_outputs_max_per_seq = outputs_per_sequence;
  params.n_threads = runtime.decode_thread_count();
  params.n_threads_batch = runtime.batch_thread_count();
  const EvaluationNumericProfile& numeric_profile =
      GetEvaluationNumericProfile(profile);
  params.flash_attn_type = numeric_profile.flash_attention;
  params.type_k = numeric_profile.key_type;
  params.type_v = numeric_profile.value_type;
  params.offload_kqv = false;
  params.op_offload = false;
  params.kv_unified = true;
  return params;
}

absl::Status ValidateContextCapacities(
    llama_context* context, const CandidateRankerModelManifest& manifest,
    uint32_t sequence_capacity) {
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const CandidateRankerModelManifest::Runtime& runtime = manifest.runtime();
  if (llama_get_memory(context) == nullptr ||
      llama_n_ctx(context) < limits.per_decode_input_node_limit() ||
      llama_n_ctx_seq(context) < limits.per_decode_input_node_limit() ||
      llama_n_batch(context) < limits.per_decode_input_node_limit() ||
      llama_n_ubatch(context) < runtime.micro_batch_token_capacity() ||
      llama_n_seq_max(context) < sequence_capacity) {
    return absl::InternalError(
        "candidate ranking context capacities are insufficient");
  }
  return absl::OkStatus();
}

using PreparedCandidateRankingBatch = CandidateRankerDecodeBatch;

struct PreparedCandidateRanking {
  converter::CandidateRankerResponse response;
  CandidateRankerRecordContext record_context;
  std::vector<PreparedCandidateRankingBatch> batches;
  CandidateRankerCapacityResult capacity;
};

absl::StatusOr<CandidateRankerDecodeCapacityUsage> BuildDecodeCapacityUsage(
    const CandidateRankerBatchPlan& plan,
    const CandidateRankerModelManifest& manifest, int32_t vocabulary_size) {
  CandidateRankerDecodeCapacityUsage usage{
      .segments = plan.segments.size(),
      .sequences = plan.sequence_count,
      .input_nodes = plan.inputs.size(),
      .output_rows = plan.output_row_count,
      .reserved_output_rows =
          std::max<uint64_t>(plan.output_row_count,
                             manifest.limits().max_sequences_per_decode()),
  };
  absl::StatusOr<uint64_t> output_logit_bytes =
      ComputeCandidateRankerOutputLogitBytes(
          usage.output_rows, usage.reserved_output_rows,
          static_cast<uint64_t>(vocabulary_size));
  if (!output_logit_bytes.ok()) {
    return output_logit_bytes.status();
  }
  usage.output_logit_bytes = *output_logit_bytes;
  return usage;
}

void UpdateDecodeCapacityMaxima(
    const CandidateRankerDecodeCapacityUsage& decode,
    CandidateRankerCapacityUsage* usage) {
  usage->max_segments_in_decode =
      std::max(usage->max_segments_in_decode, decode.segments);
  usage->max_sequences_in_decode =
      std::max(usage->max_sequences_in_decode, decode.sequences);
  usage->max_input_nodes_in_decode =
      std::max(usage->max_input_nodes_in_decode, decode.input_nodes);
  usage->max_output_rows_in_decode =
      std::max(usage->max_output_rows_in_decode, decode.output_rows);
  usage->max_reserved_output_rows_in_decode = std::max(
      usage->max_reserved_output_rows_in_decode, decode.reserved_output_rows);
  usage->max_output_logit_bytes_in_decode = std::max(
      usage->max_output_logit_bytes_in_decode, decode.output_logit_bytes);
}

void UpdateSelectedRecordMaxima(
    const CandidateRankerTokenizedSegment& segment,
    CandidateRankerCapacityUsage* usage) {
  for (const CandidateRankerTokenSequence& candidate : segment.candidates) {
    usage->max_selected_serialized_record_bytes =
        std::max(usage->max_selected_serialized_record_bytes,
                 candidate.serialized_record_bytes);
    usage->max_selected_normalized_record_bytes =
        std::max(usage->max_selected_normalized_record_bytes,
                 candidate.normalized_record_bytes);
    usage->max_selected_full_record_tokens =
        std::max(usage->max_selected_full_record_tokens,
                 candidate.full_record_tokens);
  }
}

absl::Status RequestLimitStatus(
    const CandidateRankerRequestLimitViolation& violation) {
  switch (violation.limit) {
    case CandidateRankerRequestLimit::kSegments:
      return absl::ResourceExhaustedError(
          "candidate ranking request exceeds the segment limit");
    case CandidateRankerRequestLimit::kCandidates:
      return absl::ResourceExhaustedError(
          "candidate ranking request exceeds the candidate limit");
    case CandidateRankerRequestLimit::kStringBytes:
      return absl::ResourceExhaustedError(
          "candidate ranking request exceeds the string byte limit");
  }
  return absl::InternalError("candidate ranking request limit is invalid");
}

absl::StatusOr<PreparedCandidateRanking> PrepareCandidateRanking(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerCancellation& cancellation,
    const CandidateRankerModelManifest& manifest,
    const llama_vocab* vocabulary, int32_t vocabulary_size) {
  PreparedCandidateRanking prepared{
      .response = converter::CandidateRankerResponse{.token = request.token},
  };
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const int32_t bos_token_id =
      static_cast<int32_t>(manifest.scoring_template().bos_token_id());
  const int32_t terminal_token_id =
      static_cast<int32_t>(manifest.scoring_template().terminal_token_id());
  CandidateRankerCapacityUsage* usage = &prepared.capacity.usage;
  absl::StatusOr<CandidateRankerRequestPreflight> preflight =
      PreflightCandidateRankerRequest(request, manifest, cancellation);
  if (!preflight.ok()) {
    return preflight.status();
  }
  usage->request_segment_count = preflight->segment_count;
  usage->request_candidate_count = preflight->candidate_count;
  usage->request_string_bytes = preflight->string_bytes;
  if (preflight->violation.has_value()) {
    prepared.capacity.request_limit_violation = *preflight->violation;
    return prepared;
  }

  absl::StatusOr<CandidateRankerRecordContext> record_context =
      BuildCandidateRankerRecordContext(request, manifest);
  if (!record_context.ok()) {
    return record_context.status();
  }
  prepared.record_context = std::move(*record_context);

  std::vector<CandidateRankerTokenizedSegment> selected_segments;
  selected_segments.reserve(prepared.record_context.segments.size());
  for (const converter::CandidateRankerSegment& segment :
       prepared.record_context.segments) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    uint64_t unprotected_candidate_count = 0;
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (!candidate.is_protected) {
        ++unprotected_candidate_count;
      }
    }
    usage->request_unprotected_candidate_count += unprotected_candidate_count;
    usage->max_unprotected_candidates_in_segment =
        std::max(usage->max_unprotected_candidates_in_segment,
                 unprotected_candidate_count);
    const uint64_t window_candidate_count = std::min<uint64_t>(
        unprotected_candidate_count,
        limits.max_selected_candidates_per_segment());
    usage->window_candidate_count += window_candidate_count;
    usage->candidates_omitted_by_window +=
        unprotected_candidate_count - window_candidate_count;
    if (unprotected_candidate_count <= 1) {
      continue;
    }
    ++usage->request_rankable_segment_count;

    CandidateRankerSegmentCapacityDiagnostic diagnostic{
        .segment_id = segment.id,
        .unprotected_candidate_count = unprotected_candidate_count,
        .window_candidate_count = window_candidate_count,
        .omitted_by_window =
            unprotected_candidate_count - window_candidate_count,
    };
    absl::StatusOr<std::vector<CandidateRankerCandidateReference>> candidates =
        SelectCandidateRankerCandidates(segment, manifest);
    if (!candidates.ok()) {
      return candidates.status();
    }

    CandidateRankerTokenizedSegment tokenized_segment{
        .segment_id = segment.id,
    };
    tokenized_segment.candidates.reserve(candidates->size());
    for (const CandidateRankerCandidateReference& candidate : *candidates) {
      if (cancellation.IsCancellationRequested()) {
        return absl::CancelledError("candidate ranking was cancelled");
      }
      absl::StatusOr<CandidateRankerPreparedRecord> record =
          PrepareCandidateRankerRecord(prepared.record_context, segment.id,
                                       candidate.candidate->id, manifest);
      if (cancellation.IsCancellationRequested()) {
        return absl::CancelledError("candidate ranking was cancelled");
      }
      if (!record.ok()) {
        return record.status();
      }
      if (record->violation.has_value()) {
        diagnostic.first_limiter = *record->violation;
        break;
      }
      absl::StatusOr<std::vector<int32_t>> tokens = TokenizeRecord(
          vocabulary, record->normalized_record, bos_token_id,
          terminal_token_id, true, cancellation);
      if (!tokens.ok()) {
        return tokens.status();
      }
      std::string().swap(record->normalized_record);
      const uint64_t full_record_tokens =
          static_cast<uint64_t>(tokens->size()) + 1;
      if (full_record_tokens > limits.per_record_token_limit()) {
        diagnostic.first_limiter = CandidateRankerCapacityViolation{
            .limit = CandidateRankerCapacityLimit::kFullRecordTokens,
            .observed = full_record_tokens,
            .allowed = limits.per_record_token_limit(),
        };
        break;
      }
      CandidateRankerTokenSequence tokenized_candidate{
          .candidate_id = candidate.candidate->id,
          .original_candidate_index = candidate.original_candidate_index,
          .candidate_value = candidate.candidate->value,
          .tokens = std::move(*tokens),
          .serialized_record_bytes = record->serialized_byte_count,
          .normalized_record_bytes = record->normalized_byte_count,
          .full_record_tokens = full_record_tokens,
      };

      std::vector<CandidateRankerTokenizedSegment> trial_segments = {
          tokenized_segment};
      trial_segments.front().candidates.push_back(tokenized_candidate);
      if (trial_segments.front().candidates.size() >= 2) {
        absl::StatusOr<CandidateRankerBatchPlan> trial_plan =
            BuildCandidateRankerBatchPlanForCapacityAudit(
                trial_segments, manifest, cancellation);
        if (!trial_plan.ok()) {
          return trial_plan.status();
        }
        absl::StatusOr<CandidateRankerDecodeCapacityUsage> trial_usage =
            BuildDecodeCapacityUsage(*trial_plan, manifest, vocabulary_size);
        if (!trial_usage.ok()) {
          return trial_usage.status();
        }
        absl::StatusOr<std::optional<CandidateRankerCapacityViolation>>
            violation = EvaluateCandidateRankerDecodeCapacity(
                *trial_usage, manifest);
        if (!violation.ok()) {
          return violation.status();
        }
        if (violation->has_value()) {
          diagnostic.first_limiter = **violation;
          break;
        }
      }
      tokenized_segment.candidates.push_back(std::move(tokenized_candidate));
    }
    if (tokenized_segment.candidates.size() < 2) {
      diagnostic.disposition =
          CandidateRankerSegmentCapacityDisposition::kOmittedCapacity;
      diagnostic.selected_candidate_count = 0;
      diagnostic.omitted_by_capacity = window_candidate_count;
      usage->candidates_omitted_by_capacity += window_candidate_count;
      ++usage->segments_omitted_by_capacity;
      prepared.capacity.segments.push_back(std::move(diagnostic));
      continue;
    }
    diagnostic.selected_candidate_count = tokenized_segment.candidates.size();
    diagnostic.omitted_by_capacity =
        window_candidate_count - diagnostic.selected_candidate_count;
    usage->candidates_omitted_by_capacity += diagnostic.omitted_by_capacity;
    ++usage->selected_segment_count;
    usage->selected_candidate_count += diagnostic.selected_candidate_count;
    UpdateSelectedRecordMaxima(tokenized_segment, usage);
    prepared.capacity.segments.push_back(std::move(diagnostic));
    selected_segments.push_back(std::move(tokenized_segment));
  }

  absl::StatusOr<std::vector<CandidateRankerDecodeBatch>> batches =
      PackCandidateRankerSegmentsForDecode(
          selected_segments, static_cast<uint64_t>(vocabulary_size), manifest,
          cancellation);
  if (!batches.ok()) {
    return batches.status();
  }
  prepared.batches = std::move(*batches);
  usage->decode_batch_count = prepared.batches.size();
  for (const PreparedCandidateRankingBatch& batch : prepared.batches) {
    UpdateDecodeCapacityMaxima(batch.usage, usage);
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  return prepared;
}

absl::Status AddEvaluationAttribution(
    PreparedCandidateRanking* prepared,
    const converter::CandidateRankerCancellation& cancellation,
    const CandidateRankerModelManifest& manifest,
    const llama_vocab* vocabulary) {
  const int32_t bos_token_id =
      static_cast<int32_t>(manifest.scoring_template().bos_token_id());
  const int32_t terminal_token_id =
      static_cast<int32_t>(manifest.scoring_template().terminal_token_id());
  for (PreparedCandidateRankingBatch& batch : prepared->batches) {
    for (CandidateRankerTokenizedSegment& segment :
         batch.tokenized_segments) {
      for (CandidateRankerTokenSequence& candidate : segment.candidates) {
        if (cancellation.IsCancellationRequested()) {
          return absl::CancelledError("candidate ranking was cancelled");
        }
        absl::StatusOr<CandidateRankerPreparedRecord> candidate_ending =
            PrepareCandidateRankerCandidateEndingRecord(
                prepared->record_context, segment.segment_id,
                candidate.candidate_id, manifest);
        if (cancellation.IsCancellationRequested()) {
          return absl::CancelledError("candidate ranking was cancelled");
        }
        if (!candidate_ending.ok()) {
          return candidate_ending.status();
        }
        if (candidate_ending->violation.has_value()) {
          return absl::InternalError(
              "candidate ranking attribution prefix exceeds its complete "
              "record");
        }
        absl::StatusOr<std::vector<int32_t>> candidate_ending_tokens =
            TokenizeRecord(vocabulary, candidate_ending->normalized_record,
                           bos_token_id, terminal_token_id, false,
                           cancellation);
        if (!candidate_ending_tokens.ok()) {
          return candidate_ending_tokens.status();
        }
        absl::StatusOr<uint32_t> following_context_start_position =
            FindCandidateRankerFollowingContextStartPosition(
                candidate.tokens, *candidate_ending_tokens);
        if (!following_context_start_position.ok()) {
          return following_context_start_position.status();
        }
        candidate.following_context_start_position =
            *following_context_start_position;
      }
    }
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<const CandidateRankerTokenSequence*>>
IndexTokenSequences(const PreparedCandidateRankingBatch& prepared) {
  std::vector<const CandidateRankerTokenSequence*> sequences(
      prepared.plan.sequence_count, nullptr);
  for (const CandidateRankerSegmentPlan& segment_plan :
       prepared.plan.segments) {
    const CandidateRankerTokenizedSegment* tokenized_segment = nullptr;
    for (const CandidateRankerTokenizedSegment& segment :
         prepared.tokenized_segments) {
      if (segment.segment_id == segment_plan.segment_id) {
        tokenized_segment = &segment;
        break;
      }
    }
    if (tokenized_segment == nullptr) {
      return absl::InternalError(
          "candidate ranking tokenized segment is unavailable");
    }
    for (const CandidateRankerPlannedCandidate& planned_candidate :
         segment_plan.candidates) {
      const CandidateRankerTokenSequence* token_sequence = nullptr;
      for (const CandidateRankerTokenSequence& candidate :
           tokenized_segment->candidates) {
        if (candidate.candidate_id == planned_candidate.candidate_id) {
          token_sequence = &candidate;
          break;
        }
      }
      if (token_sequence == nullptr || planned_candidate.sequence_id < 0 ||
          static_cast<size_t>(planned_candidate.sequence_id) >=
              sequences.size() ||
          sequences[planned_candidate.sequence_id] != nullptr) {
        return absl::InternalError(
            "candidate ranking token sequence index is invalid");
      }
      sequences[planned_candidate.sequence_id] = token_sequence;
    }
  }
  for (const CandidateRankerTokenSequence* sequence : sequences) {
    if (sequence == nullptr) {
      return absl::InternalError(
          "candidate ranking token sequence index is incomplete");
    }
  }
  return sequences;
}

struct DecodedCandidateScores {
  std::vector<double> sequence_scores;
  std::vector<std::vector<LlamaCandidateRankerEdgeScore>> sequence_edges;
  uint32_t input_node_count = 0;
  uint32_t output_row_count = 0;
};

absl::Status ValidateSharedPlan(const CandidateRankerBatchPlan& plan,
                                const CandidateRankerModelManifest& manifest,
                                int32_t vocabulary_size,
                                int32_t model_context,
                                const converter::CandidateRankerCancellation&
                                    cancellation) {
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  if (plan.inputs.empty() || plan.segments.empty() ||
      plan.inputs.size() > limits.per_decode_input_node_limit() ||
      plan.sequence_count == 0 ||
      plan.sequence_count > limits.max_sequences_per_decode() ||
      plan.output_row_count == 0 ||
      plan.output_row_count > limits.per_decode_output_row_limit()) {
    return absl::InternalError(
        "candidate ranking batch plan exceeds runtime capacities");
  }
  absl::StatusOr<CandidateRankerDecodeCapacityUsage> usage =
      BuildDecodeCapacityUsage(plan, manifest, vocabulary_size);
  if (!usage.ok()) {
    return usage.status();
  }
  absl::StatusOr<std::optional<CandidateRankerCapacityViolation>> violation =
      EvaluateCandidateRankerDecodeCapacity(*usage, manifest);
  if (!violation.ok()) {
    return violation.status();
  }
  if (violation->has_value()) {
    return absl::InternalError(
        "candidate ranking emitted batch exceeds a manifest capacity");
  }

  uint32_t counted_output_rows = 0;
  for (const CandidateRankerBatchInput& input : plan.inputs) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    if (input.token_id < 0 || input.token_id >= vocabulary_size ||
        input.position < 0 || input.position >= model_context ||
        input.sequence_ids.empty() ||
        input.sequence_ids.size() > limits.max_sequences_per_decode()) {
      return absl::InternalError(
          "candidate ranking batch input exceeds runtime capacities");
    }
    for (const int32_t sequence_id : input.sequence_ids) {
      if (sequence_id < 0 ||
          static_cast<uint32_t>(sequence_id) >= plan.sequence_count) {
        return absl::InternalError(
            "candidate ranking batch sequence ID is invalid");
      }
    }
    if (!input.scored_edges.empty()) {
      ++counted_output_rows;
    }
    for (const CandidateRankerScoredEdge& edge : input.scored_edges) {
      if (edge.target_token_id < 0 ||
          edge.target_token_id >= vocabulary_size) {
        return absl::InternalError(
            "candidate ranking scored token is outside the vocabulary");
      }
    }
  }
  if (counted_output_rows != plan.output_row_count) {
    return absl::InternalError(
        "candidate ranking output-row count is inconsistent");
  }
  return absl::OkStatus();
}

absl::StatusOr<DecodedCandidateScores> DecodeSharedTrie(
    llama_context* context, const PreparedCandidateRankingBatch& prepared,
    const CandidateRankerModelManifest& manifest, int32_t vocabulary_size,
    int32_t model_context, int32_t terminal_token_id, bool collect_edges,
    const converter::CandidateRankerCancellation& cancellation) {
  const absl::Status plan_status =
      ValidateSharedPlan(prepared.plan, manifest, vocabulary_size,
                         model_context, cancellation);
  if (!plan_status.ok()) {
    return plan_status;
  }
  ContextMemoryScope memory_scope(context);
  AbortCallbackScope abort_scope(context, cancellation);
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  Batch batch(static_cast<int32_t>(prepared.plan.inputs.size()),
              static_cast<int32_t>(limits.max_sequences_per_decode()));
  llama_batch* batch_data = batch.get();
  batch_data->n_tokens = static_cast<int32_t>(prepared.plan.inputs.size());
  std::vector<int32_t> output_indices;
  output_indices.reserve(prepared.plan.output_row_count);
  for (size_t i = 0; i < prepared.plan.inputs.size(); ++i) {
    const CandidateRankerBatchInput& input = prepared.plan.inputs[i];
    batch_data->token[i] = static_cast<llama_token>(input.token_id);
    batch_data->pos[i] = static_cast<llama_pos>(input.position);
    batch_data->n_seq_id[i] = static_cast<int32_t>(input.sequence_ids.size());
    for (size_t j = 0; j < input.sequence_ids.size(); ++j) {
      batch_data->seq_id[i][j] =
          static_cast<llama_seq_id>(input.sequence_ids[j]);
    }
    batch_data->logits[i] = !input.scored_edges.empty();
    if (batch_data->logits[i] != 0) {
      output_indices.push_back(static_cast<int32_t>(i));
    }
  }
  const int32_t decode_result = llama_decode(context, *batch_data);
  if (decode_result == 2 || cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (decode_result != 0) {
    return absl::InternalError("candidate ranking decode failed");
  }

  DecodedCandidateScores decoded{
      .sequence_scores = std::vector<double>(prepared.plan.sequence_count, 0.0),
      .sequence_edges =
          std::vector<std::vector<LlamaCandidateRankerEdgeScore>>(
              prepared.plan.sequence_count),
      .input_node_count = static_cast<uint32_t>(prepared.plan.inputs.size()),
      .output_row_count = prepared.plan.output_row_count,
  };
  for (const int32_t output_index : output_indices) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    const float* logits = llama_get_logits_ith(context, output_index);
    if (logits == nullptr) {
      return absl::InternalError("candidate ranking logits are unavailable");
    }
    const CandidateRankerBatchInput& input =
        prepared.plan.inputs[output_index];
    absl::StatusOr<std::vector<double>> edge_log_probabilities =
        ComputeCandidateRankerEdgeLogProbabilities(
            input, absl::Span<const float>(
                       logits, static_cast<size_t>(vocabulary_size)),
            &cancellation);
    if (!edge_log_probabilities.ok()) {
      return edge_log_probabilities.status();
    }
    const absl::Status score_status =
        AccumulateCandidateRankerEdgeLogProbabilities(
            input, *edge_log_probabilities,
            absl::MakeSpan(decoded.sequence_scores));
    if (!score_status.ok()) {
      return score_status;
    }
    for (size_t edge_index = 0; edge_index < input.scored_edges.size();
         ++edge_index) {
      const CandidateRankerScoredEdge& edge = input.scored_edges[edge_index];
      if (collect_edges) {
        for (const int32_t sequence_id : edge.descendant_sequence_ids) {
          decoded.sequence_edges[sequence_id].push_back(
              LlamaCandidateRankerEdgeScore{
                  .target_position =
                      static_cast<uint32_t>(input.position + 1),
                  .target_token_id = edge.target_token_id,
                  .is_terminal = edge.target_token_id == terminal_token_id,
                  .log_probability = (*edge_log_probabilities)[edge_index],
              });
        }
      }
    }
  }
  return decoded;
}

absl::StatusOr<uint32_t> MaximumIndependentOutputRows(
    const PreparedCandidateRankingBatch& prepared,
    absl::Span<const CandidateRankerTokenSequence* const> sequences) {
  uint32_t maximum = 0;
  for (const CandidateRankerSegmentPlan& segment : prepared.plan.segments) {
    for (const CandidateRankerPlannedCandidate& candidate :
         segment.candidates) {
      const CandidateRankerTokenSequence& sequence =
          *sequences[candidate.sequence_id];
      if (segment.common_prefix_token_count == 0 ||
          segment.common_prefix_token_count > sequence.tokens.size()) {
        return absl::InternalError(
            "candidate ranking common token prefix is invalid");
      }
      maximum = std::max(
          maximum,
          static_cast<uint32_t>(sequence.tokens.size() -
                                segment.common_prefix_token_count + 1));
    }
  }
  return maximum;
}

absl::StatusOr<DecodedCandidateScores> DecodeIndependentSequences(
    llama_context* context, const PreparedCandidateRankingBatch& prepared,
    absl::Span<const CandidateRankerTokenSequence* const> sequences,
    int32_t vocabulary_size, int32_t terminal_token_id,
    const converter::CandidateRankerCancellation& cancellation) {
  AbortCallbackScope abort_scope(context, cancellation);
  DecodedCandidateScores decoded{
      .sequence_scores = std::vector<double>(prepared.plan.sequence_count, 0.0),
      .sequence_edges =
          std::vector<std::vector<LlamaCandidateRankerEdgeScore>>(
              prepared.plan.sequence_count),
  };
  for (const CandidateRankerSegmentPlan& segment : prepared.plan.segments) {
    for (const CandidateRankerPlannedCandidate& candidate :
         segment.candidates) {
      if (cancellation.IsCancellationRequested()) {
        return absl::CancelledError("candidate ranking was cancelled");
      }
      const CandidateRankerTokenSequence& sequence =
          *sequences[candidate.sequence_id];
      ContextMemoryScope memory_scope(context);
      Batch batch(static_cast<int32_t>(sequence.tokens.size()), 1);
      llama_batch* batch_data = batch.get();
      batch_data->n_tokens = static_cast<int32_t>(sequence.tokens.size());
      std::vector<int32_t> output_indices;
      output_indices.reserve(sequence.tokens.size() -
                             segment.common_prefix_token_count + 1);
      for (size_t position = 0; position < sequence.tokens.size(); ++position) {
        batch_data->token[position] =
            static_cast<llama_token>(sequence.tokens[position]);
        batch_data->pos[position] = static_cast<llama_pos>(position);
        batch_data->n_seq_id[position] = 1;
        batch_data->seq_id[position][0] = 0;
        batch_data->logits[position] =
            position + 1 >= segment.common_prefix_token_count;
        if (batch_data->logits[position] != 0) {
          output_indices.push_back(static_cast<int32_t>(position));
        }
      }
      const int32_t decode_result = llama_decode(context, *batch_data);
      if (decode_result == 2 || cancellation.IsCancellationRequested()) {
        return absl::CancelledError("candidate ranking was cancelled");
      }
      if (decode_result != 0) {
        return absl::InternalError(
            "independent candidate ranking decode failed");
      }
      decoded.input_node_count +=
          static_cast<uint32_t>(sequence.tokens.size());
      decoded.output_row_count +=
          static_cast<uint32_t>(output_indices.size());
      for (const int32_t output_index : output_indices) {
        if (cancellation.IsCancellationRequested()) {
          return absl::CancelledError("candidate ranking was cancelled");
        }
        const size_t target_position = static_cast<size_t>(output_index) + 1;
        const bool is_terminal = target_position == sequence.tokens.size();
        const int32_t target_token_id =
            is_terminal ? terminal_token_id : sequence.tokens[target_position];
        const CandidateRankerBatchInput input{
            .token_id = sequence.tokens[output_index],
            .position = output_index,
            .sequence_ids = {candidate.sequence_id},
            .scored_edges = {{.target_token_id = target_token_id,
                              .descendant_sequence_ids =
                                  {candidate.sequence_id}}},
        };
        const float* logits = llama_get_logits_ith(context, output_index);
        if (logits == nullptr) {
          return absl::InternalError(
              "independent candidate ranking logits are unavailable");
        }
        absl::StatusOr<std::vector<double>> edge_log_probabilities =
            ComputeCandidateRankerEdgeLogProbabilities(
                input, absl::Span<const float>(
                           logits, static_cast<size_t>(vocabulary_size)),
                &cancellation);
        if (!edge_log_probabilities.ok()) {
          return edge_log_probabilities.status();
        }
        const absl::Status score_status =
            AccumulateCandidateRankerEdgeLogProbabilities(
                input, *edge_log_probabilities,
                absl::MakeSpan(decoded.sequence_scores));
        if (!score_status.ok()) {
          return score_status;
        }
        decoded.sequence_edges[candidate.sequence_id].push_back(
            LlamaCandidateRankerEdgeScore{
                .target_position = static_cast<uint32_t>(target_position),
                .target_token_id = target_token_id,
                .is_terminal = is_terminal,
                .log_probability = edge_log_probabilities->front(),
            });
      }
    }
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  return decoded;
}

absl::Status AppendEvaluationBatch(
    const PreparedCandidateRankingBatch& prepared,
    absl::Span<const CandidateRankerTokenSequence* const> sequences,
    DecodedCandidateScores decoded,
    LlamaCandidateRankerEvaluationResult* result) {
  absl::StatusOr<std::vector<converter::CandidateRankerSegmentOrder>> orders =
      OrderCandidateRankerContinuations(prepared.plan,
                                        decoded.sequence_scores);
  if (!orders.ok()) {
    return orders.status();
  }
  for (converter::CandidateRankerSegmentOrder& order : *orders) {
    result->response.segment_orders.push_back(std::move(order));
  }
  result->input_node_count += decoded.input_node_count;
  result->output_row_count += decoded.output_row_count;
  ++result->decode_batch_count;
  result->candidate_scores.reserve(result->candidate_scores.size() +
                                   prepared.plan.sequence_count);
  for (const CandidateRankerSegmentPlan& segment : prepared.plan.segments) {
    for (const CandidateRankerPlannedCandidate& candidate :
         segment.candidates) {
      std::vector<LlamaCandidateRankerEdgeScore>& edges =
          decoded.sequence_edges[candidate.sequence_id];
      const CandidateRankerTokenSequence& sequence =
          *sequences[candidate.sequence_id];
      for (LlamaCandidateRankerEdgeScore& edge : edges) {
        edge.contribution = ClassifyLlamaCandidateRankerEdgeContribution(
            edge.target_position, edge.is_terminal,
            sequence.following_context_start_position);
      }
      std::sort(edges.begin(), edges.end(),
                [](const LlamaCandidateRankerEdgeScore& lhs,
                   const LlamaCandidateRankerEdgeScore& rhs) {
                  return lhs.target_position < rhs.target_position;
                });
      if (edges.empty()) {
        return absl::InternalError(
            "candidate ranking continuation score has no token");
      }
      const uint32_t scored_continuation_token_count =
          static_cast<uint32_t>(edges.size());
      result->candidate_scores.push_back(LlamaCandidateRankerCandidateScore{
          .segment_id = segment.segment_id,
          .candidate_id = candidate.candidate_id,
          .common_prefix_token_count = segment.common_prefix_token_count,
          .input_token_ids = sequences[candidate.sequence_id]->tokens,
          .edges = std::move(edges),
          .full_continuation_log_probability =
              decoded.sequence_scores[candidate.sequence_id],
          .scored_continuation_token_count =
              scored_continuation_token_count,
      });
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateLlamaCandidateRankerModelContextCapacity(
    const CandidateRankerModelManifest& manifest, int32_t model_context) {
  if (model_context <= 0) {
    return ValidateModelContextCapacity(manifest, 0);
  }
  return ValidateModelContextCapacity(manifest,
                                      static_cast<uint32_t>(model_context));
}

LlamaCandidateRankerEvaluationIdentity
BuildLlamaCandidateRankerEvaluationIdentity(
    const CandidateRankerModelManifest& manifest,
    absl::string_view manifest_bytes) {
  const CandidateRankerModelManifest::Artifacts& artifacts =
      manifest.artifacts();
  return LlamaCandidateRankerEvaluationIdentity{
      .manifest_sha256 = ComputeBytesSha256(manifest_bytes),
      .gguf_file_name = artifacts.gguf_file_name(),
      .gguf_sha256 = artifacts.gguf_sha256(),
      .tokenizer_sha256 = artifacts.tokenizer_sha256(),
      .quantization = artifacts.quantization(),
      .source_revision = artifacts.source_revision(),
      .runtime_revision = artifacts.runtime_revision(),
  };
}

void WriteLlamaCandidateRankerEvaluationIdentity(
    std::ostream& output,
    const LlamaCandidateRankerEvaluationIdentity& identity) {
  output << "identity\tmanifest_sha256\t" << identity.manifest_sha256
         << "\tgguf_file_name\t" << identity.gguf_file_name
         << "\tgguf_sha256\t" << identity.gguf_sha256
         << "\ttokenizer_sha256\t" << identity.tokenizer_sha256
         << "\tquantization\t" << identity.quantization
         << "\tsource_revision\t" << identity.source_revision
         << "\truntime_revision\t" << identity.runtime_revision << '\n';
}

void WriteLlamaCandidateRankerEvaluationNumericProfile(
    std::ostream& output, LlamaCandidateRankerEvaluationProfile profile) {
  const EvaluationNumericProfile& numeric_profile =
      GetEvaluationNumericProfile(profile);
  output << "numeric_profile\t" << numeric_profile.name << "\tkv_key\t"
         << numeric_profile.key_type_name << "\tkv_value\t"
         << numeric_profile.value_type_name << "\tflash_attention\t"
         << numeric_profile.flash_attention_name << '\n';
}

LlamaCandidateRankerEdgeContribution
ClassifyLlamaCandidateRankerEdgeContribution(
    uint32_t target_position, bool is_terminal,
    uint32_t following_context_start_position) {
  if (is_terminal) {
    return LlamaCandidateRankerEdgeContribution::kTerminal;
  }
  if (target_position < following_context_start_position) {
    return LlamaCandidateRankerEdgeContribution::kCandidateOrBoundary;
  }
  return LlamaCandidateRankerEdgeContribution::kFollowingContext;
}

struct LlamaCandidateRankerCapacityAuditor::RuntimeState {
  ModelPointer model;
  const llama_vocab* vocabulary = nullptr;
  int32_t vocabulary_size = 0;
  std::mutex vocabulary_mutex;
};

absl::StatusOr<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>>
LlamaCandidateRankerCapacityAuditor::Create(
    absl::string_view manifest_path, absl::string_view model_directory,
    const LlamaCandidateRankerEvaluationIdentity& expected_identity) {
  absl::StatusOr<LoadedCandidateRankerManifest> loaded_manifest =
      LoadRuntimeValidatedCandidateRankerManifest(manifest_path);
  if (!loaded_manifest.ok()) {
    return loaded_manifest.status();
  }
  const absl::Status identity_status = ValidateExpectedEvaluationIdentity(
      loaded_manifest->evaluation_identity, expected_identity);
  if (!identity_status.ok()) {
    return identity_status;
  }
  absl::StatusOr<VerifiedCandidateRankerSource> source =
      OpenVerifiedCandidateRankerModel(std::move(*loaded_manifest),
                                       model_directory);
  if (!source.ok()) {
    return source.status();
  }
  LoadedCandidateRankerManifest* source_manifest = &source->loaded_manifest;
  CandidateRankerModelManifest* manifest = &source_manifest->manifest;

  EnsureLlamaProcessLifetime();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.load_mode = LoadMode(manifest->runtime());
  model_params.check_tensors = manifest->runtime().check_tensors();
  model_params.vocab_only = true;

  ModelPointer model(
      llama_model_load_from_file_ptr(source->model_file.get(), model_params));
  if (model == nullptr) {
    return absl::InternalError("candidate ranking model could not be loaded");
  }
  absl::StatusOr<ValidatedCandidateRankerModelMetadata> metadata =
      ValidateCandidateRankerModelMetadata(
          *manifest, model.get(), source->model_context, true);
  if (!metadata.ok()) {
    return metadata.status();
  }

  auto runtime_state = std::make_unique<RuntimeState>();
  runtime_state->model = std::move(model);
  runtime_state->vocabulary = metadata->vocabulary;
  runtime_state->vocabulary_size = metadata->vocabulary_size;
  return std::unique_ptr<LlamaCandidateRankerCapacityAuditor>(
      new LlamaCandidateRankerCapacityAuditor(
          std::move(*manifest),
          std::move(source_manifest->evaluation_identity),
          std::move(runtime_state)));
}

LlamaCandidateRankerCapacityAuditor::LlamaCandidateRankerCapacityAuditor(
    CandidateRankerModelManifest manifest,
    LlamaCandidateRankerEvaluationIdentity evaluation_identity,
    std::unique_ptr<RuntimeState> runtime)
    : manifest_(std::move(manifest)),
      evaluation_identity_(std::move(evaluation_identity)),
      runtime_(std::move(runtime)) {}

LlamaCandidateRankerCapacityAuditor::~LlamaCandidateRankerCapacityAuditor() =
    default;

absl::StatusOr<CandidateRankerCapacityResult>
LlamaCandidateRankerCapacityAuditor::Audit(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerCancellation& cancellation) {
  const std::lock_guard<std::mutex> vocabulary_lock(
      runtime_->vocabulary_mutex);
  absl::StatusOr<PreparedCandidateRanking> prepared =
      PrepareCandidateRanking(request, cancellation, manifest_,
                              runtime_->vocabulary,
                              runtime_->vocabulary_size);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  return std::move(prepared->capacity);
}

struct LlamaCandidateRanker::RuntimeState {
  ModelPointer model;
  ContextPointer context;
  const llama_vocab* vocabulary = nullptr;
  int32_t vocabulary_size = 0;
  int32_t model_context = 0;
  std::mutex context_mutex;
};

absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>>
LlamaCandidateRanker::Create(absl::string_view manifest_path,
                             absl::string_view model_directory) {
  return CreateImpl(manifest_path, model_directory, nullptr);
}

absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>>
LlamaCandidateRanker::Create(
    absl::string_view manifest_path, absl::string_view model_directory,
    const LlamaCandidateRankerEvaluationIdentity& expected_identity) {
  return CreateImpl(manifest_path, model_directory, &expected_identity);
}

absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>>
LlamaCandidateRanker::CreateImpl(
    absl::string_view manifest_path, absl::string_view model_directory,
    const LlamaCandidateRankerEvaluationIdentity* expected_identity) {
  absl::StatusOr<LoadedCandidateRankerManifest> loaded_manifest =
      LoadRuntimeValidatedCandidateRankerManifest(manifest_path);
  if (!loaded_manifest.ok()) {
    return loaded_manifest.status();
  }
  if (expected_identity != nullptr) {
    const absl::Status identity_status = ValidateExpectedEvaluationIdentity(
        loaded_manifest->evaluation_identity, *expected_identity);
    if (!identity_status.ok()) {
      return identity_status;
    }
  }
  absl::StatusOr<VerifiedCandidateRankerSource> source =
      OpenVerifiedCandidateRankerModel(std::move(*loaded_manifest),
                                       model_directory);
  if (!source.ok()) {
    return source.status();
  }
  LoadedCandidateRankerManifest* source_manifest = &source->loaded_manifest;
  CandidateRankerModelManifest* manifest = &source_manifest->manifest;

  EnsureLlamaProcessLifetime();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.load_mode = LoadMode(manifest->runtime());
  model_params.check_tensors = manifest->runtime().check_tensors();
  model_params.vocab_only = false;

  ModelPointer model(
      llama_model_load_from_file_ptr(source->model_file.get(), model_params));
  if (model == nullptr) {
    return absl::InternalError("candidate ranking model could not be loaded");
  }
  absl::StatusOr<ValidatedCandidateRankerModelMetadata> metadata =
      ValidateCandidateRankerModelMetadata(
          *manifest, model.get(), source->model_context, false);
  if (!metadata.ok()) {
    return metadata.status();
  }

  const CandidateRankerModelManifest::Limits& limits = manifest->limits();
  const llama_vocab* vocabulary = metadata->vocabulary;
  const int32_t vocabulary_size = metadata->vocabulary_size;

  const llama_context_params context_params = MakeContextParams(
      *manifest, limits.max_sequences_per_decode(),
      limits.per_decode_output_row_limit(), 1,
      LlamaCandidateRankerEvaluationProfile::kConfigured);

  ContextPointer context(llama_init_from_model(model.get(), context_params));
  if (context == nullptr) {
    return absl::InternalError(
        "candidate ranking context could not be initialized");
  }
  const absl::Status context_status = ValidateContextCapacities(
      context.get(), *manifest, limits.max_sequences_per_decode());
  if (!context_status.ok()) {
    return context_status;
  }

  auto runtime_state = std::make_unique<RuntimeState>();
  runtime_state->model = std::move(model);
  runtime_state->context = std::move(context);
  runtime_state->vocabulary = vocabulary;
  runtime_state->vocabulary_size = vocabulary_size;
  runtime_state->model_context =
      llama_model_n_ctx_train(runtime_state->model.get());
  return std::shared_ptr<LlamaCandidateRanker>(
      new LlamaCandidateRanker(
          std::move(*manifest),
          std::move(source_manifest->evaluation_identity),
          std::move(runtime_state)));
}

LlamaCandidateRanker::LlamaCandidateRanker(
    CandidateRankerModelManifest manifest,
    LlamaCandidateRankerEvaluationIdentity evaluation_identity,
    std::unique_ptr<RuntimeState> runtime)
    : manifest_(std::move(manifest)),
      evaluation_identity_(std::move(evaluation_identity)),
      runtime_(std::move(runtime)) {}

LlamaCandidateRanker::~LlamaCandidateRanker() = default;

absl::StatusOr<LlamaCandidateRankerEvaluationResult>
LlamaCandidateRanker::ScoreForEvaluation(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerCancellation& cancellation,
    LlamaCandidateRankerEvaluationLayout layout,
    LlamaCandidateRankerEvaluationProfile profile) {
  const std::lock_guard<std::mutex> context_lock(runtime_->context_mutex);
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (layout != LlamaCandidateRankerEvaluationLayout::kSharedTrie &&
      layout != LlamaCandidateRankerEvaluationLayout::
                    kIndependentCompleteSequences) {
    return absl::InvalidArgumentError(
        "candidate ranking evaluation layout is invalid");
  }
  if (profile != LlamaCandidateRankerEvaluationProfile::kConfigured &&
      profile != LlamaCandidateRankerEvaluationProfile::
                     kF32KvFlashDisabled) {
    return absl::InvalidArgumentError(
        "candidate ranking evaluation profile is invalid");
  }

  absl::StatusOr<PreparedCandidateRanking> prepared = PrepareCandidateRanking(
      request, cancellation, manifest_, runtime_->vocabulary,
      runtime_->vocabulary_size);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (prepared->capacity.request_limit_violation.has_value()) {
    return RequestLimitStatus(
        *prepared->capacity.request_limit_violation);
  }
  if (prepared->batches.empty()) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    return LlamaCandidateRankerEvaluationResult{
        .response = std::move(prepared->response),
    };
  }
  const absl::Status attribution_status = AddEvaluationAttribution(
      &*prepared, cancellation, manifest_, runtime_->vocabulary);
  if (!attribution_status.ok()) {
    return attribution_status;
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }

  llama_context* context = runtime_->context.get();
  ContextPointer evaluation_context;
  if (layout != LlamaCandidateRankerEvaluationLayout::kSharedTrie ||
      profile != LlamaCandidateRankerEvaluationProfile::kConfigured) {
    uint32_t sequence_capacity = manifest_.limits().max_sequences_per_decode();
    uint32_t output_capacity =
        manifest_.limits().per_decode_output_row_limit();
    uint32_t outputs_per_sequence = 1;
    if (layout == LlamaCandidateRankerEvaluationLayout::
                      kIndependentCompleteSequences) {
      sequence_capacity = 1;
      output_capacity = 0;
      for (const PreparedCandidateRankingBatch& batch : prepared->batches) {
        absl::StatusOr<std::vector<const CandidateRankerTokenSequence*>>
            sequences = IndexTokenSequences(batch);
        if (!sequences.ok()) {
          return sequences.status();
        }
        absl::StatusOr<uint32_t> maximum_output_rows =
            MaximumIndependentOutputRows(batch, *sequences);
        if (!maximum_output_rows.ok()) {
          return maximum_output_rows.status();
        }
        output_capacity = std::max(output_capacity, *maximum_output_rows);
      }
      outputs_per_sequence = 0;
      const absl::StatusOr<uint64_t> output_logit_bytes =
          ComputeCandidateRankerOutputLogitBytes(
              output_capacity, sequence_capacity,
              static_cast<uint64_t>(runtime_->vocabulary_size));
      if (!output_logit_bytes.ok()) {
        return output_logit_bytes.status();
      }
      if (*output_logit_bytes >
          manifest_.limits().max_decode_output_logit_bytes()) {
        return absl::ResourceExhaustedError(
            "independent candidate ranking output logits exceed the byte "
            "limit");
      }
    }
    const llama_context_params context_params = MakeContextParams(
        manifest_, sequence_capacity, output_capacity, outputs_per_sequence,
        profile);
    evaluation_context.reset(
        llama_init_from_model(runtime_->model.get(), context_params));
    if (evaluation_context == nullptr) {
      return absl::InternalError(
          "candidate ranking evaluation context could not be initialized");
    }
    const absl::Status context_status = ValidateContextCapacities(
        evaluation_context.get(), manifest_, sequence_capacity);
    if (!context_status.ok()) {
      return context_status;
    }
    context = evaluation_context.get();
  }

  const int32_t terminal_token_id =
      static_cast<int32_t>(manifest_.scoring_template().terminal_token_id());
  LlamaCandidateRankerEvaluationResult result{
      .response = std::move(prepared->response),
  };
  for (const PreparedCandidateRankingBatch& batch : prepared->batches) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    absl::StatusOr<std::vector<const CandidateRankerTokenSequence*>>
        sequences = IndexTokenSequences(batch);
    if (!sequences.ok()) {
      return sequences.status();
    }
    absl::StatusOr<DecodedCandidateScores> decoded =
        layout == LlamaCandidateRankerEvaluationLayout::kSharedTrie
            ? DecodeSharedTrie(
                  context, batch, manifest_, runtime_->vocabulary_size,
                  runtime_->model_context, terminal_token_id, true,
                  cancellation)
            : DecodeIndependentSequences(
                  context, batch, *sequences, runtime_->vocabulary_size,
                  terminal_token_id, cancellation);
    if (!decoded.ok()) {
      return decoded.status();
    }
    const absl::Status append_status = AppendEvaluationBatch(
        batch, *sequences, std::move(*decoded), &result);
    if (!append_status.ok()) {
      return append_status;
    }
  }
  if (result.response.segment_orders.size() !=
          prepared->capacity.usage.selected_segment_count ||
      result.candidate_scores.size() !=
          prepared->capacity.usage.selected_candidate_count) {
    return absl::InternalError(
        "candidate ranking evaluation aggregate is incomplete");
  }
  std::sort(result.response.segment_orders.begin(),
            result.response.segment_orders.end(),
            [](const converter::CandidateRankerSegmentOrder& lhs,
               const converter::CandidateRankerSegmentOrder& rhs) {
              return lhs.segment_id < rhs.segment_id;
            });
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  return result;
}

absl::StatusOr<converter::CandidateRankerResponse> LlamaCandidateRanker::Rank(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerCancellation& cancellation) {
  const std::lock_guard<std::mutex> context_lock(runtime_->context_mutex);
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }

  absl::StatusOr<PreparedCandidateRanking> prepared = PrepareCandidateRanking(
      request, cancellation, manifest_, runtime_->vocabulary,
      runtime_->vocabulary_size);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  if (prepared->capacity.request_limit_violation.has_value()) {
    return RequestLimitStatus(
        *prepared->capacity.request_limit_violation);
  }
  converter::CandidateRankerResponse response = std::move(prepared->response);
  if (prepared->batches.empty()) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    return response;
  }

  const int32_t terminal_token_id =
      static_cast<int32_t>(manifest_.scoring_template().terminal_token_id());
  std::vector<converter::CandidateRankerSegmentOrder> local_orders;
  local_orders.reserve(prepared->capacity.usage.selected_segment_count);
  for (const PreparedCandidateRankingBatch& batch : prepared->batches) {
    if (cancellation.IsCancellationRequested()) {
      return absl::CancelledError("candidate ranking was cancelled");
    }
    absl::StatusOr<DecodedCandidateScores> decoded = DecodeSharedTrie(
        runtime_->context.get(), batch, manifest_, runtime_->vocabulary_size,
        runtime_->model_context, terminal_token_id, false, cancellation);
    if (!decoded.ok()) {
      return decoded.status();
    }
    absl::StatusOr<std::vector<converter::CandidateRankerSegmentOrder>>
        batch_orders = OrderCandidateRankerContinuations(
            batch.plan, decoded->sequence_scores);
    if (!batch_orders.ok()) {
      return batch_orders.status();
    }
    for (converter::CandidateRankerSegmentOrder& order : *batch_orders) {
      local_orders.push_back(std::move(order));
    }
  }
  if (local_orders.size() != prepared->capacity.usage.selected_segment_count) {
    return absl::InternalError(
        "candidate ranking response aggregate is incomplete");
  }
  std::sort(local_orders.begin(), local_orders.end(),
            [](const converter::CandidateRankerSegmentOrder& lhs,
               const converter::CandidateRankerSegmentOrder& rhs) {
              return lhs.segment_id < rhs.segment_id;
            });
  for (size_t i = 1; i < local_orders.size(); ++i) {
    if (local_orders[i - 1].segment_id >= local_orders[i].segment_id) {
      return absl::InternalError(
          "candidate ranking response segment IDs are invalid");
    }
  }
  if (cancellation.IsCancellationRequested()) {
    return absl::CancelledError("candidate ranking was cancelled");
  }
  response.segment_orders = std::move(local_orders);
  return response;
}

}  // namespace engine
}  // namespace mozc
