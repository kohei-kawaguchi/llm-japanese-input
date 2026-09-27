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

#include "engine/candidate_ranker_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "base/strings/unicode.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.pb.h"
#include "engine/python_unicode14_lower.h"

namespace mozc {
namespace engine {
namespace {

constexpr uint32_t kMaximumSequenceCount = 256;
constexpr uint32_t kContextPadding = 256;
constexpr uint32_t kMaximumRecordByteLimit = 1024 * 1024;
constexpr uint32_t kMaximumThreadCount = 256;

struct MutableTrieNode {
  int32_t token_id = 0;
  int32_t position = 0;
  std::vector<int32_t> sequence_ids;
  std::map<int32_t, size_t> children;
  std::vector<int32_t> terminal_sequence_ids;
};

struct CanonicalSequence {
  const CandidateRankerTokenSequence* sequence = nullptr;
  int32_t sequence_id = 0;
};

bool IsLowercaseHex(const std::string& value, size_t expected_size) {
  if (value.size() != expected_size) {
    return false;
  }
  for (const char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool IsSimpleFileName(const std::string& value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find('/') == std::string::npos &&
         value.find('\\') == std::string::npos &&
         value.find(':') == std::string::npos &&
         value.find('\0') == std::string::npos;
}

bool ByteLess(absl::string_view lhs, absl::string_view rhs) {
  return std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      [](char lhs_byte, char rhs_byte) {
        return static_cast<unsigned char>(lhs_byte) <
               static_cast<unsigned char>(rhs_byte);
      });
}

absl::Status ValidateScoringTemplate(
    const CandidateRankerModelManifest::ScoringTemplate& scoring_template) {
  if (!scoring_template.has_version() || scoring_template.version().empty() ||
      !scoring_template.has_bos_token_id() ||
      !scoring_template.has_terminal_token_id()) {
    return absl::InvalidArgumentError(
        "scoring template version and token IDs are required");
  }
  const uint32_t max_token_id =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  if (scoring_template.bos_token_id() > max_token_id ||
      scoring_template.terminal_token_id() > max_token_id) {
    return absl::InvalidArgumentError(
        "scoring template token IDs must be signed 32-bit values");
  }
  if (!scoring_template.has_mode_prefix() ||
      scoring_template.mode_prefix().empty() ||
      !scoring_template.has_suggestion_mode() ||
      scoring_template.suggestion_mode().empty() ||
      !scoring_template.has_prediction_mode() ||
      scoring_template.prediction_mode().empty() ||
      !scoring_template.has_conversion_mode() ||
      scoring_template.conversion_mode().empty() ||
      !scoring_template.has_field_separator() ||
      scoring_template.field_separator().empty() ||
      !scoring_template.has_reading_prefix() ||
      scoring_template.reading_prefix().empty() ||
      !scoring_template.has_segment_key_prefix() ||
      scoring_template.segment_key_prefix().empty() ||
      !scoring_template.has_text_prefix() ||
      scoring_template.text_prefix().empty()) {
    return absl::InvalidArgumentError(
        "every scoring template literal must be present and nonempty");
  }
  if (!scoring_template.has_text_normalization() ||
      (scoring_template.text_normalization() !=
           CandidateRankerModelManifest::ScoringTemplate::NONE &&
       scoring_template.text_normalization() !=
           CandidateRankerModelManifest::ScoringTemplate::
               PYTHON_UNICODE_14_SCALAR_LOWER)) {
    return absl::InvalidArgumentError(
        "scoring template text normalization is invalid");
  }
  if (!scoring_template.has_context_policy() ||
      scoring_template.context_policy() !=
          CandidateRankerModelManifest::ScoringTemplate::
              MOZC_BASELINE_SUBSTITUTION) {
    return absl::InvalidArgumentError(
        "scoring template context policy is invalid");
  }
  if (!scoring_template.has_record_layout() ||
      (scoring_template.record_layout() !=
           CandidateRankerModelManifest::ScoringTemplate::
               STRUCTURED_WITH_TARGET &&
       scoring_template.record_layout() !=
           CandidateRankerModelManifest::ScoringTemplate::
               STRUCTURED_WITHOUT_TARGET &&
       scoring_template.record_layout() !=
           CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY)) {
    return absl::InvalidArgumentError(
        "scoring template record layout is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateArtifacts(
    const CandidateRankerModelManifest::Artifacts& artifacts) {
  if (!artifacts.has_source_model() || artifacts.source_model().empty() ||
      !artifacts.has_source_revision() || artifacts.source_revision().empty() ||
      !artifacts.has_gguf_file_name() ||
      !IsSimpleFileName(artifacts.gguf_file_name()) ||
      !artifacts.has_quantization() || artifacts.quantization().empty() ||
      !artifacts.has_runtime_name() || artifacts.runtime_name().empty() ||
      !artifacts.has_runtime_revision() ||
      artifacts.runtime_revision().empty()) {
    return absl::InvalidArgumentError(
        "model and runtime artifact metadata is required");
  }
  if (!IsLowercaseHex(artifacts.source_revision(), 40) ||
      !IsLowercaseHex(artifacts.runtime_revision(), 40)) {
    return absl::InvalidArgumentError(
        "model and runtime revisions must be 40 lowercase hexadecimal "
        "characters");
  }
  if (!artifacts.has_source_weight_sha256() ||
      !IsLowercaseHex(artifacts.source_weight_sha256(), 64) ||
      !artifacts.has_tokenizer_sha256() ||
      !IsLowercaseHex(artifacts.tokenizer_sha256(), 64) ||
      !artifacts.has_gguf_sha256() ||
      !IsLowercaseHex(artifacts.gguf_sha256(), 64) ||
      !artifacts.has_runtime_source_archive_sha256() ||
      !IsLowercaseHex(artifacts.runtime_source_archive_sha256(), 64)) {
    return absl::InvalidArgumentError(
        "artifact hashes must be 64 lowercase hexadecimal characters");
  }
  if (artifacts.model_license_references().empty() ||
      artifacts.runtime_license_references().empty()) {
    return absl::InvalidArgumentError(
        "model and runtime license references are required");
  }
  for (const std::string& reference : artifacts.model_license_references()) {
    if (reference.empty()) {
      return absl::InvalidArgumentError(
          "model license references must be nonempty");
    }
  }
  for (const std::string& reference : artifacts.runtime_license_references()) {
    if (reference.empty()) {
      return absl::InvalidArgumentError(
          "runtime license references must be nonempty");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateBoundedExecutionPolicy(
    const CandidateRankerModelManifest::BoundedExecutionPolicy& policy) {
  if (!policy.has_candidate_window_policy() ||
      policy.candidate_window_policy() !=
          CandidateRankerModelManifest::BoundedExecutionPolicy::
              MOZC_ORDER_UNPROTECTED_PREFIX ||
      !policy.has_capacity_reduction_policy() ||
      policy.capacity_reduction_policy() !=
          CandidateRankerModelManifest::BoundedExecutionPolicy::
              LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT ||
      !policy.has_decode_packing_policy() ||
      policy.decode_packing_policy() !=
          CandidateRankerModelManifest::BoundedExecutionPolicy::
              SEGMENT_ID_ORDER_GREEDY_WHOLE_SEGMENTS) {
    return absl::InvalidArgumentError(
        "candidate ranking bounded execution policy is invalid");
  }
  return absl::OkStatus();
}

absl::StatusOr<const std::string*> ModeText(
    converter::CandidateRankerMode mode,
    const CandidateRankerModelManifest::ScoringTemplate& scoring_template) {
  switch (mode) {
    case converter::CandidateRankerMode::kSuggestion:
      return &scoring_template.suggestion_mode();
    case converter::CandidateRankerMode::kPrediction:
      return &scoring_template.prediction_mode();
    case converter::CandidateRankerMode::kConversion:
      return &scoring_template.conversion_mode();
  }
  return absl::InvalidArgumentError("candidate ranker mode is invalid");
}

bool AddRecordPartSize(absl::string_view part, uint64_t* total) {
  if (part.size() > std::numeric_limits<uint64_t>::max() - *total) {
    return false;
  }
  *total += part.size();
  return true;
}

absl::StatusOr<std::string> NormalizeRecord(
    std::string record,
    CandidateRankerModelManifest::ScoringTemplate::TextNormalization
        normalization) {
  switch (normalization) {
    case CandidateRankerModelManifest::ScoringTemplate::NONE:
      return record;
    case CandidateRankerModelManifest::ScoringTemplate::
        PYTHON_UNICODE_14_SCALAR_LOWER:
      return PythonUnicode14Lower(record);
    case CandidateRankerModelManifest::ScoringTemplate::
        TEXT_NORMALIZATION_UNSPECIFIED:
      break;
  }
  return absl::InvalidArgumentError(
      "scoring template text normalization is invalid");
}

absl::Status CancelledStatus() {
  return absl::CancelledError("candidate ranking was cancelled");
}

absl::Status ValidateRecordContext(
    const CandidateRankerRecordContext& context) {
  std::set<converter::CandidateRankerCandidateId> candidate_ids;
  for (size_t segment_index = 0; segment_index < context.segments.size();
       ++segment_index) {
    const converter::CandidateRankerSegment& segment =
        context.segments[segment_index];
    if (segment_index > 0 &&
        context.segments[segment_index - 1].id >= segment.id) {
      return absl::InvalidArgumentError(
          "candidate ranking record segment IDs must be unique and ordered");
    }
    if (segment.candidates.empty()) {
      return absl::InvalidArgumentError(
          "every candidate ranking segment requires a baseline candidate");
    }
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (!candidate_ids.insert(candidate.id).second) {
        return absl::InvalidArgumentError(
            "candidate ranking candidate IDs must be globally unique");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<CandidateRankerPreparedRecord> BuildCandidateRankerRecordImpl(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest,
    bool include_following_context) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  const absl::Status context_status = ValidateRecordContext(context);
  if (!context_status.ok()) {
    return context_status;
  }

  const CandidateRankerModelManifest::ScoringTemplate& scoring_template =
      manifest.scoring_template();
  const absl::StatusOr<const std::string*> mode_text =
      ModeText(context.mode, scoring_template);
  if (!mode_text.ok()) {
    return mode_text.status();
  }

  const converter::CandidateRankerSegment* target_segment = nullptr;
  const converter::CandidateRankerCandidate* target_candidate = nullptr;
  for (const converter::CandidateRankerSegment& segment : context.segments) {
    if (segment.id != segment_id) {
      continue;
    }
    target_segment = &segment;
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (candidate.id == candidate_id) {
        target_candidate = &candidate;
        break;
      }
    }
    break;
  }
  if (target_segment == nullptr || target_candidate == nullptr) {
    return absl::InvalidArgumentError(
        "candidate ranking target candidate is absent from the request");
  }

  std::vector<absl::string_view> parts;
  switch (scoring_template.record_layout()) {
    case CandidateRankerModelManifest::ScoringTemplate::
        STRUCTURED_WITH_TARGET:
      parts = {
          scoring_template.mode_prefix(),
          **mode_text,
          scoring_template.field_separator(),
          scoring_template.reading_prefix(),
          context.reading,
          scoring_template.field_separator(),
          scoring_template.segment_key_prefix(),
          target_segment->key,
          scoring_template.field_separator(),
          scoring_template.text_prefix(),
      };
      break;
    case CandidateRankerModelManifest::ScoringTemplate::
        STRUCTURED_WITHOUT_TARGET:
      parts = {
          scoring_template.mode_prefix(),
          **mode_text,
          scoring_template.field_separator(),
          scoring_template.reading_prefix(),
          context.reading,
          scoring_template.field_separator(),
          scoring_template.text_prefix(),
      };
      break;
    case CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY:
      break;
    case CandidateRankerModelManifest::ScoringTemplate::
        RECORD_LAYOUT_UNSPECIFIED:
      return absl::InvalidArgumentError(
          "scoring template record layout is invalid");
  }
  parts.reserve(parts.size() + context.segments.size() + 2);
  parts.push_back(context.preceding_text);
  for (const converter::CandidateRankerSegment& segment : context.segments) {
    if (segment.id == segment_id) {
      parts.push_back(target_candidate->value);
      if (!include_following_context) {
        break;
      }
      continue;
    }
    const converter::CandidateRankerCandidate& baseline =
        *std::min_element(
            segment.candidates.begin(), segment.candidates.end(),
            [](const converter::CandidateRankerCandidate& lhs,
               const converter::CandidateRankerCandidate& rhs) {
              return lhs.id < rhs.id;
            });
    parts.push_back(baseline.value);
  }
  if (include_following_context) {
    parts.push_back(context.following_text);
  }

  const uint64_t byte_limit = manifest.limits().per_record_byte_limit();
  uint64_t record_size = 0;
  for (const absl::string_view part : parts) {
    if (!AddRecordPartSize(part, &record_size)) {
      return absl::ResourceExhaustedError(
          "candidate ranking record byte count overflowed");
    }
  }

  CandidateRankerPreparedRecord prepared{
      .serialized_byte_count = record_size,
  };
  if (record_size > byte_limit) {
    prepared.violation = CandidateRankerCapacityViolation{
        .limit = CandidateRankerCapacityLimit::kSerializedRecordBytes,
        .observed = record_size,
        .allowed = byte_limit,
    };
    return prepared;
  }
  for (const absl::string_view part : parts) {
    if (!strings::IsValidUtf8(part)) {
      return absl::InvalidArgumentError(
          "candidate ranking record field is not valid UTF-8");
    }
  }

  std::string record;
  record.reserve(static_cast<size_t>(record_size));
  for (const absl::string_view part : parts) {
    record.append(part.data(), part.size());
  }
  absl::StatusOr<std::string> normalized =
      NormalizeRecord(std::move(record), scoring_template.text_normalization());
  if (!normalized.ok()) {
    return normalized.status();
  }
  prepared.normalized_byte_count = normalized->size();
  if (prepared.normalized_byte_count > byte_limit) {
    prepared.violation = CandidateRankerCapacityViolation{
        .limit = CandidateRankerCapacityLimit::kNormalizedRecordBytes,
        .observed = prepared.normalized_byte_count,
        .allowed = byte_limit,
    };
    return prepared;
  }
  prepared.normalized_record = std::move(*normalized);
  return prepared;
}

absl::Status CapacityViolationStatus(
    const CandidateRankerCapacityViolation& violation) {
  switch (violation.limit) {
    case CandidateRankerCapacityLimit::kSerializedRecordBytes:
      return absl::ResourceExhaustedError(
          "candidate ranking record exceeds the byte limit");
    case CandidateRankerCapacityLimit::kNormalizedRecordBytes:
      return absl::ResourceExhaustedError(
          "normalized candidate ranking record exceeds the byte limit");
    default:
      return absl::ResourceExhaustedError(
          "candidate ranking request exceeds a manifest capacity");
  }
}

}  // namespace

absl::Status ValidateCandidateRankerModelManifest(
    const CandidateRankerModelManifest& manifest) {
  if (!manifest.has_schema_version() || manifest.schema_version() != 5 ||
      !manifest.has_limits() || !manifest.has_runtime() ||
      !manifest.has_artifacts() || !manifest.has_scoring_template() ||
      !manifest.has_bounded_execution_policy()) {
    return absl::InvalidArgumentError(
        "manifest version and all manifest sections are required");
  }

  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  if (!limits.has_max_selected_candidates_per_segment() ||
      limits.max_selected_candidates_per_segment() < 2 ||
      !limits.has_max_segments_per_decode() ||
      limits.max_segments_per_decode() == 0 ||
      limits.max_segments_per_decode() > kMaximumSequenceCount ||
      !limits.has_max_sequences_per_decode() ||
      limits.max_sequences_per_decode() < 2 ||
      limits.max_sequences_per_decode() > kMaximumSequenceCount ||
      limits.max_selected_candidates_per_segment() >
          limits.max_sequences_per_decode() ||
      !limits.has_per_record_byte_limit() ||
      limits.per_record_byte_limit() == 0 ||
      limits.per_record_byte_limit() > kMaximumRecordByteLimit ||
      !limits.has_per_record_token_limit() ||
      limits.per_record_token_limit() < 2 ||
      !limits.has_per_decode_input_node_limit() ||
      limits.per_decode_input_node_limit() == 0 ||
      limits.per_decode_input_node_limit() % kContextPadding != 0 ||
      limits.per_decode_input_node_limit() >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      limits.per_record_token_limit() - 1 >
          limits.per_decode_input_node_limit() ||
      !limits.has_per_decode_output_row_limit() ||
      limits.per_decode_output_row_limit() < limits.max_sequences_per_decode() ||
      limits.per_decode_output_row_limit() >
          limits.per_decode_input_node_limit() ||
      !limits.has_max_decode_output_logit_bytes() ||
      limits.max_decode_output_logit_bytes() <
          static_cast<uint64_t>(limits.per_decode_output_row_limit()) *
              sizeof(float) ||
      !limits.has_max_request_segments() ||
      limits.max_request_segments() < limits.max_segments_per_decode() ||
      !limits.has_max_request_candidates() ||
      limits.max_request_candidates() < limits.max_sequences_per_decode() ||
      limits.max_request_candidates() <
          limits.max_selected_candidates_per_segment() ||
      !limits.has_max_request_string_bytes() ||
      limits.max_request_string_bytes() == 0) {
    return absl::InvalidArgumentError(
        "candidate sequence, record, input, output, and byte limits are "
        "invalid");
  }

  const CandidateRankerModelManifest::Runtime& runtime = manifest.runtime();
  if (!runtime.has_execution_device() ||
      runtime.execution_device() !=
          CandidateRankerModelManifest::Runtime::CPU ||
      !runtime.has_kv_storage() ||
      runtime.kv_storage() != CandidateRankerModelManifest::Runtime::UNIFIED) {
    return absl::InvalidArgumentError(
        "candidate ranking requires CPU execution and unified KV");
  }
  if (!runtime.has_micro_batch_token_capacity() ||
      runtime.micro_batch_token_capacity() == 0 ||
      runtime.micro_batch_token_capacity() >
          limits.per_decode_input_node_limit() ||
      !runtime.has_decode_thread_count() ||
      runtime.decode_thread_count() == 0 ||
      runtime.decode_thread_count() > kMaximumThreadCount ||
      !runtime.has_batch_thread_count() || runtime.batch_thread_count() == 0 ||
      runtime.batch_thread_count() > kMaximumThreadCount ||
      !runtime.has_use_memory_map() || !runtime.has_use_memory_lock() ||
      !runtime.has_check_tensors()) {
    return absl::InvalidArgumentError(
        "runtime capacities, thread counts, and flags are invalid");
  }

  const absl::Status template_status =
      ValidateScoringTemplate(manifest.scoring_template());
  if (!template_status.ok()) {
    return template_status;
  }
  const absl::Status policy_status =
      ValidateBoundedExecutionPolicy(manifest.bounded_execution_policy());
  if (!policy_status.ok()) {
    return policy_status;
  }
  return ValidateArtifacts(manifest.artifacts());
}

absl::StatusOr<CandidateRankerRequestPreflight>
PreflightCandidateRankerRequest(
    const converter::CandidateRankerRequest& request,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }

  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  CandidateRankerRequestPreflight preflight{
      .segment_count = request.segments.size(),
  };
  if (preflight.segment_count > limits.max_request_segments()) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    preflight.violation = CandidateRankerRequestLimitViolation{
        .limit = CandidateRankerRequestLimit::kSegments,
        .observed = preflight.segment_count,
        .allowed = limits.max_request_segments(),
    };
    return preflight;
  }

  const auto add_string_bytes = [&preflight](absl::string_view value)
      -> absl::Status {
    if (!AddRecordPartSize(value, &preflight.string_bytes)) {
      return absl::ResourceExhaustedError(
          "candidate ranking request string byte count overflowed");
    }
    return absl::OkStatus();
  };
  for (const absl::string_view request_text :
       {absl::string_view(request.reading),
        absl::string_view(request.preceding_text),
        absl::string_view(request.following_text)}) {
    const absl::Status status = add_string_bytes(request_text);
    if (!status.ok()) {
      return status;
    }
  }
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    if (segment.candidates.size() >
        std::numeric_limits<uint64_t>::max() - preflight.candidate_count) {
      return absl::ResourceExhaustedError(
          "candidate ranking request candidate count overflowed");
    }
    preflight.candidate_count += segment.candidates.size();
    if (preflight.candidate_count > limits.max_request_candidates()) {
      preflight.string_bytes = 0;
      continue;
    }
    const absl::Status segment_key_status = add_string_bytes(segment.key);
    if (!segment_key_status.ok()) {
      return segment_key_status;
    }
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      const absl::Status candidate_key_status =
          add_string_bytes(candidate.key);
      if (!candidate_key_status.ok()) {
        return candidate_key_status;
      }
      const absl::Status candidate_value_status =
          add_string_bytes(candidate.value);
      if (!candidate_value_status.ok()) {
        return candidate_value_status;
      }
    }
  }
  if (preflight.candidate_count > limits.max_request_candidates()) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    preflight.violation = CandidateRankerRequestLimitViolation{
        .limit = CandidateRankerRequestLimit::kCandidates,
        .observed = preflight.candidate_count,
        .allowed = limits.max_request_candidates(),
    };
    return preflight;
  }
  if (preflight.string_bytes > limits.max_request_string_bytes()) {
    preflight.violation = CandidateRankerRequestLimitViolation{
        .limit = CandidateRankerRequestLimit::kStringBytes,
        .observed = preflight.string_bytes,
        .allowed = limits.max_request_string_bytes(),
    };
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }
  return preflight;
}

absl::StatusOr<std::vector<CandidateRankerCandidateReference>>
SelectCandidateRankerCandidates(
    const converter::CandidateRankerSegment& segment,
    const CandidateRankerModelManifest& manifest) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }

  std::set<converter::CandidateRankerCandidateId> candidate_ids;
  for (const converter::CandidateRankerCandidate& candidate :
       segment.candidates) {
    if (!candidate_ids.insert(candidate.id).second) {
      return absl::InvalidArgumentError(
          "candidate ranking candidate IDs must be unique");
    }
  }

  std::vector<CandidateRankerCandidateReference> candidates;
  candidates.reserve(std::min<size_t>(
      segment.candidates.size(),
      manifest.limits().max_selected_candidates_per_segment()));
  for (size_t i = 0; i < segment.candidates.size(); ++i) {
    if (segment.candidates[i].is_protected) {
      continue;
    }
    if (candidates.size() ==
        manifest.limits().max_selected_candidates_per_segment()) {
      break;
    }
    candidates.push_back(CandidateRankerCandidateReference{
        .candidate = &segment.candidates[i],
        .original_candidate_index = i,
    });
  }
  if (candidates.size() <= 1) {
    return std::vector<CandidateRankerCandidateReference>();
  }
  return candidates;
}

absl::StatusOr<CandidateRankerRecordContext> BuildCandidateRankerRecordContext(
    const converter::CandidateRankerRequest& request,
    const CandidateRankerModelManifest& manifest) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  const absl::StatusOr<const std::string*> mode_text = ModeText(
      request.mode, manifest.scoring_template());
  if (!mode_text.ok()) {
    return mode_text.status();
  }
  CandidateRankerRecordContext context{
      .mode = request.mode,
      .preceding_text = request.preceding_text,
      .following_text = request.following_text,
      .reading = request.reading,
      .segments = request.segments,
  };
  std::sort(context.segments.begin(), context.segments.end(),
            [](const converter::CandidateRankerSegment& lhs,
               const converter::CandidateRankerSegment& rhs) {
              return lhs.id < rhs.id;
            });
  const absl::Status context_status = ValidateRecordContext(context);
  if (!context_status.ok()) {
    return context_status;
  }
  return context;
}

absl::StatusOr<std::string> BuildCandidateRankerRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest) {
  absl::StatusOr<CandidateRankerPreparedRecord> prepared =
      PrepareCandidateRankerRecord(context, segment_id, candidate_id,
                                   manifest);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (prepared->violation.has_value()) {
    return CapacityViolationStatus(*prepared->violation);
  }
  return std::move(prepared->normalized_record);
}

absl::StatusOr<CandidateRankerPreparedRecord> PrepareCandidateRankerRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest) {
  return BuildCandidateRankerRecordImpl(context, segment_id, candidate_id,
                                        manifest, true);
}

absl::StatusOr<std::string> BuildCandidateRankerCandidateEndingRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest) {
  absl::StatusOr<CandidateRankerPreparedRecord> prepared =
      PrepareCandidateRankerCandidateEndingRecord(
          context, segment_id, candidate_id, manifest);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (prepared->violation.has_value()) {
    return CapacityViolationStatus(*prepared->violation);
  }
  return std::move(prepared->normalized_record);
}

absl::StatusOr<CandidateRankerPreparedRecord>
PrepareCandidateRankerCandidateEndingRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest) {
  return BuildCandidateRankerRecordImpl(context, segment_id, candidate_id,
                                        manifest, false);
}

namespace {

absl::StatusOr<CandidateRankerBatchPlan> BuildCandidateRankerBatchPlanImpl(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation,
    bool enforce_capacity) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }
  if (enforce_capacity &&
      segments.size() > manifest.limits().max_segments_per_decode()) {
    return absl::ResourceExhaustedError(
        "rankable segment count exceeds the manifest limit");
  }

  std::vector<const CandidateRankerTokenizedSegment*> ordered_segments;
  ordered_segments.reserve(segments.size());
  for (const CandidateRankerTokenizedSegment& segment : segments) {
    ordered_segments.push_back(&segment);
  }
  std::sort(ordered_segments.begin(), ordered_segments.end(),
            [](const CandidateRankerTokenizedSegment* lhs,
               const CandidateRankerTokenizedSegment* rhs) {
              return lhs->segment_id < rhs->segment_id;
            });
  for (size_t i = 1; i < ordered_segments.size(); ++i) {
    if (ordered_segments[i - 1]->segment_id ==
        ordered_segments[i]->segment_id) {
      return absl::InvalidArgumentError(
          "candidate ranking segment IDs must be unique");
    }
  }

  CandidateRankerBatchPlan plan;
  plan.segments.reserve(ordered_segments.size());
  std::set<converter::CandidateRankerCandidateId> candidate_ids;
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const int32_t bos_token_id =
      static_cast<int32_t>(manifest.scoring_template().bos_token_id());
  const int32_t terminal_token_id =
      static_cast<int32_t>(manifest.scoring_template().terminal_token_id());

  for (const CandidateRankerTokenizedSegment* segment : ordered_segments) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    if (segment->candidates.size() < 2) {
      return absl::InvalidArgumentError(
          "a tokenized rankable segment must contain at least two candidates");
    }
    if (enforce_capacity &&
        segment->candidates.size() > limits.max_selected_candidates_per_segment()) {
      return absl::ResourceExhaustedError(
          "rankable candidate count exceeds the manifest limit");
    }
    if (enforce_capacity && segment->candidates.size() >
        limits.max_sequences_per_decode() - plan.sequence_count) {
      return absl::ResourceExhaustedError(
          "candidate sequence count exceeds the manifest limit");
    }

    std::vector<const CandidateRankerTokenSequence*> ordered_candidates;
    ordered_candidates.reserve(segment->candidates.size());
    for (const CandidateRankerTokenSequence& candidate : segment->candidates) {
      ordered_candidates.push_back(&candidate);
    }
    std::sort(ordered_candidates.begin(), ordered_candidates.end(),
              [](const CandidateRankerTokenSequence* lhs,
                 const CandidateRankerTokenSequence* rhs) {
                if (ByteLess(lhs->candidate_value, rhs->candidate_value)) {
                  return true;
                }
                if (ByteLess(rhs->candidate_value, lhs->candidate_value)) {
                  return false;
                }
                return lhs->candidate_id < rhs->candidate_id;
              });

    std::set<size_t> original_candidate_indices;
    for (const CandidateRankerTokenSequence* candidate : ordered_candidates) {
      if (!candidate_ids.insert(candidate->candidate_id).second) {
        return absl::InvalidArgumentError(
            "candidate ranking candidate IDs must be globally unique");
      }
      if (!original_candidate_indices
               .insert(candidate->original_candidate_index)
               .second) {
        return absl::InvalidArgumentError(
            "candidate ranking original positions must be unique per segment");
      }
      if (candidate->tokens.empty() ||
          candidate->tokens.front() != bos_token_id) {
        return absl::InvalidArgumentError(
            "candidate ranking token sequences must begin with the manifest "
            "BOS");
      }
      if (enforce_capacity &&
          candidate->tokens.size() >= limits.per_record_token_limit()) {
        return absl::ResourceExhaustedError(
            "candidate ranking record exceeds the token limit");
      }
      for (const int32_t token_id : candidate->tokens) {
        if (token_id < 0) {
          return absl::InvalidArgumentError(
              "candidate ranking input token IDs must be nonnegative");
        }
      }
    }

    std::vector<CanonicalSequence> canonical_sequences;
    canonical_sequences.reserve(ordered_candidates.size());
    CandidateRankerSegmentPlan segment_plan{.segment_id = segment->segment_id};
    segment_plan.candidates.reserve(ordered_candidates.size());
    for (const CandidateRankerTokenSequence* candidate : ordered_candidates) {
      const int32_t sequence_id = static_cast<int32_t>(plan.sequence_count++);
      canonical_sequences.push_back(CanonicalSequence{
          .sequence = candidate,
          .sequence_id = sequence_id,
      });
      segment_plan.candidates.push_back(CandidateRankerPlannedCandidate{
          .candidate_id = candidate->candidate_id,
          .original_candidate_index = candidate->original_candidate_index,
          .sequence_id = sequence_id,
      });
    }
    std::sort(segment_plan.candidates.begin(), segment_plan.candidates.end(),
              [](const CandidateRankerPlannedCandidate& lhs,
                 const CandidateRankerPlannedCandidate& rhs) {
                return lhs.original_candidate_index <
                       rhs.original_candidate_index;
              });

    size_t common_prefix_size =
        canonical_sequences.front().sequence->tokens.size();
    for (size_t i = 1; i < canonical_sequences.size(); ++i) {
      const std::vector<int32_t>& tokens =
          canonical_sequences[i].sequence->tokens;
      common_prefix_size = std::min(common_prefix_size, tokens.size());
      size_t position = 0;
      while (position < common_prefix_size &&
             canonical_sequences.front().sequence->tokens[position] ==
                 tokens[position]) {
        ++position;
      }
      common_prefix_size = position;
    }
    if (common_prefix_size == 0) {
      return absl::InvalidArgumentError(
          "candidate ranking sequences do not share the manifest BOS");
    }
    segment_plan.common_prefix_token_count =
        static_cast<uint32_t>(common_prefix_size);

    if (enforce_capacity &&
        plan.inputs.size() >= limits.per_decode_input_node_limit()) {
      return absl::ResourceExhaustedError(
          "candidate ranking trie exceeds the input-node limit");
    }
    std::vector<MutableTrieNode> nodes;
    nodes.push_back(MutableTrieNode{
        .token_id = bos_token_id,
        .position = 0,
    });
    for (const CanonicalSequence& branch : canonical_sequences) {
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      size_t node_index = 0;
      nodes[node_index].sequence_ids.push_back(branch.sequence_id);
      for (size_t position = 1; position < branch.sequence->tokens.size();
           ++position) {
        if (cancellation.IsCancellationRequested()) {
          return CancelledStatus();
        }
        const int32_t token_id = branch.sequence->tokens[position];
        auto child = nodes[node_index].children.find(token_id);
        if (child == nodes[node_index].children.end()) {
          if (enforce_capacity && plan.inputs.size() + nodes.size() >=
              limits.per_decode_input_node_limit()) {
            return absl::ResourceExhaustedError(
                "candidate ranking trie exceeds the input-node limit");
          }
          const size_t child_index = nodes.size();
          nodes.push_back(MutableTrieNode{
              .token_id = token_id,
              .position = static_cast<int32_t>(position),
          });
          child =
              nodes[node_index].children.emplace(token_id, child_index).first;
        }
        node_index = child->second;
        nodes[node_index].sequence_ids.push_back(branch.sequence_id);
      }
      nodes[node_index].terminal_sequence_ids.push_back(branch.sequence_id);
    }

    std::vector<size_t> node_order = {0};
    for (size_t i = 0; i < node_order.size(); ++i) {
      for (const auto& [token_id, child_index] :
           nodes[node_order[i]].children) {
        static_cast<void>(token_id);
        node_order.push_back(child_index);
      }
    }
    for (const size_t node_index : node_order) {
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      const MutableTrieNode& node = nodes[node_index];
      std::map<int32_t, std::vector<int32_t>> scored_descendants;
      for (const auto& [token_id, child_index] : node.children) {
        const MutableTrieNode& child = nodes[child_index];
        if (static_cast<size_t>(child.position) >= common_prefix_size) {
          scored_descendants[token_id] = child.sequence_ids;
        }
      }
      if (!node.terminal_sequence_ids.empty()) {
        std::vector<int32_t>& terminal_descendants =
            scored_descendants[terminal_token_id];
        terminal_descendants.insert(terminal_descendants.end(),
                                    node.terminal_sequence_ids.begin(),
                                    node.terminal_sequence_ids.end());
      }

      CandidateRankerBatchInput input{
          .token_id = node.token_id,
          .position = node.position,
          .sequence_ids = node.sequence_ids,
      };
      input.scored_edges.reserve(scored_descendants.size());
      for (auto& [target_token_id, descendants] : scored_descendants) {
        std::sort(descendants.begin(), descendants.end());
        descendants.erase(std::unique(descendants.begin(), descendants.end()),
                          descendants.end());
        input.scored_edges.push_back(CandidateRankerScoredEdge{
            .target_token_id = target_token_id,
            .descendant_sequence_ids = std::move(descendants),
        });
      }
      if (!input.scored_edges.empty()) {
        if (enforce_capacity &&
            plan.output_row_count == limits.per_decode_output_row_limit()) {
          return absl::ResourceExhaustedError(
              "candidate ranking trie exceeds the output-row limit");
        }
        ++plan.output_row_count;
      }
      plan.inputs.push_back(std::move(input));
    }
    plan.segments.push_back(std::move(segment_plan));
  }
  return plan;
}

}  // namespace

absl::StatusOr<CandidateRankerBatchPlan> BuildCandidateRankerBatchPlan(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation) {
  return BuildCandidateRankerBatchPlanImpl(segments, manifest, cancellation,
                                           true);
}

absl::StatusOr<CandidateRankerBatchPlan>
BuildCandidateRankerBatchPlanForCapacityAudit(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation) {
  return BuildCandidateRankerBatchPlanImpl(segments, manifest, cancellation,
                                           false);
}

absl::StatusOr<uint64_t> ComputeCandidateRankerOutputLogitBytes(
    uint64_t output_rows, uint64_t reserved_output_rows,
    uint64_t vocabulary_size) {
  if (vocabulary_size == 0) {
    return absl::InvalidArgumentError(
        "candidate ranking vocabulary size must be positive");
  }
  const uint64_t rows = std::max(output_rows, reserved_output_rows);
  if (rows > std::numeric_limits<uint64_t>::max() / vocabulary_size ||
      rows * vocabulary_size >
          std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return absl::InvalidArgumentError(
        "candidate ranking output-logit size overflowed");
  }
  return rows * vocabulary_size * sizeof(float);
}

absl::StatusOr<std::optional<CandidateRankerCapacityViolation>>
EvaluateCandidateRankerDecodeCapacity(
    CandidateRankerDecodeCapacityUsage usage,
    const CandidateRankerModelManifest& manifest) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  const CandidateRankerModelManifest::Limits& limits = manifest.limits();
  const auto exceeds = [](CandidateRankerCapacityLimit limit,
                          uint64_t observed, uint64_t allowed)
      -> std::optional<CandidateRankerCapacityViolation> {
    if (observed <= allowed) {
      return std::nullopt;
    }
    return CandidateRankerCapacityViolation{
        .limit = limit,
        .observed = observed,
        .allowed = allowed,
    };
  };

  std::optional<CandidateRankerCapacityViolation> violation;
  if ((violation = exceeds(CandidateRankerCapacityLimit::kSegmentsPerDecode,
                           usage.segments,
                           limits.max_segments_per_decode()))
          .has_value() ||
      (violation = exceeds(CandidateRankerCapacityLimit::kSequencesPerDecode,
                           usage.sequences,
                           limits.max_sequences_per_decode()))
          .has_value() ||
      (violation = exceeds(CandidateRankerCapacityLimit::kInputNodesPerDecode,
                           usage.input_nodes,
                           limits.per_decode_input_node_limit()))
          .has_value() ||
      (violation = exceeds(CandidateRankerCapacityLimit::kOutputRowsPerDecode,
                           usage.output_rows,
                           limits.per_decode_output_row_limit()))
          .has_value() ||
      (violation = exceeds(
           CandidateRankerCapacityLimit::kReservedOutputRowsPerDecode,
           usage.reserved_output_rows,
           limits.per_decode_output_row_limit()))
          .has_value() ||
      (violation = exceeds(
           CandidateRankerCapacityLimit::kOutputLogitBytesPerDecode,
           usage.output_logit_bytes,
           limits.max_decode_output_logit_bytes()))
          .has_value()) {
    return violation;
  }
  return std::nullopt;
}

absl::StatusOr<std::vector<CandidateRankerDecodeBatch>>
PackCandidateRankerSegmentsForDecode(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    uint64_t vocabulary_size, const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation) {
  const absl::Status manifest_status =
      ValidateCandidateRankerModelManifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }

  std::vector<CandidateRankerTokenizedSegment> ordered_segments(
      segments.begin(), segments.end());
  std::sort(ordered_segments.begin(), ordered_segments.end(),
            [](const CandidateRankerTokenizedSegment& lhs,
               const CandidateRankerTokenizedSegment& rhs) {
              return lhs.segment_id < rhs.segment_id;
            });
  std::set<converter::CandidateRankerSegmentId> segment_ids;
  std::set<converter::CandidateRankerCandidateId> candidate_ids;
  for (const CandidateRankerTokenizedSegment& segment : ordered_segments) {
    if (!segment_ids.insert(segment.segment_id).second) {
      return absl::InvalidArgumentError(
          "candidate ranking segment IDs must be unique");
    }
    for (const CandidateRankerTokenSequence& candidate : segment.candidates) {
      if (!candidate_ids.insert(candidate.candidate_id).second) {
        return absl::InvalidArgumentError(
            "candidate ranking candidate IDs must be globally unique");
      }
    }
  }

  const auto build_usage =
      [&manifest, vocabulary_size](const CandidateRankerBatchPlan& plan)
      -> absl::StatusOr<CandidateRankerDecodeCapacityUsage> {
    CandidateRankerDecodeCapacityUsage usage{
        .segments = plan.segments.size(),
        .sequences = plan.sequence_count,
        .input_nodes = plan.inputs.size(),
        .output_rows = plan.output_row_count,
        .reserved_output_rows = std::max<uint64_t>(
            plan.output_row_count,
            manifest.limits().max_sequences_per_decode()),
    };
    absl::StatusOr<uint64_t> output_logit_bytes =
        ComputeCandidateRankerOutputLogitBytes(
            usage.output_rows, usage.reserved_output_rows, vocabulary_size);
    if (!output_logit_bytes.ok()) {
      return output_logit_bytes.status();
    }
    usage.output_logit_bytes = *output_logit_bytes;
    return usage;
  };
  const auto release_candidate_values =
      [](std::vector<CandidateRankerTokenizedSegment>* batch_segments) {
        for (CandidateRankerTokenizedSegment& segment : *batch_segments) {
          for (CandidateRankerTokenSequence& candidate : segment.candidates) {
            candidate.candidate_value = absl::string_view();
          }
        }
      };

  std::vector<CandidateRankerDecodeBatch> batches;
  std::vector<CandidateRankerTokenizedSegment> current_segments;
  CandidateRankerBatchPlan current_plan;
  CandidateRankerDecodeCapacityUsage current_usage;
  for (const CandidateRankerTokenizedSegment& segment : ordered_segments) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    current_segments.push_back(segment);
    absl::StatusOr<CandidateRankerBatchPlan> trial_plan =
        BuildCandidateRankerBatchPlanForCapacityAudit(
            current_segments, manifest, cancellation);
    if (!trial_plan.ok()) {
      return trial_plan.status();
    }
    absl::StatusOr<CandidateRankerDecodeCapacityUsage> trial_usage =
        build_usage(*trial_plan);
    if (!trial_usage.ok()) {
      return trial_usage.status();
    }
    absl::StatusOr<std::optional<CandidateRankerCapacityViolation>> violation =
        EvaluateCandidateRankerDecodeCapacity(*trial_usage, manifest);
    if (!violation.ok()) {
      return violation.status();
    }
    if (violation->has_value() && current_segments.size() > 1) {
      current_segments.pop_back();
      release_candidate_values(&current_segments);
      batches.push_back(CandidateRankerDecodeBatch{
          .tokenized_segments = std::move(current_segments),
          .plan = std::move(current_plan),
          .usage = current_usage,
      });
      current_segments = {segment};
      trial_plan = BuildCandidateRankerBatchPlanForCapacityAudit(
          current_segments, manifest, cancellation);
      if (!trial_plan.ok()) {
        return trial_plan.status();
      }
      trial_usage = build_usage(*trial_plan);
      if (!trial_usage.ok()) {
        return trial_usage.status();
      }
      violation = EvaluateCandidateRankerDecodeCapacity(*trial_usage, manifest);
      if (!violation.ok()) {
        return violation.status();
      }
    }
    if (violation->has_value()) {
      return absl::ResourceExhaustedError(
          "candidate ranking segment exceeds per-decode capacities");
    }
    current_plan = std::move(*trial_plan);
    current_usage = *trial_usage;
  }
  if (!current_segments.empty()) {
    release_candidate_values(&current_segments);
    batches.push_back(CandidateRankerDecodeBatch{
        .tokenized_segments = std::move(current_segments),
        .plan = std::move(current_plan),
        .usage = current_usage,
    });
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }
  return batches;
}

absl::StatusOr<uint32_t> FindCandidateRankerFollowingContextStartPosition(
    absl::Span<const int32_t> complete_tokens,
    absl::Span<const int32_t> candidate_prefix_tokens) {
  if (complete_tokens.empty() || candidate_prefix_tokens.empty() ||
      complete_tokens.front() != candidate_prefix_tokens.front()) {
    return absl::InvalidArgumentError(
        "candidate ranking boundary token sequences are invalid");
  }
  size_t common_prefix_size =
      std::min(complete_tokens.size(), candidate_prefix_tokens.size());
  size_t position = 0;
  while (position < common_prefix_size &&
         complete_tokens[position] == candidate_prefix_tokens[position]) {
    ++position;
  }
  if (position < complete_tokens.size() &&
      position < candidate_prefix_tokens.size()) {
    ++position;
  }
  return static_cast<uint32_t>(position);
}

absl::StatusOr<std::vector<double>>
ComputeCandidateRankerEdgeLogProbabilities(
    const CandidateRankerBatchInput& input, absl::Span<const float> logits,
    const converter::CandidateRankerCancellation* cancellation) {
  if (input.scored_edges.empty() || logits.empty()) {
    return absl::InvalidArgumentError(
        "candidate ranking output row and logits must be nonempty");
  }

  double maximum_logit = -std::numeric_limits<double>::infinity();
  for (const float logit : logits) {
    if (cancellation != nullptr &&
        cancellation->IsCancellationRequested()) {
      return CancelledStatus();
    }
    if (!std::isfinite(logit)) {
      return absl::InvalidArgumentError(
          "candidate ranking logits must be finite");
    }
    maximum_logit = std::max(maximum_logit, static_cast<double>(logit));
  }
  double exponential_sum = 0.0;
  for (const float logit : logits) {
    if (cancellation != nullptr &&
        cancellation->IsCancellationRequested()) {
      return CancelledStatus();
    }
    exponential_sum += std::exp(static_cast<double>(logit) - maximum_logit);
  }
  const double normalizer = maximum_logit + std::log(exponential_sum);
  if (!std::isfinite(normalizer)) {
    return absl::InvalidArgumentError(
        "candidate ranking logit normalizer must be finite");
  }

  std::vector<double> log_probabilities;
  log_probabilities.reserve(input.scored_edges.size());
  for (const CandidateRankerScoredEdge& edge : input.scored_edges) {
    if (cancellation != nullptr &&
        cancellation->IsCancellationRequested()) {
      return CancelledStatus();
    }
    if (edge.target_token_id < 0 ||
        static_cast<size_t>(edge.target_token_id) >= logits.size() ||
        edge.descendant_sequence_ids.empty()) {
      return absl::InvalidArgumentError(
          "candidate ranking scored edge is invalid");
    }
    const double log_probability =
        static_cast<double>(logits[edge.target_token_id]) - normalizer;
    if (!std::isfinite(log_probability)) {
      return absl::InvalidArgumentError(
          "candidate ranking log probability must be finite");
    }
    log_probabilities.push_back(log_probability);
  }
  return log_probabilities;
}

absl::Status AccumulateCandidateRankerEdgeLogProbabilities(
    const CandidateRankerBatchInput& input,
    absl::Span<const double> edge_log_probabilities,
    absl::Span<double> sequence_scores) {
  if (input.scored_edges.empty() ||
      edge_log_probabilities.size() != input.scored_edges.size()) {
    return absl::InvalidArgumentError(
        "candidate ranking scored edges and log probabilities must match");
  }
  for (const double score : sequence_scores) {
    if (!std::isfinite(score)) {
      return absl::InvalidArgumentError(
          "candidate ranking sequence scores must be finite");
    }
  }

  std::vector<double> deltas(sequence_scores.size(), 0.0);
  std::vector<bool> touched(sequence_scores.size(), false);
  for (size_t edge_index = 0; edge_index < input.scored_edges.size();
       ++edge_index) {
    const CandidateRankerScoredEdge& edge = input.scored_edges[edge_index];
    const double log_probability = edge_log_probabilities[edge_index];
    if (edge.descendant_sequence_ids.empty() ||
        !std::isfinite(log_probability)) {
      return absl::InvalidArgumentError(
          "candidate ranking scored edge log probability is invalid");
    }
    for (const int32_t sequence_id : edge.descendant_sequence_ids) {
      if (sequence_id < 0 ||
          static_cast<size_t>(sequence_id) >= sequence_scores.size() ||
          touched[sequence_id]) {
        return absl::InvalidArgumentError(
            "candidate ranking edge descendants are invalid");
      }
      touched[sequence_id] = true;
      deltas[sequence_id] = log_probability;
    }
  }
  for (size_t i = 0; i < sequence_scores.size(); ++i) {
    if (!touched[i]) {
      continue;
    }
    const double updated_score = sequence_scores[i] + deltas[i];
    if (!std::isfinite(updated_score)) {
      return absl::InvalidArgumentError(
          "candidate ranking accumulated score must be finite");
    }
    sequence_scores[i] = updated_score;
  }
  return absl::OkStatus();
}

absl::Status AccumulateCandidateRankerOutput(
    const CandidateRankerBatchInput& input, absl::Span<const float> logits,
    absl::Span<double> sequence_scores) {
  absl::StatusOr<std::vector<double>> edge_log_probabilities =
      ComputeCandidateRankerEdgeLogProbabilities(input, logits);
  if (!edge_log_probabilities.ok()) {
    return edge_log_probabilities.status();
  }
  return AccumulateCandidateRankerEdgeLogProbabilities(
      input, *edge_log_probabilities, sequence_scores);
}

absl::StatusOr<std::vector<converter::CandidateRankerSegmentOrder>>
OrderCandidateRankerContinuations(const CandidateRankerBatchPlan& plan,
                                  absl::Span<const double> sequence_scores) {
  if (sequence_scores.size() != plan.sequence_count) {
    return absl::InvalidArgumentError(
        "candidate ranking sequence and score counts must match");
  }
  for (const double score : sequence_scores) {
    if (!std::isfinite(score)) {
      return absl::InvalidArgumentError(
          "candidate ranking sequence scores must be finite");
    }
  }

  std::vector<converter::CandidateRankerSegmentOrder> orders;
  orders.reserve(plan.segments.size());
  for (const CandidateRankerSegmentPlan& segment : plan.segments) {
    std::vector<const CandidateRankerPlannedCandidate*> candidates;
    candidates.reserve(segment.candidates.size());
    for (const CandidateRankerPlannedCandidate& candidate :
         segment.candidates) {
      if (candidate.sequence_id < 0 ||
          static_cast<size_t>(candidate.sequence_id) >=
              sequence_scores.size()) {
        return absl::InvalidArgumentError(
            "candidate ranking planned sequence ID is invalid");
      }
      candidates.push_back(&candidate);
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&sequence_scores](const CandidateRankerPlannedCandidate* lhs,
                           const CandidateRankerPlannedCandidate* rhs) {
          return sequence_scores[lhs->sequence_id] >
                 sequence_scores[rhs->sequence_id];
        });

    converter::CandidateRankerSegmentOrder order{
        .segment_id = segment.segment_id,
    };
    order.candidate_ids.reserve(candidates.size());
    for (const CandidateRankerPlannedCandidate* candidate : candidates) {
      order.candidate_ids.push_back(candidate->candidate_id);
    }
    orders.push_back(std::move(order));
  }
  return orders;
}

}  // namespace engine
}  // namespace mozc
