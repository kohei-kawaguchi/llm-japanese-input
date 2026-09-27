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

#include "engine/evaluation/ajimee_capacity_audit.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"

namespace mozc::engine::evaluation {
namespace {

struct SegmentFacts {
  uint64_t segment_id = 0;
  uint64_t unprotected_candidate_count = 0;
};

struct RequestFacts {
  uint64_t segment_count = 0;
  uint64_t candidate_count = 0;
  uint64_t unprotected_candidate_count = 0;
  uint64_t rankable_segment_count = 0;
  uint64_t string_bytes = 0;
  uint64_t max_unprotected_candidates_in_segment = 0;
  std::vector<SegmentFacts> segments;
};

bool IsSimpleFileName(absl::string_view value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find('/') == absl::string_view::npos &&
         value.find('\\') == absl::string_view::npos &&
         value.find(':') == absl::string_view::npos &&
         value.find('\0') == absl::string_view::npos;
}

absl::Status AddCount(uint64_t value, uint64_t* total) {
  if (value > std::numeric_limits<uint64_t>::max() - *total) {
    return absl::OutOfRangeError("AJIMEE request count overflows uint64");
  }
  *total += value;
  return absl::OkStatus();
}

absl::StatusOr<RequestFacts> BuildRequestFacts(
    const FrozenCandidateRankerRequest& request) {
  RequestFacts facts;
  facts.segment_count = request.segments_size();
  facts.segments.reserve(request.segments_size());
  absl::Status status = AddCount(request.preceding_text().size(),
                                 &facts.string_bytes);
  if (!status.ok()) {
    return status;
  }
  status = AddCount(request.following_text().size(), &facts.string_bytes);
  if (!status.ok()) {
    return status;
  }
  status = AddCount(request.reading().size(), &facts.string_bytes);
  if (!status.ok()) {
    return status;
  }
  for (const FrozenCandidateRankerSegment& segment : request.segments()) {
    status = AddCount(segment.candidates_size(), &facts.candidate_count);
    if (!status.ok()) {
      return status;
    }
    status = AddCount(segment.key().size(), &facts.string_bytes);
    if (!status.ok()) {
      return status;
    }
    uint64_t unprotected_candidate_count = 0;
    for (const FrozenCandidateRankerCandidate& candidate :
         segment.candidates()) {
      status = AddCount(candidate.key().size(), &facts.string_bytes);
      if (!status.ok()) {
        return status;
      }
      status = AddCount(candidate.value().size(), &facts.string_bytes);
      if (!status.ok()) {
        return status;
      }
      if (!candidate.is_protected()) {
        ++unprotected_candidate_count;
      }
    }
    facts.unprotected_candidate_count += unprotected_candidate_count;
    if (unprotected_candidate_count > 1) {
      ++facts.rankable_segment_count;
    }
    facts.max_unprotected_candidates_in_segment =
        std::max(facts.max_unprotected_candidates_in_segment,
                 unprotected_candidate_count);
    facts.segments.push_back(SegmentFacts{
        .segment_id = segment.id(),
        .unprotected_candidate_count = unprotected_candidate_count,
    });
  }
  return facts;
}

bool IdentityMatchesConfig(const AjimeeCapacityAuditIdentity& identity,
                           const AjimeeCapacityAuditConfig& config) {
  return identity.frozen_corpus_sha256() == config.frozen_corpus_sha256() &&
         identity.model_manifest_sha256() == config.model_manifest_sha256() &&
         identity.gguf_file_name() == config.gguf_file_name() &&
         identity.gguf_sha256() == config.gguf_sha256() &&
         identity.tokenizer_sha256() == config.tokenizer_sha256() &&
         identity.quantization() == config.quantization() &&
         identity.source_revision() == config.source_revision() &&
         identity.runtime_revision() == config.runtime_revision();
}

absl::Status ValidateIdentity(
    const AjimeeCapacityAuditIdentity& identity) {
  if (!identity.IsInitialized() || HasUnknownFieldsRecursively(identity) ||
      !IsAjimeeLowercaseHex(identity.frozen_corpus_sha256(), 64) ||
      !IsAjimeeLowercaseHex(identity.model_manifest_sha256(), 64) ||
      !IsSimpleFileName(identity.gguf_file_name()) ||
      !IsAjimeeLowercaseHex(identity.gguf_sha256(), 64) ||
      !IsAjimeeLowercaseHex(identity.tokenizer_sha256(), 64) ||
      identity.quantization().empty() ||
      !IsAjimeeLowercaseHex(identity.source_revision(), 40) ||
      !IsAjimeeLowercaseHex(identity.runtime_revision(), 40)) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit identity is invalid");
  }
  return absl::OkStatus();
}

absl::StatusOr<AjimeeCanonicalStatusCode> ConvertStatusCode(
    absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kOk:
      return AJIMEE_STATUS_OK;
    case absl::StatusCode::kCancelled:
      return AJIMEE_STATUS_CANCELLED;
    case absl::StatusCode::kUnknown:
      return AJIMEE_STATUS_UNKNOWN;
    case absl::StatusCode::kInvalidArgument:
      return AJIMEE_STATUS_INVALID_ARGUMENT;
    case absl::StatusCode::kDeadlineExceeded:
      return AJIMEE_STATUS_DEADLINE_EXCEEDED;
    case absl::StatusCode::kNotFound:
      return AJIMEE_STATUS_NOT_FOUND;
    case absl::StatusCode::kAlreadyExists:
      return AJIMEE_STATUS_ALREADY_EXISTS;
    case absl::StatusCode::kPermissionDenied:
      return AJIMEE_STATUS_PERMISSION_DENIED;
    case absl::StatusCode::kResourceExhausted:
      return AJIMEE_STATUS_RESOURCE_EXHAUSTED;
    case absl::StatusCode::kFailedPrecondition:
      return AJIMEE_STATUS_FAILED_PRECONDITION;
    case absl::StatusCode::kAborted:
      return AJIMEE_STATUS_ABORTED;
    case absl::StatusCode::kOutOfRange:
      return AJIMEE_STATUS_OUT_OF_RANGE;
    case absl::StatusCode::kUnimplemented:
      return AJIMEE_STATUS_UNIMPLEMENTED;
    case absl::StatusCode::kInternal:
      return AJIMEE_STATUS_INTERNAL;
    case absl::StatusCode::kUnavailable:
      return AJIMEE_STATUS_UNAVAILABLE;
    case absl::StatusCode::kDataLoss:
      return AJIMEE_STATUS_DATA_LOSS;
    case absl::StatusCode::kUnauthenticated:
      return AJIMEE_STATUS_UNAUTHENTICATED;
    default:
      return absl::InvalidArgumentError(
          "AJIMEE canonical status code is invalid");
  }
}

bool IsKnownCanonicalStatusCode(AjimeeCanonicalStatusCode code) {
  switch (code) {
    case AJIMEE_STATUS_OK:
    case AJIMEE_STATUS_CANCELLED:
    case AJIMEE_STATUS_UNKNOWN:
    case AJIMEE_STATUS_INVALID_ARGUMENT:
    case AJIMEE_STATUS_DEADLINE_EXCEEDED:
    case AJIMEE_STATUS_NOT_FOUND:
    case AJIMEE_STATUS_ALREADY_EXISTS:
    case AJIMEE_STATUS_PERMISSION_DENIED:
    case AJIMEE_STATUS_RESOURCE_EXHAUSTED:
    case AJIMEE_STATUS_FAILED_PRECONDITION:
    case AJIMEE_STATUS_ABORTED:
    case AJIMEE_STATUS_OUT_OF_RANGE:
    case AJIMEE_STATUS_UNIMPLEMENTED:
    case AJIMEE_STATUS_INTERNAL:
    case AJIMEE_STATUS_UNAVAILABLE:
    case AJIMEE_STATUS_DATA_LOSS:
    case AJIMEE_STATUS_UNAUTHENTICATED:
      return true;
  }
  return false;
}

absl::StatusOr<AjimeeRequestLimit> ConvertRequestLimit(
    CandidateRankerRequestLimit limit) {
  switch (limit) {
    case CandidateRankerRequestLimit::kSegments:
      return AJIMEE_REQUEST_SEGMENTS;
    case CandidateRankerRequestLimit::kCandidates:
      return AJIMEE_REQUEST_CANDIDATES;
    case CandidateRankerRequestLimit::kStringBytes:
      return AJIMEE_REQUEST_STRING_BYTES;
  }
  return absl::InvalidArgumentError("AJIMEE request limit is invalid");
}

absl::StatusOr<AjimeeCapacityLimit> ConvertCapacityLimit(
    CandidateRankerCapacityLimit limit) {
  switch (limit) {
    case CandidateRankerCapacityLimit::kSerializedRecordBytes:
      return AJIMEE_SERIALIZED_RECORD_BYTES;
    case CandidateRankerCapacityLimit::kNormalizedRecordBytes:
      return AJIMEE_NORMALIZED_RECORD_BYTES;
    case CandidateRankerCapacityLimit::kFullRecordTokens:
      return AJIMEE_FULL_RECORD_TOKENS;
    case CandidateRankerCapacityLimit::kSegmentsPerDecode:
      return AJIMEE_SEGMENTS_PER_DECODE;
    case CandidateRankerCapacityLimit::kSequencesPerDecode:
      return AJIMEE_SEQUENCES_PER_DECODE;
    case CandidateRankerCapacityLimit::kInputNodesPerDecode:
      return AJIMEE_INPUT_NODES_PER_DECODE;
    case CandidateRankerCapacityLimit::kOutputRowsPerDecode:
      return AJIMEE_OUTPUT_ROWS_PER_DECODE;
    case CandidateRankerCapacityLimit::kReservedOutputRowsPerDecode:
      return AJIMEE_RESERVED_OUTPUT_ROWS_PER_DECODE;
    case CandidateRankerCapacityLimit::kOutputLogitBytesPerDecode:
      return AJIMEE_OUTPUT_LOGIT_BYTES_PER_DECODE;
  }
  return absl::InvalidArgumentError("AJIMEE capacity limit is invalid");
}

absl::StatusOr<AjimeeSegmentCapacityDisposition> ConvertDisposition(
    CandidateRankerSegmentCapacityDisposition disposition) {
  switch (disposition) {
    case CandidateRankerSegmentCapacityDisposition::kSelected:
      return AJIMEE_SEGMENT_CAPACITY_SELECTED;
    case CandidateRankerSegmentCapacityDisposition::kOmittedCapacity:
      return AJIMEE_SEGMENT_CAPACITY_OMITTED_CAPACITY;
  }
  return absl::InvalidArgumentError(
      "AJIMEE segment capacity disposition is invalid");
}

void CopyUsage(const CandidateRankerCapacityUsage& source,
               AjimeeCapacityAuditUsage* destination) {
  destination->set_request_segment_count(source.request_segment_count);
  destination->set_request_candidate_count(source.request_candidate_count);
  destination->set_request_unprotected_candidate_count(
      source.request_unprotected_candidate_count);
  destination->set_request_rankable_segment_count(
      source.request_rankable_segment_count);
  destination->set_request_string_bytes(source.request_string_bytes);
  destination->set_max_unprotected_candidates_in_segment(
      source.max_unprotected_candidates_in_segment);
  destination->set_window_candidate_count(source.window_candidate_count);
  destination->set_candidates_omitted_by_window(
      source.candidates_omitted_by_window);
  destination->set_selected_segment_count(source.selected_segment_count);
  destination->set_selected_candidate_count(source.selected_candidate_count);
  destination->set_candidates_omitted_by_capacity(
      source.candidates_omitted_by_capacity);
  destination->set_segments_omitted_by_capacity(
      source.segments_omitted_by_capacity);
  destination->set_decode_batch_count(source.decode_batch_count);
  destination->set_max_selected_serialized_record_bytes(
      source.max_selected_serialized_record_bytes);
  destination->set_max_selected_normalized_record_bytes(
      source.max_selected_normalized_record_bytes);
  destination->set_max_selected_full_record_tokens(
      source.max_selected_full_record_tokens);
  destination->set_max_segments_in_decode(source.max_segments_in_decode);
  destination->set_max_sequences_in_decode(source.max_sequences_in_decode);
  destination->set_max_input_nodes_in_decode(source.max_input_nodes_in_decode);
  destination->set_max_output_rows_in_decode(source.max_output_rows_in_decode);
  destination->set_max_reserved_output_rows_in_decode(
      source.max_reserved_output_rows_in_decode);
  destination->set_max_output_logit_bytes_in_decode(
      source.max_output_logit_bytes_in_decode);
}

absl::Status CopyRequestLimitViolation(
    const CandidateRankerRequestLimitViolation& source,
    AjimeeRequestLimitViolation* destination) {
  absl::StatusOr<AjimeeRequestLimit> limit = ConvertRequestLimit(source.limit);
  if (!limit.ok()) {
    return limit.status();
  }
  destination->set_limit(*limit);
  destination->set_observed(source.observed);
  destination->set_allowed(source.allowed);
  return absl::OkStatus();
}

absl::Status CopyCapacityViolation(
    const CandidateRankerCapacityViolation& source,
    AjimeeCapacityViolation* destination) {
  absl::StatusOr<AjimeeCapacityLimit> limit =
      ConvertCapacityLimit(source.limit);
  if (!limit.ok()) {
    return limit.status();
  }
  destination->set_limit(*limit);
  destination->set_observed(source.observed);
  destination->set_allowed(source.allowed);
  return absl::OkStatus();
}

absl::Status CopyCapacityResult(
    const CandidateRankerCapacityResult& source,
    AjimeeCapacityAuditCaseResult* destination) {
  CopyUsage(source.usage, destination->mutable_usage());
  for (const CandidateRankerSegmentCapacityDiagnostic& source_segment :
       source.segments) {
    AjimeeSegmentCapacityDiagnostic* destination_segment =
        destination->add_segments();
    destination_segment->set_segment_id(source_segment.segment_id);
    destination_segment->set_unprotected_candidate_count(
        source_segment.unprotected_candidate_count);
    destination_segment->set_window_candidate_count(
        source_segment.window_candidate_count);
    destination_segment->set_selected_candidate_count(
        source_segment.selected_candidate_count);
    destination_segment->set_omitted_by_window(
        source_segment.omitted_by_window);
    destination_segment->set_omitted_by_capacity(
        source_segment.omitted_by_capacity);
    absl::StatusOr<AjimeeSegmentCapacityDisposition> disposition =
        ConvertDisposition(source_segment.disposition);
    if (!disposition.ok()) {
      return disposition.status();
    }
    destination_segment->set_disposition(*disposition);
    if (source_segment.first_limiter.has_value()) {
      absl::Status status = CopyCapacityViolation(
          *source_segment.first_limiter,
          destination_segment->mutable_first_limiter());
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (source.request_limit_violation.has_value()) {
    return CopyRequestLimitViolation(
        *source.request_limit_violation,
        destination->mutable_request_limit_violation());
  }
  return absl::OkStatus();
}

absl::Status RecordAuditError(const absl::Status& status,
                              AjimeeCapacityAuditCaseResult* result) {
  absl::StatusOr<AjimeeCanonicalStatusCode> status_code =
      ConvertStatusCode(status.code());
  if (!status_code.ok()) {
    return status_code.status();
  }
  result->set_canonical_status_code(*status_code);
  result->set_outcome(AJIMEE_CAPACITY_AUDIT_ERROR);
  return absl::OkStatus();
}

bool IsZeroAfterRequestPreflight(const AjimeeCapacityAuditUsage& usage) {
  return usage.request_unprotected_candidate_count() == 0 &&
         usage.request_rankable_segment_count() == 0 &&
         usage.max_unprotected_candidates_in_segment() == 0 &&
         usage.window_candidate_count() == 0 &&
         usage.candidates_omitted_by_window() == 0 &&
         usage.selected_segment_count() == 0 &&
         usage.selected_candidate_count() == 0 &&
         usage.candidates_omitted_by_capacity() == 0 &&
         usage.segments_omitted_by_capacity() == 0 &&
         usage.decode_batch_count() == 0 &&
         usage.max_selected_serialized_record_bytes() == 0 &&
         usage.max_selected_normalized_record_bytes() == 0 &&
         usage.max_selected_full_record_tokens() == 0 &&
         usage.max_segments_in_decode() == 0 &&
         usage.max_sequences_in_decode() == 0 &&
         usage.max_input_nodes_in_decode() == 0 &&
         usage.max_output_rows_in_decode() == 0 &&
         usage.max_reserved_output_rows_in_decode() == 0 &&
         usage.max_output_logit_bytes_in_decode() == 0;
}

absl::Status ValidateRequestLimitResult(
    const AjimeeCapacityAuditCaseResult& result, const RequestFacts& facts) {
  if (result.canonical_status_code() != AJIMEE_STATUS_RESOURCE_EXHAUSTED ||
      !result.has_usage() || !result.has_request_limit_violation() ||
      result.segments_size() != 0 ||
      !IsZeroAfterRequestPreflight(result.usage())) {
    return absl::InvalidArgumentError(
        "AJIMEE request-limit audit result is inconsistent");
  }
  const AjimeeCapacityAuditUsage& usage = result.usage();
  const AjimeeRequestLimitViolation& violation =
      result.request_limit_violation();
  uint64_t expected_observed = 0;
  switch (violation.limit()) {
    case AJIMEE_REQUEST_SEGMENTS:
      expected_observed = facts.segment_count;
      if (usage.request_segment_count() != facts.segment_count ||
          usage.request_candidate_count() != 0 ||
          usage.request_string_bytes() != 0) {
        return absl::InvalidArgumentError(
            "AJIMEE segment-limit usage is inconsistent");
      }
      break;
    case AJIMEE_REQUEST_CANDIDATES:
      expected_observed = facts.candidate_count;
      if (usage.request_segment_count() != facts.segment_count ||
          usage.request_candidate_count() != facts.candidate_count ||
          usage.request_string_bytes() != 0) {
        return absl::InvalidArgumentError(
            "AJIMEE candidate-limit usage is inconsistent");
      }
      break;
    case AJIMEE_REQUEST_STRING_BYTES:
      expected_observed = facts.string_bytes;
      if (usage.request_segment_count() != facts.segment_count ||
          usage.request_candidate_count() != facts.candidate_count ||
          usage.request_string_bytes() != facts.string_bytes) {
        return absl::InvalidArgumentError(
            "AJIMEE string-limit usage is inconsistent");
      }
      break;
    case AJIMEE_REQUEST_LIMIT_UNSPECIFIED:
      return absl::InvalidArgumentError("AJIMEE request limit is unspecified");
    default:
      return absl::InvalidArgumentError("AJIMEE request limit is invalid");
  }
  if (violation.observed() != expected_observed ||
      violation.observed() <= violation.allowed()) {
    return absl::InvalidArgumentError(
        "AJIMEE request-limit violation is inconsistent");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapacityViolation(
    const AjimeeCapacityViolation& violation,
    const AjimeeCapacityAuditUsage& usage) {
  uint64_t selected_maximum = 0;
  switch (violation.limit()) {
    case AJIMEE_SERIALIZED_RECORD_BYTES:
      selected_maximum = usage.max_selected_serialized_record_bytes();
      break;
    case AJIMEE_NORMALIZED_RECORD_BYTES:
      selected_maximum = usage.max_selected_normalized_record_bytes();
      break;
    case AJIMEE_FULL_RECORD_TOKENS:
      selected_maximum = usage.max_selected_full_record_tokens();
      break;
    case AJIMEE_SEGMENTS_PER_DECODE:
      selected_maximum = usage.max_segments_in_decode();
      break;
    case AJIMEE_SEQUENCES_PER_DECODE:
      selected_maximum = usage.max_sequences_in_decode();
      break;
    case AJIMEE_INPUT_NODES_PER_DECODE:
      selected_maximum = usage.max_input_nodes_in_decode();
      break;
    case AJIMEE_OUTPUT_ROWS_PER_DECODE:
      selected_maximum = usage.max_output_rows_in_decode();
      break;
    case AJIMEE_RESERVED_OUTPUT_ROWS_PER_DECODE:
      selected_maximum = usage.max_reserved_output_rows_in_decode();
      break;
    case AJIMEE_OUTPUT_LOGIT_BYTES_PER_DECODE:
      selected_maximum = usage.max_output_logit_bytes_in_decode();
      break;
    case AJIMEE_CAPACITY_LIMIT_UNSPECIFIED:
    default:
      return absl::InvalidArgumentError(
          "AJIMEE segment capacity limiter is invalid");
  }
  if (violation.observed() <= violation.allowed() ||
      selected_maximum > violation.allowed()) {
    return absl::InvalidArgumentError(
        "AJIMEE segment capacity limiter is invalid");
  }
  return absl::OkStatus();
}

bool SelectedMaximaAreZero(const AjimeeCapacityAuditUsage& usage) {
  return usage.decode_batch_count() == 0 &&
         usage.max_selected_serialized_record_bytes() == 0 &&
         usage.max_selected_normalized_record_bytes() == 0 &&
         usage.max_selected_full_record_tokens() == 0 &&
         usage.max_segments_in_decode() == 0 &&
         usage.max_sequences_in_decode() == 0 &&
         usage.max_input_nodes_in_decode() == 0 &&
         usage.max_output_rows_in_decode() == 0 &&
         usage.max_reserved_output_rows_in_decode() == 0 &&
         usage.max_output_logit_bytes_in_decode() == 0;
}

absl::Status ValidatePassResult(const AjimeeCapacityAuditCaseResult& result,
                                const RequestFacts& facts) {
  if (result.canonical_status_code() != AJIMEE_STATUS_OK ||
      !result.has_usage() || result.has_request_limit_violation()) {
    return absl::InvalidArgumentError(
        "AJIMEE passing audit result is inconsistent");
  }
  const AjimeeCapacityAuditUsage& usage = result.usage();
  if (usage.request_segment_count() != facts.segment_count ||
      usage.request_candidate_count() != facts.candidate_count ||
      usage.request_unprotected_candidate_count() !=
          facts.unprotected_candidate_count ||
      usage.request_rankable_segment_count() !=
          facts.rankable_segment_count ||
      usage.request_string_bytes() != facts.string_bytes ||
      usage.max_unprotected_candidates_in_segment() !=
          facts.max_unprotected_candidates_in_segment ||
      result.segments_size() != facts.rankable_segment_count) {
    return absl::InvalidArgumentError(
        "AJIMEE passing request usage is inconsistent");
  }

  uint64_t window_candidate_count = 0;
  uint64_t candidates_omitted_by_window = 0;
  uint64_t selected_segment_count = 0;
  uint64_t selected_candidate_count = 0;
  uint64_t candidates_omitted_by_capacity = 0;
  uint64_t segments_omitted_by_capacity = 0;
  int diagnostic_index = 0;
  for (const SegmentFacts& segment : facts.segments) {
    if (segment.unprotected_candidate_count <= 1) {
      window_candidate_count += segment.unprotected_candidate_count;
      continue;
    }
    const AjimeeSegmentCapacityDiagnostic& diagnostic =
        result.segments(diagnostic_index++);
    if (diagnostic.segment_id() != segment.segment_id ||
        diagnostic.unprotected_candidate_count() !=
            segment.unprotected_candidate_count ||
        diagnostic.window_candidate_count() < 2 ||
        diagnostic.window_candidate_count() >
            diagnostic.unprotected_candidate_count() ||
        diagnostic.omitted_by_window() !=
            diagnostic.unprotected_candidate_count() -
                diagnostic.window_candidate_count()) {
      return absl::InvalidArgumentError(
          "AJIMEE segment window diagnostic is inconsistent");
    }
    window_candidate_count += diagnostic.window_candidate_count();
    candidates_omitted_by_window += diagnostic.omitted_by_window();
    switch (diagnostic.disposition()) {
      case AJIMEE_SEGMENT_CAPACITY_SELECTED:
        if (diagnostic.selected_candidate_count() < 2 ||
            diagnostic.selected_candidate_count() >
                diagnostic.window_candidate_count() ||
            diagnostic.omitted_by_capacity() !=
                diagnostic.window_candidate_count() -
                    diagnostic.selected_candidate_count()) {
          return absl::InvalidArgumentError(
              "AJIMEE selected segment diagnostic is inconsistent");
        }
        ++selected_segment_count;
        selected_candidate_count += diagnostic.selected_candidate_count();
        break;
      case AJIMEE_SEGMENT_CAPACITY_OMITTED_CAPACITY:
        if (diagnostic.selected_candidate_count() != 0 ||
            diagnostic.omitted_by_capacity() !=
                diagnostic.window_candidate_count()) {
          return absl::InvalidArgumentError(
              "AJIMEE omitted segment diagnostic is inconsistent");
        }
        ++segments_omitted_by_capacity;
        break;
      case AJIMEE_SEGMENT_CAPACITY_DISPOSITION_UNSPECIFIED:
      default:
        return absl::InvalidArgumentError(
            "AJIMEE segment disposition is unspecified");
    }
    candidates_omitted_by_capacity += diagnostic.omitted_by_capacity();
    if (diagnostic.has_first_limiter() !=
        (diagnostic.omitted_by_capacity() > 0)) {
      return absl::InvalidArgumentError(
          "AJIMEE segment limiter presence is inconsistent");
    }
    if (diagnostic.has_first_limiter()) {
      absl::Status status =
          ValidateCapacityViolation(diagnostic.first_limiter(), usage);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (usage.window_candidate_count() != window_candidate_count ||
      usage.candidates_omitted_by_window() !=
          candidates_omitted_by_window ||
      usage.selected_segment_count() != selected_segment_count ||
      usage.selected_candidate_count() != selected_candidate_count ||
      usage.candidates_omitted_by_capacity() !=
          candidates_omitted_by_capacity ||
      usage.segments_omitted_by_capacity() !=
          segments_omitted_by_capacity) {
    return absl::InvalidArgumentError(
        "AJIMEE aggregate capacity coverage is inconsistent");
  }
  if (selected_segment_count == 0) {
    if (!SelectedMaximaAreZero(usage)) {
      return absl::InvalidArgumentError(
          "AJIMEE empty selection maxima are inconsistent");
    }
  } else if (usage.decode_batch_count() == 0 ||
             usage.decode_batch_count() > selected_segment_count ||
             usage.max_selected_serialized_record_bytes() == 0 ||
             usage.max_selected_normalized_record_bytes() == 0 ||
             usage.max_selected_full_record_tokens() == 0 ||
             usage.max_segments_in_decode() == 0 ||
             usage.max_segments_in_decode() > selected_segment_count ||
             usage.max_sequences_in_decode() < 2 ||
             usage.max_sequences_in_decode() > selected_candidate_count ||
             usage.max_input_nodes_in_decode() == 0 ||
             usage.max_output_rows_in_decode() == 0 ||
             usage.max_output_rows_in_decode() >
                 usage.max_input_nodes_in_decode() ||
             usage.max_reserved_output_rows_in_decode() <
                 usage.max_sequences_in_decode() ||
             usage.max_output_logit_bytes_in_decode() == 0) {
    return absl::InvalidArgumentError(
        "AJIMEE selected capacity maxima are inconsistent");
  }
  return absl::OkStatus();
}

absl::Status ValidateErrorResult(
    const AjimeeCapacityAuditCaseResult& result) {
  if (result.has_usage() || result.has_request_limit_violation() ||
      result.segments_size() != 0) {
    return absl::InvalidArgumentError(
        "AJIMEE failed audit result contains capacity data");
  }
  if (!IsKnownCanonicalStatusCode(result.canonical_status_code()) ||
      result.canonical_status_code() == AJIMEE_STATUS_OK) {
    return absl::InvalidArgumentError(
        "AJIMEE audit error status is inconsistent");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<AjimeeCanonicalStatusCode> ConvertAjimeeCanonicalStatusCode(
    absl::StatusCode code) {
  return ConvertStatusCode(code);
}

bool IsKnownAjimeeCanonicalStatusCode(AjimeeCanonicalStatusCode code) {
  return IsKnownCanonicalStatusCode(code);
}

absl::Status ValidateAjimeeCapacityAuditConfig(
    const AjimeeCapacityAuditConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit config is incomplete");
  }
  if (config.schema_version() !=
          kAjimeeCapacityAuditConfigSchemaVersion ||
      config.frozen_corpus_schema_version() !=
          kAjimeeFrozenCorpusSchemaVersion ||
      config.capacity_audit_schema_version() !=
          kAjimeeCapacityAuditSchemaVersion ||
      config.model_manifest_schema_version() !=
          kAjimeeCapacityAuditModelManifestSchemaVersion) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit config schema mismatch");
  }
  if (!IsAjimeeLowercaseHex(config.frozen_corpus_sha256(), 64) ||
      config.expected_case_count() == 0 ||
      !IsAjimeeLowercaseHex(config.model_manifest_sha256(), 64) ||
      !IsSimpleFileName(config.gguf_file_name()) ||
      !IsAjimeeLowercaseHex(config.gguf_sha256(), 64) ||
      !IsAjimeeLowercaseHex(config.tokenizer_sha256(), 64) ||
      config.quantization().empty() ||
      !IsAjimeeLowercaseHex(config.source_revision(), 40) ||
      !IsAjimeeLowercaseHex(config.runtime_revision(), 40)) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit config identity is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateAjimeeCapacityAudit(
    const AjimeeCapacityAudit& audit, const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& config) {
  absl::Status status = ValidateAjimeeCapacityAuditConfig(config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  if (!audit.IsInitialized() || HasUnknownFieldsRecursively(audit)) {
    return absl::InvalidArgumentError("AJIMEE capacity audit is incomplete");
  }
  if (audit.schema_version() != kAjimeeCapacityAuditSchemaVersion ||
      audit.schema_version() != config.capacity_audit_schema_version()) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit schema mismatch");
  }
  status = ValidateIdentity(audit.identity());
  if (!status.ok()) {
    return status;
  }
  if (!IdentityMatchesConfig(audit.identity(), config)) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity audit identity does not match config");
  }
  if (corpus.schema_version() != config.frozen_corpus_schema_version() ||
      corpus.cases_size() != config.expected_case_count() ||
      audit.cases_size() != corpus.cases_size()) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit case count or frozen schema mismatch");
  }

  bool all_passed = true;
  for (int case_index = 0; case_index < audit.cases_size(); ++case_index) {
    const AjimeeCapacityAuditCaseResult& result = audit.cases(case_index);
    const AjimeeFrozenCase& frozen_case = corpus.cases(case_index);
    if (result.source_index() != frozen_case.source_index()) {
      return absl::InvalidArgumentError(
          "AJIMEE capacity audit cases are unordered");
    }
    absl::StatusOr<RequestFacts> facts =
        BuildRequestFacts(frozen_case.request());
    if (!facts.ok()) {
      return facts.status();
    }
    switch (result.outcome()) {
      case AJIMEE_CAPACITY_AUDIT_PASS:
        status = ValidatePassResult(result, *facts);
        break;
      case AJIMEE_CAPACITY_AUDIT_REQUEST_LIMIT_EXCEEDED:
        status = ValidateRequestLimitResult(result, *facts);
        all_passed = false;
        break;
      case AJIMEE_CAPACITY_AUDIT_ERROR:
        status = ValidateErrorResult(result);
        all_passed = false;
        break;
      case AJIMEE_CAPACITY_AUDIT_OUTCOME_UNSPECIFIED:
      default:
        return absl::InvalidArgumentError(
            "AJIMEE capacity audit outcome is unspecified");
    }
    if (!status.ok()) {
      return status;
    }
  }
  if (audit.all_passed() != all_passed) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity audit all-passed flag is inconsistent");
  }
  return absl::OkStatus();
}

absl::StatusOr<AjimeeCapacityAudit> BuildAjimeeCapacityAudit(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& config,
    const AjimeeCapacityAuditIdentity& identity,
    AjimeeCapacityAuditCallback audit_callback) {
  absl::Status status = ValidateAjimeeCapacityAuditConfig(config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  status = ValidateIdentity(identity);
  if (!status.ok()) {
    return status;
  }
  if (!IdentityMatchesConfig(identity, config)) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity audit identity does not match config");
  }
  if (corpus.cases_size() != config.expected_case_count()) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus case count does not match config");
  }

  AjimeeCapacityAudit audit;
  audit.set_schema_version(kAjimeeCapacityAuditSchemaVersion);
  AjimeeCapacityAuditIdentity* audit_identity = audit.mutable_identity();
  audit_identity->set_frozen_corpus_sha256(identity.frozen_corpus_sha256());
  audit_identity->set_model_manifest_sha256(identity.model_manifest_sha256());
  audit_identity->set_gguf_file_name(identity.gguf_file_name());
  audit_identity->set_gguf_sha256(identity.gguf_sha256());
  audit_identity->set_tokenizer_sha256(identity.tokenizer_sha256());
  audit_identity->set_quantization(identity.quantization());
  audit_identity->set_source_revision(identity.source_revision());
  audit_identity->set_runtime_revision(identity.runtime_revision());
  bool all_passed = true;
  for (const AjimeeFrozenCase& frozen_case : corpus.cases()) {
    AjimeeCapacityAuditCaseResult* case_result = audit.add_cases();
    case_result->set_source_index(frozen_case.source_index());
    absl::StatusOr<converter::CandidateRankerRequest> request =
        ConvertFrozenCandidateRankerRequest(frozen_case.request());
    if (!request.ok()) {
      return request.status();
    }
    absl::StatusOr<CandidateRankerCapacityResult> capacity =
        audit_callback(*request);
    if (!capacity.ok()) {
      status = RecordAuditError(capacity.status(), case_result);
      if (!status.ok()) {
        return status;
      }
      all_passed = false;
      continue;
    }
    status = CopyCapacityResult(*capacity, case_result);
    if (!status.ok()) {
      return status;
    }
    if (capacity->request_limit_violation.has_value()) {
      case_result->set_outcome(
          AJIMEE_CAPACITY_AUDIT_REQUEST_LIMIT_EXCEEDED);
      case_result->set_canonical_status_code(
          AJIMEE_STATUS_RESOURCE_EXHAUSTED);
      all_passed = false;
    } else {
      case_result->set_outcome(AJIMEE_CAPACITY_AUDIT_PASS);
      case_result->set_canonical_status_code(AJIMEE_STATUS_OK);
    }
  }
  audit.set_all_passed(all_passed);
  status = ValidateAjimeeCapacityAudit(audit, corpus, config);
  if (!status.ok()) {
    return status;
  }
  return audit;
}

}  // namespace mozc::engine::evaluation
