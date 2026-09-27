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

#ifndef MOZC_ENGINE_CANDIDATE_RANKER_MODEL_H_
#define MOZC_ENGINE_CANDIDATE_RANKER_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.pb.h"

namespace mozc {
namespace engine {

struct CandidateRankerCandidateReference {
  const converter::CandidateRankerCandidate* candidate = nullptr;
  size_t original_candidate_index = 0;
};

struct CandidateRankerRecordContext {
  converter::CandidateRankerMode mode =
      converter::CandidateRankerMode::kConversion;
  std::string preceding_text;
  std::string following_text;
  std::string reading;
  std::vector<converter::CandidateRankerSegment> segments;
};

enum class CandidateRankerCapacityLimit : uint8_t {
  kSerializedRecordBytes,
  kNormalizedRecordBytes,
  kFullRecordTokens,
  kSegmentsPerDecode,
  kSequencesPerDecode,
  kInputNodesPerDecode,
  kOutputRowsPerDecode,
  kReservedOutputRowsPerDecode,
  kOutputLogitBytesPerDecode,
};

struct CandidateRankerCapacityUsage {
  uint64_t request_segment_count = 0;
  uint64_t request_candidate_count = 0;
  uint64_t request_unprotected_candidate_count = 0;
  uint64_t request_rankable_segment_count = 0;
  uint64_t request_string_bytes = 0;
  uint64_t max_unprotected_candidates_in_segment = 0;
  uint64_t window_candidate_count = 0;
  uint64_t candidates_omitted_by_window = 0;
  uint64_t selected_segment_count = 0;
  uint64_t selected_candidate_count = 0;
  uint64_t candidates_omitted_by_capacity = 0;
  uint64_t segments_omitted_by_capacity = 0;
  uint64_t decode_batch_count = 0;
  uint64_t max_selected_serialized_record_bytes = 0;
  uint64_t max_selected_normalized_record_bytes = 0;
  uint64_t max_selected_full_record_tokens = 0;
  uint64_t max_segments_in_decode = 0;
  uint64_t max_sequences_in_decode = 0;
  uint64_t max_input_nodes_in_decode = 0;
  uint64_t max_output_rows_in_decode = 0;
  uint64_t max_reserved_output_rows_in_decode = 0;
  uint64_t max_output_logit_bytes_in_decode = 0;
};

enum class CandidateRankerRequestLimit : uint8_t {
  kSegments,
  kCandidates,
  kStringBytes,
};

struct CandidateRankerRequestLimitViolation {
  CandidateRankerRequestLimit limit = CandidateRankerRequestLimit::kSegments;
  uint64_t observed = 0;
  uint64_t allowed = 0;
};

struct CandidateRankerRequestPreflight {
  uint64_t segment_count = 0;
  uint64_t candidate_count = 0;
  uint64_t string_bytes = 0;
  std::optional<CandidateRankerRequestLimitViolation> violation;
};

struct CandidateRankerCapacityViolation {
  CandidateRankerCapacityLimit limit =
      CandidateRankerCapacityLimit::kSerializedRecordBytes;
  uint64_t observed = 0;
  uint64_t allowed = 0;
};

enum class CandidateRankerSegmentCapacityDisposition : uint8_t {
  kSelected,
  kOmittedCapacity,
};

struct CandidateRankerSegmentCapacityDiagnostic {
  converter::CandidateRankerSegmentId segment_id = 0;
  uint64_t unprotected_candidate_count = 0;
  uint64_t window_candidate_count = 0;
  uint64_t selected_candidate_count = 0;
  uint64_t omitted_by_window = 0;
  uint64_t omitted_by_capacity = 0;
  CandidateRankerSegmentCapacityDisposition disposition =
      CandidateRankerSegmentCapacityDisposition::kSelected;
  std::optional<CandidateRankerCapacityViolation> first_limiter;
};

struct CandidateRankerCapacityResult {
  CandidateRankerCapacityUsage usage;
  std::vector<CandidateRankerSegmentCapacityDiagnostic> segments;
  std::optional<CandidateRankerRequestLimitViolation> request_limit_violation;
};

struct CandidateRankerDecodeCapacityUsage {
  uint64_t segments = 0;
  uint64_t sequences = 0;
  uint64_t input_nodes = 0;
  uint64_t output_rows = 0;
  uint64_t reserved_output_rows = 0;
  uint64_t output_logit_bytes = 0;
};

struct CandidateRankerPreparedRecord {
  std::string normalized_record;
  uint64_t serialized_byte_count = 0;
  uint64_t normalized_byte_count = 0;
  std::optional<CandidateRankerCapacityViolation> violation;
};

struct CandidateRankerTokenSequence {
  converter::CandidateRankerCandidateId candidate_id = 0;
  size_t original_candidate_index = 0;
  // The source value must outlive planning and every returned decode batch.
  absl::string_view candidate_value;
  // Contains the directly inserted BOS followed by the tokenized record. The
  // terminal token is represented as a scored edge and is not an input token.
  std::vector<int32_t> tokens;
  uint64_t serialized_record_bytes = 0;
  uint64_t normalized_record_bytes = 0;
  uint64_t full_record_tokens = 0;
  uint32_t following_context_start_position = 0;
};

struct CandidateRankerTokenizedSegment {
  converter::CandidateRankerSegmentId segment_id = 0;
  std::vector<CandidateRankerTokenSequence> candidates;
};

struct CandidateRankerScoredEdge {
  int32_t target_token_id = 0;
  std::vector<int32_t> descendant_sequence_ids;
};

struct CandidateRankerBatchInput {
  int32_t token_id = 0;
  int32_t position = 0;
  std::vector<int32_t> sequence_ids;
  std::vector<CandidateRankerScoredEdge> scored_edges;
};

struct CandidateRankerPlannedCandidate {
  converter::CandidateRankerCandidateId candidate_id = 0;
  size_t original_candidate_index = 0;
  int32_t sequence_id = 0;
};

struct CandidateRankerSegmentPlan {
  converter::CandidateRankerSegmentId segment_id = 0;
  uint32_t common_prefix_token_count = 0;
  // Kept in original Mozc order so stable score ties preserve that order.
  std::vector<CandidateRankerPlannedCandidate> candidates;
};

struct CandidateRankerBatchPlan {
  std::vector<CandidateRankerBatchInput> inputs;
  std::vector<CandidateRankerSegmentPlan> segments;
  uint32_t sequence_count = 0;
  uint32_t output_row_count = 0;
};

struct CandidateRankerDecodeBatch {
  std::vector<CandidateRankerTokenizedSegment> tokenized_segments;
  CandidateRankerBatchPlan plan;
  CandidateRankerDecodeCapacityUsage usage;
};

absl::Status ValidateCandidateRankerModelManifest(
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<CandidateRankerRequestPreflight>
PreflightCandidateRankerRequest(
    const converter::CandidateRankerRequest& request,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation);

absl::StatusOr<std::vector<CandidateRankerCandidateReference>>
SelectCandidateRankerCandidates(
    const converter::CandidateRankerSegment& segment,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<CandidateRankerRecordContext> BuildCandidateRankerRecordContext(
    const converter::CandidateRankerRequest& request,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<std::string> BuildCandidateRankerRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<CandidateRankerPreparedRecord> PrepareCandidateRankerRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<std::string> BuildCandidateRankerCandidateEndingRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<CandidateRankerPreparedRecord>
PrepareCandidateRankerCandidateEndingRecord(
    const CandidateRankerRecordContext& context,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<CandidateRankerBatchPlan> BuildCandidateRankerBatchPlan(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation);

absl::StatusOr<CandidateRankerBatchPlan>
BuildCandidateRankerBatchPlanForCapacityAudit(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation);

absl::StatusOr<uint64_t> ComputeCandidateRankerOutputLogitBytes(
    uint64_t output_rows, uint64_t reserved_output_rows,
    uint64_t vocabulary_size);

absl::StatusOr<std::optional<CandidateRankerCapacityViolation>>
EvaluateCandidateRankerDecodeCapacity(
    CandidateRankerDecodeCapacityUsage usage,
    const CandidateRankerModelManifest& manifest);

absl::StatusOr<std::vector<CandidateRankerDecodeBatch>>
PackCandidateRankerSegmentsForDecode(
    absl::Span<const CandidateRankerTokenizedSegment> segments,
    uint64_t vocabulary_size, const CandidateRankerModelManifest& manifest,
    const converter::CandidateRankerCancellation& cancellation);

absl::StatusOr<uint32_t> FindCandidateRankerFollowingContextStartPosition(
    absl::Span<const int32_t> complete_tokens,
    absl::Span<const int32_t> candidate_prefix_tokens);

absl::StatusOr<std::vector<double>>
ComputeCandidateRankerEdgeLogProbabilities(
    const CandidateRankerBatchInput& input, absl::Span<const float> logits,
    const converter::CandidateRankerCancellation* cancellation = nullptr);

absl::Status AccumulateCandidateRankerEdgeLogProbabilities(
    const CandidateRankerBatchInput& input,
    absl::Span<const double> edge_log_probabilities,
    absl::Span<double> sequence_scores);

absl::Status AccumulateCandidateRankerOutput(
    const CandidateRankerBatchInput& input, absl::Span<const float> logits,
    absl::Span<double> sequence_scores);

absl::StatusOr<std::vector<converter::CandidateRankerSegmentOrder>>
OrderCandidateRankerContinuations(const CandidateRankerBatchPlan& plan,
                                  absl::Span<const double> sequence_scores);

}  // namespace engine
}  // namespace mozc

#endif  // MOZC_ENGINE_CANDIDATE_RANKER_MODEL_H_
