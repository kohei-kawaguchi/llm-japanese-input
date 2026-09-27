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
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.pb.h"
#include "testing/gunit.h"

namespace mozc {
namespace engine {
namespace {

constexpr char kSha256[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class TestCancellation final : public converter::CandidateRankerCancellation {
 public:
  explicit TestCancellation(bool cancelled = false) : cancelled_(cancelled) {}

  bool IsCancellationRequested() const override { return cancelled_; }

 private:
  bool cancelled_;
};

CandidateRankerModelManifest MakeManifest() {
  CandidateRankerModelManifest manifest;
  manifest.set_schema_version(5);

  CandidateRankerModelManifest::ScoringTemplate* scoring_template =
      manifest.mutable_scoring_template();
  scoring_template->set_version("continuation-test-v2");
  scoring_template->set_bos_token_id(1);
  scoring_template->set_terminal_token_id(2);
  scoring_template->set_mode_prefix("mode=");
  scoring_template->set_suggestion_mode("suggestion");
  scoring_template->set_prediction_mode("prediction");
  scoring_template->set_conversion_mode("conversion");
  scoring_template->set_field_separator("\n");
  scoring_template->set_reading_prefix("reading=");
  scoring_template->set_segment_key_prefix("segment=");
  scoring_template->set_text_prefix("text=");
  scoring_template->set_text_normalization(
      CandidateRankerModelManifest::ScoringTemplate::NONE);
  scoring_template->set_context_policy(
      CandidateRankerModelManifest::ScoringTemplate::
          MOZC_BASELINE_SUBSTITUTION);
  scoring_template->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET);

  CandidateRankerModelManifest::BoundedExecutionPolicy* bounded_policy =
      manifest.mutable_bounded_execution_policy();
  bounded_policy->set_candidate_window_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          MOZC_ORDER_UNPROTECTED_PREFIX);
  bounded_policy->set_capacity_reduction_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT);
  bounded_policy->set_decode_packing_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          SEGMENT_ID_ORDER_GREEDY_WHOLE_SEGMENTS);

  CandidateRankerModelManifest::Limits* limits = manifest.mutable_limits();
  limits->set_max_selected_candidates_per_segment(26);
  limits->set_max_segments_per_decode(8);
  limits->set_max_sequences_per_decode(64);
  limits->set_per_record_byte_limit(32768);
  limits->set_per_record_token_limit(64);
  limits->set_per_decode_input_node_limit(4096);
  limits->set_per_decode_output_row_limit(512);
  limits->set_max_decode_output_logit_bytes(67108864);
  limits->set_max_request_segments(64);
  limits->set_max_request_candidates(2048);
  limits->set_max_request_string_bytes(1048576);

  CandidateRankerModelManifest::Runtime* runtime = manifest.mutable_runtime();
  runtime->set_execution_device(CandidateRankerModelManifest::Runtime::CPU);
  runtime->set_kv_storage(CandidateRankerModelManifest::Runtime::UNIFIED);
  runtime->set_micro_batch_token_capacity(64);
  runtime->set_decode_thread_count(2);
  runtime->set_batch_thread_count(4);
  runtime->set_use_memory_map(true);
  runtime->set_use_memory_lock(false);
  runtime->set_check_tensors(true);

  CandidateRankerModelManifest::Artifacts* artifacts =
      manifest.mutable_artifacts();
  artifacts->set_source_model("example/model");
  artifacts->set_source_revision("0123456789abcdef0123456789abcdef01234567");
  artifacts->set_source_weight_sha256(kSha256);
  artifacts->set_tokenizer_sha256(kSha256);
  artifacts->set_gguf_file_name("model.gguf");
  artifacts->set_gguf_sha256(kSha256);
  artifacts->set_quantization("Q4_K_M");
  artifacts->add_model_license_references("MODEL_LICENSE.txt");
  artifacts->set_runtime_name("llama.cpp");
  artifacts->set_runtime_revision("fedcba9876543210fedcba9876543210fedcba98");
  artifacts->set_runtime_source_archive_sha256(kSha256);
  artifacts->add_runtime_license_references("LLAMA_CPP_LICENSE.txt");
  return manifest;
}

absl::StatusOr<std::string> BuildRecord(
    const converter::CandidateRankerRequest& request,
    converter::CandidateRankerSegmentId segment_id,
    converter::CandidateRankerCandidateId candidate_id,
    const CandidateRankerModelManifest& manifest) {
  const absl::StatusOr<CandidateRankerRecordContext> context =
      BuildCandidateRankerRecordContext(request, manifest);
  if (!context.ok()) {
    return context.status();
  }
  return BuildCandidateRankerRecord(*context, segment_id, candidate_id,
                                    manifest);
}

converter::CandidateRankerCandidate Candidate(
    converter::CandidateRankerCandidateId id, std::string key,
    std::string value, bool is_protected = false) {
  return converter::CandidateRankerCandidate{
      .id = id,
      .key = std::move(key),
      .value = std::move(value),
      .is_protected = is_protected,
  };
}

CandidateRankerTokenSequence TokenSequence(
    converter::CandidateRankerCandidateId id, size_t original_index,
    absl::string_view value, std::vector<int32_t> tokens) {
  return CandidateRankerTokenSequence{
      .candidate_id = id,
      .original_candidate_index = original_index,
      .candidate_value = value,
      .tokens = std::move(tokens),
  };
}

void ExpectBatchPlansEqual(const CandidateRankerBatchPlan& lhs,
                           const CandidateRankerBatchPlan& rhs) {
  EXPECT_EQ(lhs.sequence_count, rhs.sequence_count);
  EXPECT_EQ(lhs.output_row_count, rhs.output_row_count);
  ASSERT_EQ(lhs.inputs.size(), rhs.inputs.size());
  for (size_t i = 0; i < lhs.inputs.size(); ++i) {
    EXPECT_EQ(lhs.inputs[i].token_id, rhs.inputs[i].token_id);
    EXPECT_EQ(lhs.inputs[i].position, rhs.inputs[i].position);
    EXPECT_EQ(lhs.inputs[i].sequence_ids, rhs.inputs[i].sequence_ids);
    ASSERT_EQ(lhs.inputs[i].scored_edges.size(),
              rhs.inputs[i].scored_edges.size());
    for (size_t edge = 0; edge < lhs.inputs[i].scored_edges.size(); ++edge) {
      EXPECT_EQ(lhs.inputs[i].scored_edges[edge].target_token_id,
                rhs.inputs[i].scored_edges[edge].target_token_id);
      EXPECT_EQ(lhs.inputs[i].scored_edges[edge].descendant_sequence_ids,
                rhs.inputs[i].scored_edges[edge].descendant_sequence_ids);
    }
  }
  ASSERT_EQ(lhs.segments.size(), rhs.segments.size());
  for (size_t i = 0; i < lhs.segments.size(); ++i) {
    EXPECT_EQ(lhs.segments[i].segment_id, rhs.segments[i].segment_id);
    EXPECT_EQ(lhs.segments[i].common_prefix_token_count,
              rhs.segments[i].common_prefix_token_count);
    ASSERT_EQ(lhs.segments[i].candidates.size(),
              rhs.segments[i].candidates.size());
    for (size_t candidate = 0; candidate < lhs.segments[i].candidates.size();
         ++candidate) {
      EXPECT_EQ(lhs.segments[i].candidates[candidate].candidate_id,
                rhs.segments[i].candidates[candidate].candidate_id);
      EXPECT_EQ(lhs.segments[i].candidates[candidate].original_candidate_index,
                rhs.segments[i].candidates[candidate].original_candidate_index);
      EXPECT_EQ(lhs.segments[i].candidates[candidate].sequence_id,
                rhs.segments[i].candidates[candidate].sequence_id);
    }
  }
}

TEST(CandidateRankerModelTest, ValidatesManifestVersionAndLimitRelationships) {
  CandidateRankerModelManifest manifest = MakeManifest();
  EXPECT_TRUE(ValidateCandidateRankerModelManifest(manifest).ok());

  manifest.set_schema_version(4);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_selected_candidates_per_segment(1);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_selected_candidates_per_segment(65);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_sequences_per_decode(256);
  manifest.mutable_limits()->set_per_decode_output_row_limit(256);
  EXPECT_TRUE(ValidateCandidateRankerModelManifest(manifest).ok());
  manifest.mutable_limits()->set_max_sequences_per_decode(257);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_per_decode_input_node_limit(255);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_per_decode_output_row_limit(7);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_per_decode_output_row_limit(4097);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_decode_output_logit_bytes(63);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_request_segments(7);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_request_candidates(63);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_request_string_bytes(0);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_runtime()->set_micro_batch_token_capacity(4097);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest, EvaluatesEveryPerDecodeCapacityAtItsBoundary) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  CandidateRankerDecodeCapacityUsage boundary{
      .segments = 8,
      .sequences = 64,
      .input_nodes = 4096,
      .output_rows = 512,
      .reserved_output_rows = 512,
      .output_logit_bytes = 67108864,
  };
  const auto passed =
      EvaluateCandidateRankerDecodeCapacity(boundary, manifest);
  ASSERT_TRUE(passed.ok()) << passed.status();
  EXPECT_FALSE(passed->has_value());

  const auto expect_violation =
      [&manifest](CandidateRankerDecodeCapacityUsage usage,
                  CandidateRankerCapacityLimit limit, uint64_t observed,
                  uint64_t allowed) {
        const auto result =
            EvaluateCandidateRankerDecodeCapacity(usage, manifest);
        ASSERT_TRUE(result.ok()) << result.status();
        ASSERT_TRUE(result->has_value());
        EXPECT_EQ((*result)->limit, limit);
        EXPECT_EQ((*result)->observed, observed);
        EXPECT_EQ((*result)->allowed, allowed);
      };

  boundary.segments = 9;
  expect_violation(boundary,
                   CandidateRankerCapacityLimit::kSegmentsPerDecode, 9, 8);
  boundary = CandidateRankerDecodeCapacityUsage{.sequences = 65};
  expect_violation(boundary,
                   CandidateRankerCapacityLimit::kSequencesPerDecode, 65, 64);
  boundary = CandidateRankerDecodeCapacityUsage{.input_nodes = 4097};
  expect_violation(boundary,
                   CandidateRankerCapacityLimit::kInputNodesPerDecode, 4097,
                   4096);
  boundary = CandidateRankerDecodeCapacityUsage{.output_rows = 513};
  expect_violation(boundary,
                   CandidateRankerCapacityLimit::kOutputRowsPerDecode, 513,
                   512);
  boundary = CandidateRankerDecodeCapacityUsage{.reserved_output_rows = 513};
  expect_violation(
      boundary,
      CandidateRankerCapacityLimit::kReservedOutputRowsPerDecode, 513, 512);
  boundary = CandidateRankerDecodeCapacityUsage{
      .output_logit_bytes = 67108865,
  };
  expect_violation(
      boundary, CandidateRankerCapacityLimit::kOutputLogitBytesPerDecode,
      67108865, 67108864);

  const auto logit_bytes =
      ComputeCandidateRankerOutputLogitBytes(2, 3, 4);
  ASSERT_TRUE(logit_bytes.ok()) << logit_bytes.status();
  EXPECT_EQ(*logit_bytes, 48);
}

TEST(CandidateRankerModelTest, PreflightsRequestLimitsAtExactBoundaries) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;

  converter::CandidateRankerRequest request;
  request.segments.resize(64);
  auto preflight =
      PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  EXPECT_EQ(preflight->segment_count, 64);
  EXPECT_FALSE(preflight->violation.has_value());

  request.segments.emplace_back();
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  ASSERT_TRUE(preflight->violation.has_value());
  EXPECT_EQ(preflight->violation->limit,
            CandidateRankerRequestLimit::kSegments);
  EXPECT_EQ(preflight->violation->observed, 65);
  EXPECT_EQ(preflight->violation->allowed, 64);
  EXPECT_EQ(preflight->candidate_count, 0);
  EXPECT_EQ(preflight->string_bytes, 0);

  request = converter::CandidateRankerRequest{};
  request.segments.resize(1);
  request.segments[0].candidates.resize(2048);
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  EXPECT_EQ(preflight->candidate_count, 2048);
  EXPECT_FALSE(preflight->violation.has_value());

  request.segments[0].candidates.emplace_back();
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  ASSERT_TRUE(preflight->violation.has_value());
  EXPECT_EQ(preflight->violation->limit,
            CandidateRankerRequestLimit::kCandidates);
  EXPECT_EQ(preflight->violation->observed, 2049);
  EXPECT_EQ(preflight->violation->allowed, 2048);
  EXPECT_EQ(preflight->string_bytes, 0);

  request = converter::CandidateRankerRequest{};
  request.reading.assign(1048576, 'a');
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  EXPECT_EQ(preflight->string_bytes, 1048576);
  EXPECT_FALSE(preflight->violation.has_value());

  request.reading.push_back('a');
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  ASSERT_TRUE(preflight->violation.has_value());
  EXPECT_EQ(preflight->violation->limit,
            CandidateRankerRequestLimit::kStringBytes);
  EXPECT_EQ(preflight->violation->observed, 1048577);
  EXPECT_EQ(preflight->violation->allowed, 1048576);
}

TEST(CandidateRankerModelTest,
     PreflightCountsEveryRequestStringAndHonorsCancellation) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request{
      .preceding_text = "before",
      .following_text = "after",
      .reading = "reading",
      .segments = {
          {
              .key = "segment",
              .candidates =
                  {
                      Candidate(1, "key-one", "value-one"),
                      Candidate(2, "key-two", "value-two"),
                  },
          },
      },
  };
  const TestCancellation cancellation;
  const auto preflight =
      PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  EXPECT_EQ(preflight->segment_count, 1);
  EXPECT_EQ(preflight->candidate_count, 2);
  EXPECT_EQ(preflight->string_bytes,
            request.reading.size() + request.preceding_text.size() +
                request.following_text.size() + request.segments[0].key.size() +
                request.segments[0].candidates[0].key.size() +
                request.segments[0].candidates[0].value.size() +
                request.segments[0].candidates[1].key.size() +
                request.segments[0].candidates[1].value.size());

  const TestCancellation cancelled(true);
  EXPECT_EQ(PreflightCandidateRankerRequest(request, manifest, cancelled)
                .status()
                .code(),
            absl::StatusCode::kCancelled);
}

TEST(CandidateRankerModelTest, PreflightUsesManifestLimitPrecedence) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_limits()->set_max_request_segments(8);
  manifest.mutable_limits()->set_max_request_candidates(64);
  manifest.mutable_limits()->set_max_request_string_bytes(1);
  const TestCancellation cancellation;
  converter::CandidateRankerRequest request{
      .reading = "xx",
      .segments = std::vector<converter::CandidateRankerSegment>(9),
  };
  request.segments[0].candidates.resize(65);

  auto preflight =
      PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  ASSERT_TRUE(preflight->violation.has_value());
  EXPECT_EQ(preflight->violation->limit,
            CandidateRankerRequestLimit::kSegments);
  EXPECT_EQ(preflight->candidate_count, 0);
  EXPECT_EQ(preflight->string_bytes, 0);

  request.segments.resize(1);
  preflight = PreflightCandidateRankerRequest(request, manifest, cancellation);
  ASSERT_TRUE(preflight.ok()) << preflight.status();
  ASSERT_TRUE(preflight->violation.has_value());
  EXPECT_EQ(preflight->violation->limit,
            CandidateRankerRequestLimit::kCandidates);
  EXPECT_EQ(preflight->candidate_count, 65);
  EXPECT_EQ(preflight->string_bytes, 0);
}

TEST(CandidateRankerModelTest, RequiresCheckedTemplateAndUnifiedRuntime) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_scoring_template()->set_text_normalization(
      CandidateRankerModelManifest::ScoringTemplate::
          TEXT_NORMALIZATION_UNSPECIFIED);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_bounded_execution_policy()->set_candidate_window_policy(
      CandidateRankerModelManifest::BoundedExecutionPolicy::
          CANDIDATE_WINDOW_POLICY_UNSPECIFIED);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_scoring_template()->clear_context_policy();
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_scoring_template()->set_context_policy(
      CandidateRankerModelManifest::ScoringTemplate::
          CONTEXT_POLICY_UNSPECIFIED);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_scoring_template()->clear_record_layout();
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::
          RECORD_LAYOUT_UNSPECIFIED);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  for (const CandidateRankerModelManifest::ScoringTemplate::RecordLayout
           record_layout :
       {CandidateRankerModelManifest::ScoringTemplate::STRUCTURED_WITH_TARGET,
        CandidateRankerModelManifest::ScoringTemplate::
            STRUCTURED_WITHOUT_TARGET,
        CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY}) {
    manifest = MakeManifest();
    manifest.mutable_scoring_template()->set_record_layout(record_layout);
    EXPECT_TRUE(ValidateCandidateRankerModelManifest(manifest).ok());
  }

  manifest = MakeManifest();
  manifest.mutable_scoring_template()->clear_text_prefix();
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_runtime()->set_kv_storage(
      CandidateRankerModelManifest::Runtime::KV_STORAGE_UNSPECIFIED);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_runtime()->set_decode_thread_count(256);
  manifest.mutable_runtime()->set_batch_thread_count(256);
  EXPECT_TRUE(ValidateCandidateRankerModelManifest(manifest).ok());
  manifest.mutable_runtime()->set_batch_thread_count(257);
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);

  manifest = MakeManifest();
  manifest.mutable_artifacts()->set_gguf_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeF");
  EXPECT_EQ(ValidateCandidateRankerModelManifest(manifest).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest,
     SerializesArchivedStructuredWithTargetRecordExactly) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.mode = converter::CandidateRankerMode::kPrediction;
  request.reading = "よみ";
  request.preceding_text = "前";
  request.following_text = "後";
  const converter::CandidateRankerSegment segment{
      .id = 3,
      .key = "せぐめんと",
      .candidates =
          {
              Candidate(10, "candidate-key-is-not-serialized", "候補"),
          },
  };
  request.segments = {segment};

  const auto record = BuildRecord(request, segment.id, 10, manifest);
  ASSERT_TRUE(record.ok()) << record.status();
  EXPECT_EQ(*record,
            "mode=prediction\n"
            "reading=よみ\n"
            "segment=せぐめんと\n"
            "text=前候補後");
}

TEST(CandidateRankerModelTest,
     SerializesThreeSegmentBaselineSubstitutionsExactly) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.mode = converter::CandidateRankerMode::kPrediction;
  request.reading = "よみ";
  request.preceding_text = "<前>";
  request.following_text = "<後>";
  const converter::CandidateRankerSegment left{
      .id = 10,
      .key = "左キー",
      .candidates =
          {
              Candidate(50, "", "左候補"),
              Candidate(10, "", "左基準", true),
          },
  };
  const converter::CandidateRankerSegment middle{
      .id = 20,
      .key = "中キー",
      .candidates =
          {
              Candidate(60, "", "中候補"),
              Candidate(20, "", "中基準"),
          },
  };
  const converter::CandidateRankerSegment right{
      .id = 30,
      .key = "右キー",
      .candidates =
          {
              Candidate(70, "", "右候補"),
              Candidate(30, "", "右基準"),
          },
  };
  request.segments = {right, left, middle};

  const auto context = BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(context.ok()) << context.status();
  const auto first = BuildCandidateRankerRecord(*context, 10, 50, manifest);
  const auto center = BuildCandidateRankerRecord(*context, 20, 60, manifest);
  const auto last = BuildCandidateRankerRecord(*context, 30, 70, manifest);
  const auto center_ending =
      BuildCandidateRankerCandidateEndingRecord(*context, 20, 60, manifest);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(center.ok()) << center.status();
  ASSERT_TRUE(last.ok()) << last.status();
  ASSERT_TRUE(center_ending.ok()) << center_ending.status();
  EXPECT_EQ(*first,
            "mode=prediction\nreading=よみ\nsegment=左キー\n"
            "text=<前>左候補中基準右基準<後>");
  EXPECT_EQ(*center,
            "mode=prediction\nreading=よみ\nsegment=中キー\n"
            "text=<前>左基準中候補右基準<後>");
  EXPECT_EQ(*last,
            "mode=prediction\nreading=よみ\nsegment=右キー\n"
            "text=<前>左基準中基準右候補<後>");
  EXPECT_EQ(*center_ending,
            "mode=prediction\nreading=よみ\nsegment=中キー\n"
            "text=<前>左基準中候補");

  std::reverse(request.segments.begin(), request.segments.end());
  const auto permuted_context =
      BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(permuted_context.ok()) << permuted_context.status();
  const auto permuted =
      BuildCandidateRankerRecord(*permuted_context, 20, 60, manifest);
  ASSERT_TRUE(permuted.ok()) << permuted.status();
  EXPECT_EQ(*permuted, *center);
}

TEST(CandidateRankerModelTest,
     SerializesTargetAblationAndNaturalTextRecordsExactly) {
  converter::CandidateRankerRequest request;
  request.mode = converter::CandidateRankerMode::kPrediction;
  request.reading = "よみ";
  request.preceding_text = "<前>";
  request.following_text = "<後>";
  request.segments = {
      {
          .id = 30,
          .key = "右キー",
          .candidates = {Candidate(30, "", "右基準")},
      },
      {
          .id = 20,
          .key = "中キー",
          .candidates =
              {
                  Candidate(20, "", "中基準"),
                  Candidate(60, "", "中候補"),
              },
      },
      {
          .id = 10,
          .key = "左キー",
          .candidates = {Candidate(10, "", "左基準")},
      },
  };

  CandidateRankerModelManifest without_target = MakeManifest();
  without_target.mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITHOUT_TARGET);
  const auto without_target_context =
      BuildCandidateRankerRecordContext(request, without_target);
  ASSERT_TRUE(without_target_context.ok())
      << without_target_context.status();
  const auto without_target_record = BuildCandidateRankerRecord(
      *without_target_context, 20, 60, without_target);
  const auto without_target_ending =
      BuildCandidateRankerCandidateEndingRecord(
          *without_target_context, 20, 60, without_target);
  ASSERT_TRUE(without_target_record.ok()) << without_target_record.status();
  ASSERT_TRUE(without_target_ending.ok()) << without_target_ending.status();
  EXPECT_EQ(*without_target_record,
            "mode=prediction\nreading=よみ\n"
            "text=<前>左基準中候補右基準<後>");
  EXPECT_EQ(*without_target_ending,
            "mode=prediction\nreading=よみ\n"
            "text=<前>左基準中候補");

  CandidateRankerModelManifest natural_text = MakeManifest();
  natural_text.mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY);
  const auto natural_text_context =
      BuildCandidateRankerRecordContext(request, natural_text);
  ASSERT_TRUE(natural_text_context.ok()) << natural_text_context.status();
  const auto natural_text_record = BuildCandidateRankerRecord(
      *natural_text_context, 20, 60, natural_text);
  const auto natural_text_ending = BuildCandidateRankerCandidateEndingRecord(
      *natural_text_context, 20, 60, natural_text);
  ASSERT_TRUE(natural_text_record.ok()) << natural_text_record.status();
  ASSERT_TRUE(natural_text_ending.ok()) << natural_text_ending.status();
  EXPECT_EQ(*natural_text_record, "<前>左基準中候補右基準<後>");
  EXPECT_EQ(*natural_text_ending, "<前>左基準中候補");

  std::reverse(request.segments.begin(), request.segments.end());
  const auto permuted_without_target_context =
      BuildCandidateRankerRecordContext(request, without_target);
  const auto permuted_natural_text_context =
      BuildCandidateRankerRecordContext(request, natural_text);
  ASSERT_TRUE(permuted_without_target_context.ok())
      << permuted_without_target_context.status();
  ASSERT_TRUE(permuted_natural_text_context.ok())
      << permuted_natural_text_context.status();
  const auto permuted_without_target = BuildCandidateRankerRecord(
      *permuted_without_target_context, 20, 60, without_target);
  const auto permuted_natural_text = BuildCandidateRankerRecord(
      *permuted_natural_text_context, 20, 60, natural_text);
  ASSERT_TRUE(permuted_without_target.ok())
      << permuted_without_target.status();
  ASSERT_TRUE(permuted_natural_text.ok()) << permuted_natural_text.status();
  EXPECT_EQ(*permuted_without_target, *without_target_record);
  EXPECT_EQ(*permuted_natural_text, *natural_text_record);
}

TEST(CandidateRankerModelTest, OmittedMetadataDoesNotAlterRecordBytes) {
  converter::CandidateRankerRequest request;
  request.mode = converter::CandidateRankerMode::kSuggestion;
  request.reading = "first-reading";
  request.preceding_text = "before";
  request.following_text = "after";
  request.segments = {{
      .id = 1,
      .key = "first-target",
      .candidates = {Candidate(1, "unused", "candidate")},
  }};

  CandidateRankerModelManifest without_target = MakeManifest();
  without_target.mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::
          STRUCTURED_WITHOUT_TARGET);
  const auto initial_without_target =
      BuildRecord(request, 1, 1, without_target);
  ASSERT_TRUE(initial_without_target.ok()) << initial_without_target.status();
  request.segments[0].key = "second-target";
  const auto changed_without_target =
      BuildRecord(request, 1, 1, without_target);
  ASSERT_TRUE(changed_without_target.ok()) << changed_without_target.status();
  EXPECT_EQ(*changed_without_target, *initial_without_target);

  CandidateRankerModelManifest natural_text = MakeManifest();
  natural_text.mutable_scoring_template()->set_record_layout(
      CandidateRankerModelManifest::ScoringTemplate::NATURAL_TEXT_ONLY);
  const auto initial_natural_text = BuildRecord(request, 1, 1, natural_text);
  ASSERT_TRUE(initial_natural_text.ok()) << initial_natural_text.status();
  request.mode = converter::CandidateRankerMode::kConversion;
  request.reading = "second-reading";
  request.segments[0].key = "third-target";
  const auto changed_natural_text = BuildRecord(request, 1, 1, natural_text);
  ASSERT_TRUE(changed_natural_text.ok()) << changed_natural_text.status();
  EXPECT_EQ(*changed_natural_text, *initial_natural_text);
  EXPECT_EQ(*changed_natural_text, "beforecandidateafter");
}

TEST(CandidateRankerModelTest,
     KeepsUnrankableAndProtectedOnlyNeighborBaselines) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.segments = {
      {
          .id = 1,
          .key = "protected",
          .candidates = {Candidate(10, "", "保護", true)},
      },
      {
          .id = 2,
          .key = "single",
          .candidates = {Candidate(20, "", "一個")},
      },
      {
          .id = 3,
          .key = "target",
          .candidates =
              {
                  Candidate(30, "", "対象"),
                  Candidate(31, "", "別"),
              },
      },
  };

  const auto context = BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(context.ok()) << context.status();
  const auto record = BuildCandidateRankerRecord(*context, 3, 30, manifest);
  ASSERT_TRUE(record.ok()) << record.status();
  EXPECT_EQ(*record,
            "mode=conversion\nreading=\nsegment=target\n"
            "text=保護一個対象");
}

TEST(CandidateRankerModelTest, RejectsMissingBaselineOrTargetCandidate) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.segments = {{
      .id = 1,
      .key = "target",
      .candidates = {Candidate(10, "", "候補")},
  }};
  const auto context = BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(context.ok()) << context.status();
  EXPECT_EQ(BuildCandidateRankerRecord(*context, 1, 99, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  request.segments.push_back({.id = 2, .key = "missing"});
  EXPECT_EQ(BuildCandidateRankerRecordContext(request, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest, RejectsDuplicateGlobalRequestIds) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request{
      .segments = {
          {.id = 1, .candidates = {Candidate(10, "", "first")}},
          {.id = 2, .candidates = {Candidate(10, "", "second")}},
      },
  };
  EXPECT_EQ(BuildCandidateRankerRecordContext(request, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  request.segments[1].candidates[0].id = 11;
  request.segments[1].id = 1;
  EXPECT_EQ(BuildCandidateRankerRecordContext(request, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest, AppliesPythonLowerToTheCompleteRecord) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_scoring_template()->set_text_normalization(
      CandidateRankerModelManifest::ScoringTemplate::
          PYTHON_UNICODE_14_SCALAR_LOWER);
  converter::CandidateRankerRequest request;
  request.reading = "ABCİ";
  request.preceding_text = "PRE";
  request.following_text = "POST";
  const converter::CandidateRankerSegment segment{
      .key = "KEY",
      .candidates = {Candidate(1, "unused", "VALUE")},
  };
  request.segments = {segment};

  const auto record = BuildRecord(request, segment.id, 1, manifest);
  ASSERT_TRUE(record.ok()) << record.status();
  EXPECT_EQ(*record,
            "mode=conversion\n"
            "reading=abci\u0307\n"
            "segment=key\n"
            "text=prevaluepost");
}

TEST(CandidateRankerModelTest, RejectsInvalidUtf8BeforeNormalization) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.reading = std::string(1, static_cast<char>(0x80));
  const converter::CandidateRankerSegment segment{
      .key = "segment",
      .candidates = {Candidate(1, "unused", "value")},
  };
  request.segments = {segment};

  EXPECT_EQ(BuildRecord(request, segment.id, 1, manifest).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest,
     RejectsUtf8ThatOnlyBecomesValidAcrossRecordFields) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.following_text = std::string(1, static_cast<char>(0xa9));
  const converter::CandidateRankerSegment segment{
      .key = "segment",
      .candidates =
          {Candidate(1, "unused",
                     std::string(1, static_cast<char>(0xc3))),
           Candidate(2, "unused",
                     std::string(1, static_cast<char>(0xc3)))},
  };
  request.segments = {segment};

  const auto context = BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(context.ok()) << context.status();
  EXPECT_EQ(BuildCandidateRankerRecord(*context, segment.id, 1, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(BuildCandidateRankerCandidateEndingRecord(
                *context, segment.id, 1, manifest)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest, EnforcesRawAndNormalizedRecordByteLimits) {
  CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerRequest request;
  request.reading = "reading";
  const converter::CandidateRankerSegment segment{
      .key = "segment",
      .candidates = {Candidate(1, "unused", "value")},
  };
  request.segments = {segment};
  const auto initial = BuildRecord(request, segment.id, 1, manifest);
  ASSERT_TRUE(initial.ok()) << initial.status();

  manifest.mutable_limits()->set_per_record_byte_limit(initial->size());
  EXPECT_TRUE(BuildRecord(request, segment.id, 1, manifest).ok());
  manifest.mutable_limits()->set_per_record_byte_limit(initial->size() - 1);
  const auto context = BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(context.ok()) << context.status();
  const auto raw_violation =
      PrepareCandidateRankerRecord(*context, segment.id, 1, manifest);
  ASSERT_TRUE(raw_violation.ok()) << raw_violation.status();
  ASSERT_TRUE(raw_violation->violation.has_value());
  EXPECT_EQ(raw_violation->violation->limit,
            CandidateRankerCapacityLimit::kSerializedRecordBytes);
  EXPECT_EQ(raw_violation->violation->observed, initial->size());
  EXPECT_EQ(raw_violation->violation->allowed, initial->size() - 1);

  manifest = MakeManifest();
  request.reading = "İ";
  const auto raw = BuildRecord(request, segment.id, 1, manifest);
  ASSERT_TRUE(raw.ok()) << raw.status();
  manifest.mutable_limits()->set_per_record_byte_limit(raw->size());
  manifest.mutable_scoring_template()->set_text_normalization(
      CandidateRankerModelManifest::ScoringTemplate::
          PYTHON_UNICODE_14_SCALAR_LOWER);
  const auto normalized_context =
      BuildCandidateRankerRecordContext(request, manifest);
  ASSERT_TRUE(normalized_context.ok()) << normalized_context.status();
  const auto normalized_violation = PrepareCandidateRankerRecord(
      *normalized_context, segment.id, 1, manifest);
  ASSERT_TRUE(normalized_violation.ok()) << normalized_violation.status();
  ASSERT_TRUE(normalized_violation->violation.has_value());
  EXPECT_EQ(normalized_violation->violation->limit,
            CandidateRankerCapacityLimit::kNormalizedRecordBytes);
  EXPECT_EQ(normalized_violation->violation->observed, raw->size() + 1);
  EXPECT_EQ(normalized_violation->violation->allowed, raw->size());
}

TEST(CandidateRankerModelTest, SelectsEligibleCandidatesInMozcOrder) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  converter::CandidateRankerSegment segment{
      .candidates =
          {
              Candidate(9, "p", "protected", true),
              Candidate(3, "z", "same"),
              Candidate(2, "a", "same"),
              Candidate(1, "b", "alpha"),
          },
  };

  const auto selected = SelectCandidateRankerCandidates(segment, manifest);
  ASSERT_TRUE(selected.ok()) << selected.status();
  ASSERT_EQ(selected->size(), 3);
  EXPECT_EQ((*selected)[0].candidate->id, 3);
  EXPECT_EQ((*selected)[1].candidate->id, 2);
  EXPECT_EQ((*selected)[2].candidate->id, 1);
  EXPECT_EQ((*selected)[0].original_candidate_index, 1);

  segment.candidates.resize(1);
  const auto omitted = SelectCandidateRankerCandidates(segment, manifest);
  ASSERT_TRUE(omitted.ok()) << omitted.status();
  EXPECT_TRUE(omitted->empty());
}

TEST(CandidateRankerModelTest,
     SelectsTheMozcOrderWindowBeforeCanonicalGraphSorting) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_limits()->set_max_selected_candidates_per_segment(2);
  converter::CandidateRankerSegment segment{
      .candidates =
          {
              Candidate(1, "a", "z"),
              Candidate(9, "p", "protected", true),
              Candidate(2, "b", "y"),
              Candidate(3, "c", "alpha"),
          },
  };

  const auto selected = SelectCandidateRankerCandidates(segment, manifest);
  ASSERT_TRUE(selected.ok()) << selected.status();
  ASSERT_EQ(selected->size(), 2);
  EXPECT_EQ((*selected)[0].candidate->id, 1);
  EXPECT_EQ((*selected)[0].original_candidate_index, 0);
  EXPECT_EQ((*selected)[1].candidate->id, 2);
  EXPECT_EQ((*selected)[1].original_candidate_index, 2);
}

TEST(CandidateRankerModelTest, BuildsCanonicalSharedTrieAndTerminalLeaves) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;
  const CandidateRankerTokenizedSegment segment{
      .segment_id = 7,
      .candidates =
          {
              TokenSequence(20, 0, "b", {1, 10, 20}),
              TokenSequence(21, 1, "a", {1, 10, 21}),
              TokenSequence(22, 2, "c", {1, 10, 20, 30}),
          },
  };

  const auto plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&segment, 1), manifest, cancellation);
  ASSERT_TRUE(plan.ok()) << plan.status();
  EXPECT_EQ(plan->sequence_count, 3);
  EXPECT_EQ(plan->output_row_count, 4);
  ASSERT_EQ(plan->segments.size(), 1);
  EXPECT_EQ(plan->segments[0].segment_id, 7);
  EXPECT_EQ(plan->segments[0].common_prefix_token_count, 2);
  ASSERT_EQ(plan->segments[0].candidates.size(), 3);
  EXPECT_EQ(plan->segments[0].candidates[0].candidate_id, 20);
  EXPECT_EQ(plan->segments[0].candidates[0].sequence_id, 1);
  EXPECT_EQ(plan->segments[0].candidates[1].candidate_id, 21);
  EXPECT_EQ(plan->segments[0].candidates[1].sequence_id, 0);
  EXPECT_EQ(plan->segments[0].candidates[2].sequence_id, 2);

  ASSERT_EQ(plan->inputs.size(), 5);
  EXPECT_EQ(plan->inputs[0].token_id, 1);
  EXPECT_EQ(plan->inputs[0].position, 0);
  EXPECT_EQ(plan->inputs[0].sequence_ids, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_TRUE(plan->inputs[0].scored_edges.empty());
  EXPECT_EQ(plan->inputs[1].token_id, 10);
  ASSERT_EQ(plan->inputs[1].scored_edges.size(), 2);
  EXPECT_EQ(plan->inputs[1].scored_edges[0].target_token_id, 20);
  EXPECT_EQ(plan->inputs[1].scored_edges[0].descendant_sequence_ids,
            (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(plan->inputs[1].scored_edges[1].target_token_id, 21);
  EXPECT_EQ(plan->inputs[1].scored_edges[1].descendant_sequence_ids,
            (std::vector<int32_t>{0}));
  EXPECT_EQ(plan->inputs[2].token_id, 20);
  ASSERT_EQ(plan->inputs[2].scored_edges.size(), 2);
  EXPECT_EQ(plan->inputs[2].scored_edges[0].target_token_id, 2);
  EXPECT_EQ(plan->inputs[2].scored_edges[1].target_token_id, 30);
  EXPECT_EQ(plan->inputs[3].token_id, 21);
  EXPECT_EQ(plan->inputs[4].token_id, 30);
  EXPECT_TRUE(std::none_of(plan->inputs.begin(), plan->inputs.end(),
                           [](const CandidateRankerBatchInput& input) {
                             return input.token_id == 2;
                           }));
}

TEST(CandidateRankerModelTest, DuplicateRecordsShareOneScoredTerminalEdge) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;
  const CandidateRankerTokenizedSegment segment{
      .segment_id = 1,
      .candidates =
          {
              TokenSequence(10, 0, "A", {1, 5}),
              TokenSequence(11, 1, "a", {1, 5}),
          },
  };

  const auto plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&segment, 1), manifest, cancellation);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->segments.size(), 1);
  EXPECT_EQ(plan->segments[0].common_prefix_token_count, 2);
  ASSERT_EQ(plan->inputs.size(), 2);
  ASSERT_EQ(plan->inputs[1].scored_edges.size(), 1);
  EXPECT_EQ(plan->inputs[1].scored_edges[0].target_token_id, 2);
  EXPECT_EQ(plan->inputs[1].scored_edges[0].descendant_sequence_ids,
            (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(plan->output_row_count, 1);
}

TEST(CandidateRankerModelTest, CandidatePermutationKeepsCanonicalBatchPlan) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;
  CandidateRankerTokenizedSegment first{
      .segment_id = 1,
      .candidates =
          {
              TokenSequence(10, 2, "c", {1, 4}),
              TokenSequence(11, 0, "a", {1, 3}),
              TokenSequence(12, 1, "b", {1, 5}),
          },
  };
  CandidateRankerTokenizedSegment second = first;
  std::reverse(second.candidates.begin(), second.candidates.end());

  const auto first_plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&first, 1), manifest, cancellation);
  const auto second_plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&second, 1), manifest, cancellation);
  ASSERT_TRUE(first_plan.ok()) << first_plan.status();
  ASSERT_TRUE(second_plan.ok()) << second_plan.status();
  ExpectBatchPlansEqual(*first_plan, *second_plan);
}

TEST(CandidateRankerModelTest,
     SegmentPermutationKeepsDisjointCanonicalSequenceIds) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;
  const CandidateRankerTokenizedSegment high_segment{
      .segment_id = 9,
      .candidates =
          {
              TokenSequence(91, 0, "b", {1, 91}),
              TokenSequence(90, 1, "a", {1, 90}),
          },
  };
  const CandidateRankerTokenizedSegment low_segment{
      .segment_id = 3,
      .candidates =
          {
              TokenSequence(31, 0, "z", {1, 31}),
              TokenSequence(30, 1, "a", {1, 30}),
          },
  };
  const std::vector<CandidateRankerTokenizedSegment> first = {high_segment,
                                                              low_segment};
  const std::vector<CandidateRankerTokenizedSegment> second = {low_segment,
                                                               high_segment};

  const auto first_plan =
      BuildCandidateRankerBatchPlan(first, manifest, cancellation);
  const auto second_plan =
      BuildCandidateRankerBatchPlan(second, manifest, cancellation);
  ASSERT_TRUE(first_plan.ok()) << first_plan.status();
  ASSERT_TRUE(second_plan.ok()) << second_plan.status();
  ExpectBatchPlansEqual(*first_plan, *second_plan);

  ASSERT_EQ(first_plan->segments.size(), 2);
  EXPECT_EQ(first_plan->segments[0].segment_id, 3);
  EXPECT_EQ(first_plan->segments[0].candidates[0].sequence_id, 1);
  EXPECT_EQ(first_plan->segments[0].candidates[1].sequence_id, 0);
  EXPECT_EQ(first_plan->segments[1].segment_id, 9);
  EXPECT_EQ(first_plan->segments[1].candidates[0].sequence_id, 3);
  EXPECT_EQ(first_plan->segments[1].candidates[1].sequence_id, 2);
  ASSERT_EQ(first_plan->inputs.size(), 6);
  EXPECT_EQ(first_plan->inputs[0].sequence_ids, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(first_plan->inputs[3].sequence_ids, (std::vector<int32_t>{2, 3}));

  CandidateRankerModelManifest limited_manifest = MakeManifest();
  limited_manifest.mutable_limits()->set_max_selected_candidates_per_segment(2);
  limited_manifest.mutable_limits()->set_max_sequences_per_decode(3);
  EXPECT_EQ(BuildCandidateRankerBatchPlan(first, limited_manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
}

TEST(CandidateRankerModelTest, EnforcesAggregateInputNodeBoundary) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_limits()->set_per_record_token_limit(257);
  manifest.mutable_limits()->set_per_decode_input_node_limit(256);
  manifest.mutable_limits()->set_per_decode_output_row_limit(64);
  const TestCancellation cancellation;
  std::vector<int32_t> first_tokens(255, 10);
  std::vector<int32_t> second_tokens(256, 10);
  first_tokens[0] = 1;
  second_tokens[0] = 1;
  second_tokens.back() = 20;
  CandidateRankerTokenizedSegment segment{
      .segment_id = 1,
      .candidates =
          {
              TokenSequence(1, 0, "a", std::move(first_tokens)),
              TokenSequence(2, 1, "b", std::move(second_tokens)),
          },
  };

  const auto boundary_plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&segment, 1), manifest, cancellation);
  ASSERT_TRUE(boundary_plan.ok()) << boundary_plan.status();
  EXPECT_EQ(boundary_plan->inputs.size(), 256);
  EXPECT_EQ(boundary_plan->output_row_count, 2);

  segment.candidates[0].tokens.push_back(30);
  EXPECT_EQ(BuildCandidateRankerBatchPlan(absl::MakeConstSpan(&segment, 1),
                                          manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
}

TEST(CandidateRankerModelTest,
     PacksWholeSegmentsGreedilyInSegmentIdOrder) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_limits()->set_max_selected_candidates_per_segment(2);
  manifest.mutable_limits()->set_max_sequences_per_decode(4);
  const TestCancellation cancellation;
  const std::vector<CandidateRankerTokenizedSegment> segments = {
      {.segment_id = 30,
       .candidates = {TokenSequence(30, 0, "a", {1, 30}),
                      TokenSequence(31, 1, "b", {1, 31})}},
      {.segment_id = 10,
       .candidates = {TokenSequence(10, 0, "a", {1, 10}),
                      TokenSequence(11, 1, "b", {1, 11})}},
      {.segment_id = 20,
       .candidates = {TokenSequence(20, 0, "a", {1, 20}),
                      TokenSequence(21, 1, "b", {1, 21})}},
  };

  const auto batches = PackCandidateRankerSegmentsForDecode(
      segments, 32, manifest, cancellation);
  ASSERT_TRUE(batches.ok()) << batches.status();
  ASSERT_EQ(batches->size(), 2);
  ASSERT_EQ((*batches)[0].plan.segments.size(), 2);
  EXPECT_EQ((*batches)[0].plan.segments[0].segment_id, 10);
  EXPECT_EQ((*batches)[0].plan.segments[1].segment_id, 20);
  ASSERT_EQ((*batches)[1].plan.segments.size(), 1);
  EXPECT_EQ((*batches)[1].plan.segments[0].segment_id, 30);
  EXPECT_EQ((*batches)[0].plan.sequence_count, 4);
  EXPECT_EQ((*batches)[1].plan.sequence_count, 2);
  EXPECT_EQ((*batches)[1].plan.segments[0].candidates[0].sequence_id, 0);
  EXPECT_EQ((*batches)[1].plan.segments[0].candidates[1].sequence_id, 1);
}

TEST(CandidateRankerModelTest, EnforcesAggregateOutputRowBoundary) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_limits()->set_per_record_token_limit(131);
  manifest.mutable_limits()->set_per_decode_input_node_limit(512);
  manifest.mutable_limits()->set_per_decode_output_row_limit(256);
  const TestCancellation cancellation;
  std::vector<int32_t> first_tokens(128, 10);
  std::vector<int32_t> second_tokens(129, 20);
  first_tokens[0] = 1;
  second_tokens[0] = 1;
  CandidateRankerTokenizedSegment segment{
      .segment_id = 1,
      .candidates =
          {
              TokenSequence(1, 0, "a", std::move(first_tokens)),
              TokenSequence(2, 1, "b", std::move(second_tokens)),
          },
  };

  const auto boundary_plan = BuildCandidateRankerBatchPlan(
      absl::MakeConstSpan(&segment, 1), manifest, cancellation);
  ASSERT_TRUE(boundary_plan.ok()) << boundary_plan.status();
  EXPECT_EQ(boundary_plan->inputs.size(), 256);
  EXPECT_EQ(boundary_plan->output_row_count, 256);

  segment.candidates[1].tokens.push_back(20);
  EXPECT_EQ(BuildCandidateRankerBatchPlan(absl::MakeConstSpan(&segment, 1),
                                          manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
}

TEST(CandidateRankerModelTest, EnforcesSequenceTokenAndTrieLimits) {
  CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation;
  manifest.mutable_limits()->set_per_record_token_limit(3);
  CandidateRankerTokenizedSegment segment{
      .segment_id = 1,
      .candidates =
          {
              TokenSequence(1, 0, "a", {1, 3}),
              TokenSequence(2, 1, "b", {1, 4}),
          },
  };
  EXPECT_TRUE(BuildCandidateRankerBatchPlan(absl::MakeConstSpan(&segment, 1),
                                            manifest, cancellation)
                  .ok());
  segment.candidates[1].tokens.push_back(5);
  EXPECT_EQ(BuildCandidateRankerBatchPlan(absl::MakeConstSpan(&segment, 1),
                                          manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);

  manifest = MakeManifest();
  manifest.mutable_limits()->set_max_selected_candidates_per_segment(2);
  manifest.mutable_limits()->set_max_sequences_per_decode(8);
  manifest.mutable_limits()->set_per_decode_output_row_limit(8);
  segment.candidates = {
      TokenSequence(1, 0, "a", {1, 10, 11, 12, 13, 14}),
      TokenSequence(2, 1, "b", {1, 20, 21, 22, 23, 24}),
  };
  EXPECT_EQ(BuildCandidateRankerBatchPlan(absl::MakeConstSpan(&segment, 1),
                                          manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
}

TEST(CandidateRankerModelTest, RejectsCancelledPlanning) {
  const CandidateRankerModelManifest manifest = MakeManifest();
  const TestCancellation cancellation(true);
  const std::vector<CandidateRankerTokenizedSegment> segments;
  EXPECT_EQ(BuildCandidateRankerBatchPlan(segments, manifest, cancellation)
                .status()
                .code(),
            absl::StatusCode::kCancelled);
}

TEST(CandidateRankerModelTest, LocatesFollowingContextAfterBoundaryRetokenizing) {
  const auto exact_prefix = FindCandidateRankerFollowingContextStartPosition(
      std::vector<int32_t>{1, 10, 20, 30},
      std::vector<int32_t>{1, 10, 20});
  ASSERT_TRUE(exact_prefix.ok()) << exact_prefix.status();
  EXPECT_EQ(*exact_prefix, 3);

  const auto retokenized_boundary =
      FindCandidateRankerFollowingContextStartPosition(
          std::vector<int32_t>{1, 10, 99, 30},
          std::vector<int32_t>{1, 10, 20});
  ASSERT_TRUE(retokenized_boundary.ok()) << retokenized_boundary.status();
  EXPECT_EQ(*retokenized_boundary, 3);

  const auto complete_sequence_ended =
      FindCandidateRankerFollowingContextStartPosition(
          std::vector<int32_t>{1, 10}, std::vector<int32_t>{1, 10, 20});
  ASSERT_TRUE(complete_sequence_ended.ok())
      << complete_sequence_ended.status();
  EXPECT_EQ(*complete_sequence_ended, 2);

  EXPECT_EQ(FindCandidateRankerFollowingContextStartPosition(
                std::vector<int32_t>{1}, std::vector<int32_t>{2})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CandidateRankerModelTest,
     AttributionBoundaryDoesNotApplyTheProductionTokenLimit) {
  std::vector<int32_t> candidate_ending_tokens(256, 20);
  candidate_ending_tokens[0] = 1;
  const auto boundary = FindCandidateRankerFollowingContextStartPosition(
      std::vector<int32_t>{1, 10}, candidate_ending_tokens);
  ASSERT_TRUE(boundary.ok()) << boundary.status();
  EXPECT_EQ(*boundary, 2);
}

TEST(CandidateRankerModelTest,
     AccumulatesNormalizedLogProbabilitiesOncePerRow) {
  const CandidateRankerBatchInput input{
      .scored_edges =
          {
              {.target_token_id = 0, .descendant_sequence_ids = {0, 2}},
              {.target_token_id = 1, .descendant_sequence_ids = {1}},
          },
  };
  const std::vector<float> logits = {1000.0f, 999.0f};
  std::vector<double> scores(3, 0.0);

  const auto edge_log_probabilities =
      ComputeCandidateRankerEdgeLogProbabilities(input, logits);
  ASSERT_TRUE(edge_log_probabilities.ok())
      << edge_log_probabilities.status();
  ASSERT_EQ(edge_log_probabilities->size(), 2);
  ASSERT_TRUE(
      AccumulateCandidateRankerEdgeLogProbabilities(
          input, *edge_log_probabilities, absl::MakeSpan(scores))
          .ok());
  const double expected_first = -std::log1p(std::exp(-1.0));
  const double expected_second = -1.0 - std::log1p(std::exp(-1.0));
  EXPECT_NEAR((*edge_log_probabilities)[0], expected_first, 1e-12);
  EXPECT_NEAR((*edge_log_probabilities)[1], expected_second, 1e-12);
  EXPECT_NEAR(scores[0], expected_first, 1e-12);
  EXPECT_NEAR(scores[1], expected_second, 1e-12);
  EXPECT_NEAR(scores[2], expected_first, 1e-12);
}

TEST(CandidateRankerModelTest, RejectsInvalidLogitsEdgesAndScores) {
  CandidateRankerBatchInput input{
      .scored_edges =
          {
              {.target_token_id = 0, .descendant_sequence_ids = {0}},
          },
  };
  std::vector<double> scores(1, 0.0);
  EXPECT_EQ(AccumulateCandidateRankerOutput(
                input, {std::numeric_limits<float>::infinity()},
                absl::MakeSpan(scores))
                .code(),
            absl::StatusCode::kInvalidArgument);

  input.scored_edges[0].target_token_id = 1;
  EXPECT_EQ(
      AccumulateCandidateRankerOutput(input, {0.0f}, absl::MakeSpan(scores))
          .code(),
      absl::StatusCode::kInvalidArgument);

  input.scored_edges = {
      {.target_token_id = 0, .descendant_sequence_ids = {0}},
      {.target_token_id = 1, .descendant_sequence_ids = {0}},
  };
  EXPECT_EQ(AccumulateCandidateRankerOutput(input, {0.0f, 0.0f},
                                            absl::MakeSpan(scores))
                .code(),
            absl::StatusCode::kInvalidArgument);

  const TestCancellation cancellation(true);
  EXPECT_EQ(ComputeCandidateRankerEdgeLogProbabilities(
                input, {0.0f, 0.0f}, &cancellation)
                .status()
                .code(),
            absl::StatusCode::kCancelled);
}

TEST(CandidateRankerModelTest, OrdersScoresAndPreservesOriginalTies) {
  CandidateRankerBatchPlan plan;
  plan.sequence_count = 3;
  plan.segments = {{
      .segment_id = 9,
      .candidates =
          {
              {.candidate_id = 20,
               .original_candidate_index = 0,
               .sequence_id = 1},
              {.candidate_id = 21,
               .original_candidate_index = 1,
               .sequence_id = 0},
              {.candidate_id = 22,
               .original_candidate_index = 2,
               .sequence_id = 2},
          },
  }};
  const std::vector<double> scores = {1.0, 1.0, 2.0};

  const auto orders = OrderCandidateRankerContinuations(plan, scores);
  ASSERT_TRUE(orders.ok()) << orders.status();
  ASSERT_EQ(orders->size(), 1);
  EXPECT_EQ((*orders)[0].segment_id, 9);
  EXPECT_EQ((*orders)[0].candidate_ids,
            (std::vector<converter::CandidateRankerCandidateId>{22, 20, 21}));

  EXPECT_EQ(OrderCandidateRankerContinuations(plan, {1.0, 2.0}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(OrderCandidateRankerContinuations(
                plan, {1.0, 2.0, std::numeric_limits<double>::quiet_NaN()})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace engine
}  // namespace mozc
