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

#include "engine/evaluation/ajimee_semantic_results.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/evaluation/ajimee_semantic_results.pb.h"
#include "engine/evaluation/frozen_candidate_ranker_util.h"

namespace mozc::engine::evaluation {
namespace {

bool CapacityIdentityEquals(const AjimeeCapacityAuditIdentity& lhs,
                            const AjimeeCapacityAuditIdentity& rhs) {
  return lhs.frozen_corpus_sha256() == rhs.frozen_corpus_sha256() &&
         lhs.model_manifest_sha256() == rhs.model_manifest_sha256() &&
         lhs.gguf_file_name() == rhs.gguf_file_name() &&
         lhs.gguf_sha256() == rhs.gguf_sha256() &&
         lhs.tokenizer_sha256() == rhs.tokenizer_sha256() &&
         lhs.quantization() == rhs.quantization() &&
         lhs.source_revision() == rhs.source_revision() &&
         lhs.runtime_revision() == rhs.runtime_revision();
}

void CopyCapacityIdentity(const AjimeeCapacityAuditIdentity& source,
                          AjimeeCapacityAuditIdentity* destination) {
  destination->set_frozen_corpus_sha256(source.frozen_corpus_sha256());
  destination->set_model_manifest_sha256(source.model_manifest_sha256());
  destination->set_gguf_file_name(source.gguf_file_name());
  destination->set_gguf_sha256(source.gguf_sha256());
  destination->set_tokenizer_sha256(source.tokenizer_sha256());
  destination->set_quantization(source.quantization());
  destination->set_source_revision(source.source_revision());
  destination->set_runtime_revision(source.runtime_revision());
}

absl::Status ValidateSemanticInputs(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config) {
  absl::Status status = ValidateAjimeeSemanticRunnerConfig(config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeCapacityAuditConfig(capacity_config);
  if (!status.ok()) {
    return status;
  }
  if (HasUnknownFieldsRecursively(corpus)) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus contains unknown fields");
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeCapacityAudit(capacity_audit, corpus,
                                       capacity_config);
  if (!status.ok()) {
    return status;
  }
  if (!capacity_audit.all_passed()) {
    return absl::FailedPreconditionError(
        "AJIMEE semantic runner requires a passing capacity audit");
  }
  return absl::OkStatus();
}

absl::Status ValidateSemanticIdentity(
    const AjimeeSemanticResultsIdentity& identity,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config) {
  if (!identity.IsInitialized() || HasUnknownFieldsRecursively(identity) ||
      !IsAjimeeLowercaseHex(identity.frozen_corpus_sha256(), 64) ||
      !IsAjimeeLowercaseHex(identity.capacity_audit_config_sha256(), 64) ||
      !IsAjimeeLowercaseHex(identity.capacity_audit_sha256(), 64) ||
      identity.numeric_profile() != AJIMEE_SEMANTIC_CONFIGURED) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic results identity is invalid");
  }
  if (identity.frozen_corpus_sha256() !=
          capacity_config.frozen_corpus_sha256() ||
      identity.capacity_audit_config_sha256() !=
          config.capacity_audit_config_sha256() ||
      identity.capacity_audit_sha256() != config.capacity_audit_sha256() ||
      identity.numeric_profile() != config.numeric_profile() ||
      !CapacityIdentityEquals(identity.capacity_audit_identity(),
                              capacity_audit.identity()) ||
      identity.frozen_corpus_sha256() !=
          identity.capacity_audit_identity().frozen_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE semantic results identity does not match its inputs");
  }
  return absl::OkStatus();
}

struct ResponseShape {
  uint64_t segment_order_count = 0;
  uint64_t candidate_id_count = 0;
  bool within_request_bounds = false;
};

absl::Status AddSizeToCount(size_t size, uint64_t* count) {
  if (size > std::numeric_limits<uint64_t>::max() - *count) {
    return absl::OutOfRangeError(
        "AJIMEE semantic response count overflows uint64");
  }
  *count += static_cast<uint64_t>(size);
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RequestCandidateCount(
    const converter::CandidateRankerRequest& request) {
  uint64_t count = 0;
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    absl::Status status = AddSizeToCount(segment.candidates.size(), &count);
    if (!status.ok()) {
      return status;
    }
  }
  return count;
}

absl::StatusOr<ResponseShape> ComputeResponseShape(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerResponse& response) {
  ResponseShape shape;
  absl::Status status =
      AddSizeToCount(response.segment_orders.size(),
                     &shape.segment_order_count);
  if (!status.ok()) {
    return status;
  }
  for (const converter::CandidateRankerSegmentOrder& order :
       response.segment_orders) {
    status = AddSizeToCount(order.candidate_ids.size(),
                            &shape.candidate_id_count);
    if (!status.ok()) {
      return status;
    }
  }
  absl::StatusOr<uint64_t> request_candidate_count =
      RequestCandidateCount(request);
  if (!request_candidate_count.ok()) {
    return request_candidate_count.status();
  }
  shape.within_request_bounds =
      shape.segment_order_count <= request.segments.size() &&
      shape.candidate_id_count <= *request_candidate_count;
  return shape;
}

absl::Status ValidateStoredResponseShape(
    const AjimeeSemanticCaseResult& result,
    const converter::CandidateRankerRequest& request) {
  if (!result.has_response_token_matches_request() ||
      !result.has_response_within_request_bounds() ||
      !result.has_response_segment_order_count() ||
      !result.has_response_candidate_id_count()) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic response metadata is incomplete");
  }
  absl::StatusOr<uint64_t> request_candidate_count =
      RequestCandidateCount(request);
  if (!request_candidate_count.ok()) {
    return request_candidate_count.status();
  }
  const bool expected_within_request_bounds =
      result.response_segment_order_count() <= request.segments.size() &&
      result.response_candidate_id_count() <= *request_candidate_count;
  if (result.response_within_request_bounds() !=
      expected_within_request_bounds) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic response bounds metadata is inconsistent");
  }
  if (!result.response_within_request_bounds()) {
    if (result.response_segment_orders_size() != 0) {
      return absl::InvalidArgumentError(
          "AJIMEE out-of-bounds response stores raw orders");
    }
    return absl::OkStatus();
  }
  if (result.response_segment_order_count() !=
      static_cast<uint64_t>(result.response_segment_orders_size())) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic response segment count is inconsistent");
  }
  uint64_t stored_candidate_count = 0;
  for (const AjimeeSemanticSegmentOrder& order :
       result.response_segment_orders()) {
    absl::Status status = AddSizeToCount(order.candidate_ids_size(),
                                         &stored_candidate_count);
    if (!status.ok()) {
      return status;
    }
  }
  if (result.response_candidate_id_count() != stored_candidate_count) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic response candidate count is inconsistent");
  }
  return absl::OkStatus();
}

converter::CandidateRankerResponse RestoreResponse(
    const converter::CandidateRankerRequest& request,
    const AjimeeSemanticCaseResult& result) {
  converter::CandidateRankerResponse response{.token = request.token};
  response.segment_orders.reserve(result.response_segment_orders_size());
  for (const AjimeeSemanticSegmentOrder& stored_order :
       result.response_segment_orders()) {
    converter::CandidateRankerSegmentOrder order{
        .segment_id = stored_order.segment_id(),
    };
    order.candidate_ids.reserve(stored_order.candidate_ids_size());
    for (const uint64_t candidate_id : stored_order.candidate_ids()) {
      order.candidate_ids.push_back(candidate_id);
    }
    response.segment_orders.push_back(std::move(order));
  }
  return response;
}

void CopyResponseOrders(const converter::CandidateRankerResponse& response,
                        AjimeeSemanticCaseResult* result) {
  for (const converter::CandidateRankerSegmentOrder& source_order :
       response.segment_orders) {
    AjimeeSemanticSegmentOrder* destination_order =
        result->add_response_segment_orders();
    destination_order->set_segment_id(source_order.segment_id);
    for (const uint64_t candidate_id : source_order.candidate_ids) {
      destination_order->add_candidate_ids(candidate_id);
    }
  }
}

absl::StatusOr<std::string> ComputeMergedTopOutput(
    const converter::CandidateRankerRequest& request,
    const converter::CandidateRankerMergeOrder& merge_order) {
  if (merge_order.segment_orders.size() != request.segments.size()) {
    return absl::InternalError(
        "AJIMEE semantic merge does not cover every request segment");
  }
  std::string output;
  for (size_t segment_index = 0; segment_index < request.segments.size();
       ++segment_index) {
    const converter::CandidateRankerSegment& segment =
        request.segments[segment_index];
    const converter::CandidateRankerSegmentOrder& order =
        merge_order.segment_orders[segment_index];
    if (order.segment_id != segment.id ||
        order.candidate_ids.size() != segment.candidates.size() ||
        order.candidate_ids.empty()) {
      return absl::InternalError(
          "AJIMEE semantic complete merge order is invalid");
    }
    const uint64_t top_candidate_id = order.candidate_ids.front();
    const converter::CandidateRankerCandidate* top_candidate = nullptr;
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (candidate.id == top_candidate_id) {
        top_candidate = &candidate;
        break;
      }
    }
    if (top_candidate == nullptr) {
      return absl::InternalError(
          "AJIMEE semantic top candidate is unavailable");
    }
    output.append(top_candidate->value);
  }
  return output;
}

absl::Status ValidateCaseResult(
    const AjimeeSemanticCaseResult& result,
    const converter::CandidateRankerRequest& request) {
  if (!IsKnownAjimeeCanonicalStatusCode(result.canonical_status_code())) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic canonical status code is invalid");
  }
  if (result.outcome() == AJIMEE_SEMANTIC_BACKEND_ERROR) {
    if (result.canonical_status_code() == AJIMEE_STATUS_OK ||
        result.has_response_token_matches_request() ||
        result.response_segment_orders_size() != 0 ||
        result.has_merged_top_output() ||
        result.has_response_within_request_bounds() ||
        result.has_response_segment_order_count() ||
        result.has_response_candidate_id_count()) {
      return absl::InvalidArgumentError(
          "AJIMEE semantic backend error is inconsistent");
    }
    return absl::OkStatus();
  }
  if (result.outcome() != AJIMEE_SEMANTIC_SUCCESS &&
      result.outcome() != AJIMEE_SEMANTIC_INVALID_RESPONSE) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic outcome is unspecified");
  }
  absl::Status status = ValidateStoredResponseShape(result, request);
  if (!status.ok()) {
    return status;
  }
  if (!result.response_token_matches_request()) {
    if (result.outcome() != AJIMEE_SEMANTIC_INVALID_RESPONSE ||
        result.canonical_status_code() != AJIMEE_STATUS_ABORTED ||
        result.has_merged_top_output()) {
      return absl::InvalidArgumentError(
          "AJIMEE semantic token mismatch is inconsistent");
    }
    return absl::OkStatus();
  }
  if (!result.response_within_request_bounds()) {
    if (result.outcome() != AJIMEE_SEMANTIC_INVALID_RESPONSE ||
        result.canonical_status_code() != AJIMEE_STATUS_INVALID_ARGUMENT ||
        result.has_merged_top_output()) {
      return absl::InvalidArgumentError(
          "AJIMEE out-of-bounds semantic response is inconsistent");
    }
    return absl::OkStatus();
  }

  converter::CandidateRankerResponse response =
      RestoreResponse(request, result);
  absl::StatusOr<converter::CandidateRankerMergeOrder> merge_order =
      converter::BuildCandidateRankerMergeOrder(
          request, response, request.token);
  if (result.outcome() == AJIMEE_SEMANTIC_SUCCESS) {
    if (result.canonical_status_code() != AJIMEE_STATUS_OK ||
        !result.has_merged_top_output() || !merge_order.ok()) {
      return absl::InvalidArgumentError(
          "AJIMEE successful semantic result is inconsistent");
    }
    absl::StatusOr<std::string> output =
        ComputeMergedTopOutput(request, *merge_order);
    if (!output.ok()) {
      return output.status();
    }
    if (*output != result.merged_top_output()) {
      return absl::InvalidArgumentError(
          "AJIMEE semantic merged top output is inconsistent");
    }
    return absl::OkStatus();
  }
  if (result.canonical_status_code() == AJIMEE_STATUS_OK ||
      result.has_merged_top_output() || merge_order.ok()) {
    return absl::InvalidArgumentError(
        "AJIMEE invalid semantic response is inconsistent");
  }
  absl::StatusOr<AjimeeCanonicalStatusCode> canonical =
      ConvertAjimeeCanonicalStatusCode(merge_order.status().code());
  if (!canonical.ok()) {
    return canonical.status();
  }
  if (*canonical != result.canonical_status_code()) {
    return absl::InvalidArgumentError(
        "AJIMEE invalid semantic response status is inconsistent");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateAjimeeSemanticRunnerConfig(
    const AjimeeSemanticRunnerConfig& config) {
  if (!config.IsInitialized() || HasUnknownFieldsRecursively(config)) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic runner config is incomplete");
  }
  if (config.schema_version() !=
          kAjimeeSemanticRunnerConfigSchemaVersion ||
      config.semantic_results_schema_version() !=
          kAjimeeSemanticResultsSchemaVersion) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic runner config schema mismatch");
  }
  if (!IsAjimeeLowercaseHex(config.capacity_audit_config_sha256(), 64) ||
      !IsAjimeeLowercaseHex(config.capacity_audit_sha256(), 64) ||
      config.numeric_profile() != AJIMEE_SEMANTIC_CONFIGURED) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic runner config identity is invalid");
  }
  return absl::OkStatus();
}

absl::Status ValidateAjimeeSemanticResults(
    const AjimeeSemanticResults& results,
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config) {
  absl::Status status = ValidateSemanticInputs(
      corpus, capacity_config, capacity_audit, config);
  if (!status.ok()) {
    return status;
  }
  if (!results.IsInitialized() || HasUnknownFieldsRecursively(results)) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic results are incomplete");
  }
  if (results.schema_version() != kAjimeeSemanticResultsSchemaVersion ||
      results.schema_version() != config.semantic_results_schema_version()) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic results schema mismatch");
  }
  status = ValidateSemanticIdentity(results.identity(), capacity_config,
                                    capacity_audit, config);
  if (!status.ok()) {
    return status;
  }
  if (results.cases_size() != corpus.cases_size()) {
    return absl::InvalidArgumentError(
        "AJIMEE semantic result case count is invalid");
  }
  for (int case_index = 0; case_index < results.cases_size(); ++case_index) {
    const AjimeeSemanticCaseResult& result = results.cases(case_index);
    const AjimeeFrozenCase& frozen_case = corpus.cases(case_index);
    if (result.source_index() != frozen_case.source_index()) {
      return absl::InvalidArgumentError(
          "AJIMEE semantic results are unordered");
    }
    absl::StatusOr<converter::CandidateRankerRequest> request =
        ConvertFrozenCandidateRankerRequest(frozen_case.request());
    if (!request.ok()) {
      return request.status();
    }
    status = ValidateCaseResult(result, *request);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<AjimeeSemanticResults> BuildAjimeeSemanticResults(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit,
    const AjimeeSemanticRunnerConfig& config,
    const AjimeeSemanticResultsIdentity& identity,
    AjimeeSemanticRankCallback rank_callback) {
  absl::Status status = ValidateSemanticInputs(
      corpus, capacity_config, capacity_audit, config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateSemanticIdentity(identity, capacity_config,
                                    capacity_audit, config);
  if (!status.ok()) {
    return status;
  }

  AjimeeSemanticResults results;
  results.set_schema_version(kAjimeeSemanticResultsSchemaVersion);
  AjimeeSemanticResultsIdentity* results_identity =
      results.mutable_identity();
  results_identity->set_frozen_corpus_sha256(
      identity.frozen_corpus_sha256());
  results_identity->set_capacity_audit_config_sha256(
      identity.capacity_audit_config_sha256());
  results_identity->set_capacity_audit_sha256(
      identity.capacity_audit_sha256());
  CopyCapacityIdentity(identity.capacity_audit_identity(),
                       results_identity->mutable_capacity_audit_identity());
  results_identity->set_numeric_profile(identity.numeric_profile());

  for (const AjimeeFrozenCase& frozen_case : corpus.cases()) {
    absl::StatusOr<converter::CandidateRankerRequest> request =
        ConvertFrozenCandidateRankerRequest(frozen_case.request());
    if (!request.ok()) {
      return request.status();
    }
    AjimeeSemanticCaseResult* result = results.add_cases();
    result->set_source_index(frozen_case.source_index());
    absl::StatusOr<converter::CandidateRankerResponse> backend_result =
        rank_callback(*request);
    if (!backend_result.ok()) {
      absl::StatusOr<AjimeeCanonicalStatusCode> canonical =
          ConvertAjimeeCanonicalStatusCode(backend_result.status().code());
      if (!canonical.ok()) {
        return canonical.status();
      }
      result->set_outcome(AJIMEE_SEMANTIC_BACKEND_ERROR);
      result->set_canonical_status_code(*canonical);
      continue;
    }

    absl::StatusOr<ResponseShape> response_shape =
        ComputeResponseShape(*request, *backend_result);
    if (!response_shape.ok()) {
      return response_shape.status();
    }
    const bool token_matches = backend_result->token == request->token;
    result->set_response_token_matches_request(token_matches);
    result->set_response_within_request_bounds(
        response_shape->within_request_bounds);
    result->set_response_segment_order_count(
        response_shape->segment_order_count);
    result->set_response_candidate_id_count(
        response_shape->candidate_id_count);
    if (response_shape->within_request_bounds) {
      CopyResponseOrders(*backend_result, result);
    }
    if (!token_matches) {
      result->set_outcome(AJIMEE_SEMANTIC_INVALID_RESPONSE);
      result->set_canonical_status_code(AJIMEE_STATUS_ABORTED);
      continue;
    }
    if (!response_shape->within_request_bounds) {
      result->set_outcome(AJIMEE_SEMANTIC_INVALID_RESPONSE);
      result->set_canonical_status_code(AJIMEE_STATUS_INVALID_ARGUMENT);
      continue;
    }
    absl::StatusOr<converter::CandidateRankerMergeOrder> merge_order =
        converter::BuildCandidateRankerMergeOrder(
            *request, *backend_result, request->token);
    if (!merge_order.ok()) {
      absl::StatusOr<AjimeeCanonicalStatusCode> canonical =
          ConvertAjimeeCanonicalStatusCode(merge_order.status().code());
      if (!canonical.ok()) {
        return canonical.status();
      }
      result->set_outcome(AJIMEE_SEMANTIC_INVALID_RESPONSE);
      result->set_canonical_status_code(*canonical);
      continue;
    }
    absl::StatusOr<std::string> output =
        ComputeMergedTopOutput(*request, *merge_order);
    if (!output.ok()) {
      return output.status();
    }
    result->set_outcome(AJIMEE_SEMANTIC_SUCCESS);
    result->set_canonical_status_code(AJIMEE_STATUS_OK);
    result->set_merged_top_output(*output);
  }

  status = ValidateAjimeeSemanticResults(
      results, corpus, capacity_config, capacity_audit, config);
  if (!status.ok()) {
    return status;
  }
  return results;
}

}  // namespace mozc::engine::evaluation
