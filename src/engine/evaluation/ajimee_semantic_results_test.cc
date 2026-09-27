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

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/file_util.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/evaluation/ajimee_semantic_results.pb.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kFrozenSha256[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kManifestSha256[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kGgufSha256[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kTokenizerSha256[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr char kSourceRevision[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr char kRuntimeRevision[] =
    "ffffffffffffffffffffffffffffffffffffffff";
constexpr char kCapacityConfigSha256[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kCapacityAuditSha256[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr char kPrivacyPoison[] =
    "PRIVATE_AJIMEE_SEMANTIC_REQUEST_MUST_NOT_ENTER_THE_ARTIFACT";
constexpr char kStatusPoison[] =
    "PRIVATE_AJIMEE_SEMANTIC_STATUS_MUST_NOT_ENTER_THE_ARTIFACT";
constexpr char kUnknownFieldPoison[] =
    "PRIVATE_AJIMEE_SEMANTIC_UNKNOWN_FIELD_MUST_BE_REJECTED";

template <typename Message>
void AddUnknownFieldPoison(Message* message) {
  std::string serialized;
  ASSERT_TRUE(message->SerializeToString(&serialized));
  serialized.append("\xA2\x06", 2);
  static_assert(sizeof(kUnknownFieldPoison) - 1 < 128);
  serialized.push_back(static_cast<char>(sizeof(kUnknownFieldPoison) - 1));
  serialized.append(kUnknownFieldPoison);
  ASSERT_TRUE(message->ParseFromString(serialized));
}

void FillSourceIdentity(EvaluationSourceIdentity* source) {
  source->set_benchmark_name("AJIMEE-Bench synthetic");
  source->set_source_revision(kSourceRevision);
  source->set_source_relative_path("synthetic/evaluation_items.json");
  source->set_source_sha256(kFrozenSha256);
  source->set_creator("Synthetic test creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("CC-BY-SA-3.0");
  source->set_license_url("https://creativecommons.org/licenses/by-sa/3.0/");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic transformed corpus.");
}

void AddCandidate(FrozenCandidateRankerSegment* segment, uint64_t id,
                  const std::string& value, bool is_protected = false) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key("key");
  candidate->set_value(value);
  candidate->set_cost(static_cast<int32_t>(id));
  candidate->set_attributes(static_cast<uint32_t>(id + 1));
  candidate->set_consumed_key_size(static_cast<uint32_t>(id + 2));
  candidate->set_is_protected(is_protected);
}

AjimeeFrozenCorpus MakeCorpus(int case_count) {
  AjimeeFrozenCorpus corpus;
  corpus.set_schema_version(kAjimeeFrozenCorpusSchemaVersion);
  FillSourceIdentity(corpus.mutable_source());
  corpus.set_input_corpus_sha256(kFrozenSha256);
  EvaluationMozcIdentity* mozc = corpus.mutable_mozc();
  mozc->set_source_revision(kSourceRevision);
  mozc->set_data_type("oss");
  mozc->set_data_sha256(kFrozenSha256);
  mozc->set_default_desktop_request_sha256(kFrozenSha256);
  mozc->set_default_desktop_config_sha256(kFrozenSha256);
  mozc->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");

  for (int case_index = 0; case_index < case_count; ++case_index) {
    AjimeeFrozenCase* frozen_case = corpus.add_cases();
    frozen_case->set_source_index(static_cast<uint64_t>(case_index) * 10 + 1);
    frozen_case->set_normalized_hiragana_reading("よみ");
    frozen_case->set_history_reconstructed(false);
    FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
    request->mutable_token()->set_session_generation(0);
    request->mutable_token()->set_state_revision(0);
    request->mutable_token()->set_request_sequence(case_index + 1);
    request->set_mode(FrozenCandidateRankerRequest::MODE_CONVERSION);
    request->set_preceding_text(kPrivacyPoison);
    request->set_following_text("");
    request->set_reading("よみ");
    request->set_focused_segment_id(0);

    FrozenCandidateRankerSegment* first = request->add_segments();
    first->set_id(0);
    first->set_key("first-segment");
    AddCandidate(first, 0, "baseline-zero");
    AddCandidate(first, 1, "protected-middle", true);
    AddCandidate(first, 2, "winner-zero");
    AddCandidate(first, 3, kPrivacyPoison);

    FrozenCandidateRankerSegment* second = request->add_segments();
    second->set_id(1);
    second->set_key("second-segment");
    AddCandidate(second, 4, "baseline-one");
    AddCandidate(second, 5, "winner-one");
    AddCandidate(second, 6, "candidate-one-three");
    frozen_case->set_baseline_output("baseline-zerobaseline-one");
  }
  return corpus;
}

AjimeeCapacityAuditConfig MakeCapacityConfig(uint32_t case_count) {
  AjimeeCapacityAuditConfig config;
  config.set_schema_version(kAjimeeCapacityAuditConfigSchemaVersion);
  config.set_frozen_corpus_schema_version(kAjimeeFrozenCorpusSchemaVersion);
  config.set_capacity_audit_schema_version(kAjimeeCapacityAuditSchemaVersion);
  config.set_frozen_corpus_sha256(kFrozenSha256);
  config.set_expected_case_count(case_count);
  config.set_model_manifest_schema_version(
      kAjimeeCapacityAuditModelManifestSchemaVersion);
  config.set_model_manifest_sha256(kManifestSha256);
  config.set_gguf_file_name("model-q4_k_m.gguf");
  config.set_gguf_sha256(kGgufSha256);
  config.set_tokenizer_sha256(kTokenizerSha256);
  config.set_quantization("Q4_K_M");
  config.set_source_revision(kSourceRevision);
  config.set_runtime_revision(kRuntimeRevision);
  return config;
}

AjimeeCapacityAuditIdentity MakeCapacityIdentity() {
  AjimeeCapacityAuditIdentity identity;
  identity.set_frozen_corpus_sha256(kFrozenSha256);
  identity.set_model_manifest_sha256(kManifestSha256);
  identity.set_gguf_file_name("model-q4_k_m.gguf");
  identity.set_gguf_sha256(kGgufSha256);
  identity.set_tokenizer_sha256(kTokenizerSha256);
  identity.set_quantization("Q4_K_M");
  identity.set_source_revision(kSourceRevision);
  identity.set_runtime_revision(kRuntimeRevision);
  return identity;
}

uint64_t RequestStringBytes(
    const converter::CandidateRankerRequest& request) {
  uint64_t result = request.preceding_text.size() +
                    request.following_text.size() + request.reading.size();
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    result += segment.key.size();
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      result += candidate.key.size() + candidate.value.size();
    }
  }
  return result;
}

CandidateRankerCapacityResult MakePassingCapacityResult(
    const converter::CandidateRankerRequest& request) {
  CandidateRankerCapacityResult result;
  CandidateRankerCapacityUsage& usage = result.usage;
  usage.request_segment_count = request.segments.size();
  usage.request_string_bytes = RequestStringBytes(request);
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    usage.request_candidate_count += segment.candidates.size();
    uint64_t unprotected = 0;
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (!candidate.is_protected) {
        ++unprotected;
      }
    }
    usage.request_unprotected_candidate_count += unprotected;
    usage.max_unprotected_candidates_in_segment =
        std::max(usage.max_unprotected_candidates_in_segment, unprotected);
    usage.window_candidate_count += unprotected;
    if (unprotected <= 1) {
      continue;
    }
    ++usage.request_rankable_segment_count;
    ++usage.selected_segment_count;
    usage.selected_candidate_count += unprotected - 1;
    ++usage.candidates_omitted_by_capacity;
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = segment.id,
        .unprotected_candidate_count = unprotected,
        .window_candidate_count = unprotected,
        .selected_candidate_count = unprotected - 1,
        .omitted_by_window = 0,
        .omitted_by_capacity = 1,
        .disposition = CandidateRankerSegmentCapacityDisposition::kSelected,
        .first_limiter = CandidateRankerCapacityViolation{
            .limit = CandidateRankerCapacityLimit::kFullRecordTokens,
            .observed = 21,
            .allowed = 20,
        },
    });
  }
  usage.decode_batch_count = 1;
  usage.max_selected_serialized_record_bytes = 80;
  usage.max_selected_normalized_record_bytes = 81;
  usage.max_selected_full_record_tokens = 20;
  usage.max_segments_in_decode = usage.selected_segment_count;
  usage.max_sequences_in_decode = usage.selected_candidate_count;
  usage.max_input_nodes_in_decode = 16;
  usage.max_output_rows_in_decode = usage.selected_candidate_count;
  usage.max_reserved_output_rows_in_decode = usage.selected_candidate_count;
  usage.max_output_logit_bytes_in_decode = 1024;
  return result;
}

absl::StatusOr<AjimeeCapacityAudit> MakeCapacityAudit(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& config) {
  return BuildAjimeeCapacityAudit(
      corpus, config, MakeCapacityIdentity(),
      [](const converter::CandidateRankerRequest& request) {
        return MakePassingCapacityResult(request);
      });
}

AjimeeSemanticRunnerConfig MakeSemanticConfig() {
  AjimeeSemanticRunnerConfig config;
  config.set_schema_version(kAjimeeSemanticRunnerConfigSchemaVersion);
  config.set_semantic_results_schema_version(
      kAjimeeSemanticResultsSchemaVersion);
  config.set_capacity_audit_config_sha256(kCapacityConfigSha256);
  config.set_capacity_audit_sha256(kCapacityAuditSha256);
  config.set_numeric_profile(AJIMEE_SEMANTIC_CONFIGURED);
  return config;
}

AjimeeSemanticResultsIdentity MakeSemanticIdentity() {
  AjimeeSemanticResultsIdentity identity;
  identity.set_frozen_corpus_sha256(kFrozenSha256);
  identity.set_capacity_audit_config_sha256(kCapacityConfigSha256);
  identity.set_capacity_audit_sha256(kCapacityAuditSha256);
  *identity.mutable_capacity_audit_identity() = MakeCapacityIdentity();
  identity.set_numeric_profile(AJIMEE_SEMANTIC_CONFIGURED);
  return identity;
}

converter::CandidateRankerResponse MakeSuccessfulResponse(
    const converter::CandidateRankerRequest& request) {
  return converter::CandidateRankerResponse{
      .token = request.token,
      .segment_orders = {
          {.segment_id = 1, .candidate_ids = {5, 4}},
          {.segment_id = 0, .candidate_ids = {2, 0}},
      },
  };
}

converter::CandidateRankerResponse MakeOversizedResponse(
    const converter::CandidateRankerRequest& request,
    bool token_matches) {
  converter::CandidateRankerResponse response{
      .token = request.token,
      .segment_orders = {
          {.segment_id = 0, .candidate_ids = {0, 1, 2, 3}},
          {.segment_id = 1, .candidate_ids = {4, 5, 6}},
          {.segment_id = 2, .candidate_ids = {7}},
      },
  };
  if (!token_matches) {
    ++response.token.request_sequence;
  }
  return response;
}

absl::StatusOr<AjimeeSemanticResults> BuildAllSuccessfulResults(
    const AjimeeFrozenCorpus& corpus,
    const AjimeeCapacityAuditConfig& capacity_config,
    const AjimeeCapacityAudit& capacity_audit) {
  return BuildAjimeeSemanticResults(
      corpus, capacity_config, capacity_audit, MakeSemanticConfig(),
      MakeSemanticIdentity(),
      [](const converter::CandidateRankerRequest& request) {
        return MakeSuccessfulResponse(request);
      });
}

TEST(AjimeeSemanticResultsTest,
     BuildsDeterministicPrivacySafeTypedResultsExactlyOnce) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(7);
  const AjimeeCapacityAuditConfig capacity_config = MakeCapacityConfig(7);
  absl::StatusOr<AjimeeCapacityAudit> capacity_audit =
      MakeCapacityAudit(corpus, capacity_config);
  ASSERT_TRUE(capacity_audit.ok()) << capacity_audit.status();
  int callback_count = 0;
  const auto callback =
      [&callback_count](const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<converter::CandidateRankerResponse> {
    ++callback_count;
    switch (request.token.request_sequence) {
      case 1:
        return MakeSuccessfulResponse(request);
      case 2:
        return absl::InvalidArgumentError(kStatusPoison);
      case 3: {
        converter::CandidateRankerResponse response =
            MakeSuccessfulResponse(request);
        ++response.token.request_sequence;
        return response;
      }
      case 4:
        return converter::CandidateRankerResponse{
            .token = request.token,
            .segment_orders = {
                {.segment_id = 0, .candidate_ids = {99}},
            },
        };
      case 5:
        return MakeOversizedResponse(request, false);
      case 6:
        return MakeOversizedResponse(request, true);
      case 7:
        return MakeSuccessfulResponse(request);
    }
    return absl::UnknownError(kStatusPoison);
  };

  absl::StatusOr<AjimeeSemanticResults> first = BuildAjimeeSemanticResults(
      corpus, capacity_config, *capacity_audit, MakeSemanticConfig(),
      MakeSemanticIdentity(), callback);
  absl::StatusOr<AjimeeSemanticResults> second = BuildAjimeeSemanticResults(
      corpus, capacity_config, *capacity_audit, MakeSemanticConfig(),
      MakeSemanticIdentity(), callback);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(callback_count, 14);
  ASSERT_EQ(first->cases_size(), 7);

  const AjimeeSemanticCaseResult& success = first->cases(0);
  EXPECT_EQ(success.outcome(), AJIMEE_SEMANTIC_SUCCESS);
  EXPECT_EQ(success.canonical_status_code(), AJIMEE_STATUS_OK);
  EXPECT_TRUE(success.response_token_matches_request());
  EXPECT_TRUE(success.response_within_request_bounds());
  EXPECT_EQ(success.response_segment_order_count(), 2);
  EXPECT_EQ(success.response_candidate_id_count(), 4);
  EXPECT_EQ(success.merged_top_output(), "winner-zerowinner-one");
  ASSERT_EQ(success.response_segment_orders_size(), 2);
  EXPECT_EQ(success.response_segment_orders(0).segment_id(), 1);
  EXPECT_EQ(success.response_segment_orders(1).segment_id(), 0);

  const AjimeeSemanticCaseResult& backend_error = first->cases(1);
  EXPECT_EQ(backend_error.outcome(), AJIMEE_SEMANTIC_BACKEND_ERROR);
  EXPECT_EQ(backend_error.canonical_status_code(),
            AJIMEE_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(backend_error.has_response_token_matches_request());
  EXPECT_FALSE(backend_error.has_response_within_request_bounds());
  EXPECT_FALSE(backend_error.has_response_segment_order_count());
  EXPECT_FALSE(backend_error.has_response_candidate_id_count());
  EXPECT_EQ(backend_error.response_segment_orders_size(), 0);
  EXPECT_FALSE(backend_error.has_merged_top_output());

  const AjimeeSemanticCaseResult& token_error = first->cases(2);
  EXPECT_EQ(token_error.outcome(), AJIMEE_SEMANTIC_INVALID_RESPONSE);
  EXPECT_EQ(token_error.canonical_status_code(), AJIMEE_STATUS_ABORTED);
  EXPECT_TRUE(token_error.has_response_token_matches_request());
  EXPECT_FALSE(token_error.response_token_matches_request());
  EXPECT_TRUE(token_error.has_response_within_request_bounds());
  EXPECT_TRUE(token_error.response_within_request_bounds());
  EXPECT_TRUE(token_error.has_response_segment_order_count());
  EXPECT_EQ(token_error.response_segment_order_count(), 2);
  EXPECT_TRUE(token_error.has_response_candidate_id_count());
  EXPECT_EQ(token_error.response_candidate_id_count(), 4);
  EXPECT_EQ(token_error.response_segment_orders_size(), 2);
  EXPECT_FALSE(token_error.has_merged_top_output());

  const AjimeeSemanticCaseResult& reference_error = first->cases(3);
  EXPECT_EQ(reference_error.outcome(), AJIMEE_SEMANTIC_INVALID_RESPONSE);
  EXPECT_EQ(reference_error.canonical_status_code(), AJIMEE_STATUS_NOT_FOUND);
  EXPECT_TRUE(reference_error.response_token_matches_request());
  EXPECT_TRUE(reference_error.response_within_request_bounds());
  EXPECT_EQ(reference_error.response_segment_order_count(), 1);
  EXPECT_EQ(reference_error.response_candidate_id_count(), 1);

  const AjimeeSemanticCaseResult& token_oversized = first->cases(4);
  EXPECT_EQ(token_oversized.outcome(), AJIMEE_SEMANTIC_INVALID_RESPONSE);
  EXPECT_EQ(token_oversized.canonical_status_code(), AJIMEE_STATUS_ABORTED);
  EXPECT_TRUE(token_oversized.has_response_token_matches_request());
  EXPECT_FALSE(token_oversized.response_token_matches_request());
  EXPECT_TRUE(token_oversized.has_response_within_request_bounds());
  EXPECT_FALSE(token_oversized.response_within_request_bounds());
  EXPECT_TRUE(token_oversized.has_response_segment_order_count());
  EXPECT_EQ(token_oversized.response_segment_order_count(), 3);
  EXPECT_TRUE(token_oversized.has_response_candidate_id_count());
  EXPECT_EQ(token_oversized.response_candidate_id_count(), 8);
  EXPECT_EQ(token_oversized.response_segment_orders_size(), 0);

  const AjimeeSemanticCaseResult& matching_oversized = first->cases(5);
  EXPECT_EQ(matching_oversized.outcome(), AJIMEE_SEMANTIC_INVALID_RESPONSE);
  EXPECT_EQ(matching_oversized.canonical_status_code(),
            AJIMEE_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(matching_oversized.has_response_token_matches_request());
  EXPECT_TRUE(matching_oversized.response_token_matches_request());
  EXPECT_TRUE(matching_oversized.has_response_within_request_bounds());
  EXPECT_FALSE(matching_oversized.response_within_request_bounds());
  EXPECT_TRUE(matching_oversized.has_response_segment_order_count());
  EXPECT_EQ(matching_oversized.response_segment_order_count(), 3);
  EXPECT_TRUE(matching_oversized.has_response_candidate_id_count());
  EXPECT_EQ(matching_oversized.response_candidate_id_count(), 8);
  EXPECT_EQ(matching_oversized.response_segment_orders_size(), 0);

  const AjimeeSemanticCaseResult& later_success = first->cases(6);
  EXPECT_EQ(later_success.outcome(), AJIMEE_SEMANTIC_SUCCESS);
  EXPECT_EQ(later_success.canonical_status_code(), AJIMEE_STATUS_OK);
  EXPECT_EQ(later_success.merged_top_output(),
            "winner-zerowinner-one");

  absl::StatusOr<std::string> first_binary =
      SerializeDeterministically(*first);
  absl::StatusOr<std::string> second_binary =
      SerializeDeterministically(*second);
  absl::StatusOr<std::string> first_text = ReviewTextproto(*first);
  absl::StatusOr<std::string> second_text = ReviewTextproto(*second);
  ASSERT_TRUE(first_binary.ok());
  ASSERT_TRUE(second_binary.ok());
  ASSERT_TRUE(first_text.ok());
  ASSERT_TRUE(second_text.ok());
  EXPECT_EQ(*first_binary, *second_binary);
  EXPECT_EQ(*first_text, *second_text);
  EXPECT_EQ(first_binary->find(kPrivacyPoison), std::string::npos);
  EXPECT_EQ(first_text->find(kPrivacyPoison), std::string::npos);
  EXPECT_EQ(first_binary->find(kStatusPoison), std::string::npos);
  EXPECT_EQ(first_text->find(kStatusPoison), std::string::npos);
}

TEST(AjimeeSemanticResultsTest, RejectsInconsistentResults) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(2);
  const AjimeeCapacityAuditConfig capacity_config = MakeCapacityConfig(2);
  absl::StatusOr<AjimeeCapacityAudit> capacity_audit =
      MakeCapacityAudit(corpus, capacity_config);
  ASSERT_TRUE(capacity_audit.ok()) << capacity_audit.status();
  absl::StatusOr<AjimeeSemanticResults> valid =
      BuildAllSuccessfulResults(corpus, capacity_config, *capacity_audit);
  ASSERT_TRUE(valid.ok()) << valid.status();

  AjimeeSemanticResults wrong_output = *valid;
  wrong_output.mutable_cases(0)->set_merged_top_output("tampered");
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                wrong_output, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults wrong_backend = *valid;
  wrong_backend.mutable_cases(0)->set_outcome(AJIMEE_SEMANTIC_BACKEND_ERROR);
  wrong_backend.mutable_cases(0)->set_canonical_status_code(
      AJIMEE_STATUS_INTERNAL);
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                wrong_backend, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults false_invalid = *valid;
  false_invalid.mutable_cases(0)->set_outcome(
      AJIMEE_SEMANTIC_INVALID_RESPONSE);
  false_invalid.mutable_cases(0)->set_canonical_status_code(
      AJIMEE_STATUS_NOT_FOUND);
  false_invalid.mutable_cases(0)->clear_merged_top_output();
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                false_invalid, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults unordered = *valid;
  unordered.mutable_cases()->SwapElements(0, 1);
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                unordered, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults oversized = *valid;
  AjimeeSemanticSegmentOrder* extra =
      oversized.mutable_cases(0)->add_response_segment_orders();
  extra->set_segment_id(0);
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                oversized, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults missing_metadata = *valid;
  missing_metadata.mutable_cases(0)->clear_response_candidate_id_count();
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                missing_metadata, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults wrong_count = *valid;
  wrong_count.mutable_cases(0)->set_response_candidate_id_count(3);
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                wrong_count, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults wrong_bounds = *valid;
  AjimeeSemanticCaseResult* wrong_bounds_case =
      wrong_bounds.mutable_cases(0);
  wrong_bounds_case->set_outcome(AJIMEE_SEMANTIC_INVALID_RESPONSE);
  wrong_bounds_case->set_canonical_status_code(
      AJIMEE_STATUS_INVALID_ARGUMENT);
  wrong_bounds_case->set_response_within_request_bounds(false);
  wrong_bounds_case->clear_response_segment_orders();
  wrong_bounds_case->clear_merged_top_output();
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                wrong_bounds, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults raw_out_of_bounds = *valid;
  AjimeeSemanticCaseResult* raw_out_of_bounds_case =
      raw_out_of_bounds.mutable_cases(0);
  raw_out_of_bounds_case->set_outcome(AJIMEE_SEMANTIC_INVALID_RESPONSE);
  raw_out_of_bounds_case->set_canonical_status_code(
      AJIMEE_STATUS_INVALID_ARGUMENT);
  raw_out_of_bounds_case->set_response_within_request_bounds(false);
  raw_out_of_bounds_case->set_response_segment_order_count(3);
  raw_out_of_bounds_case->clear_merged_top_output();
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                raw_out_of_bounds, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeSemanticResults wrong_token_precedence = raw_out_of_bounds;
  AjimeeSemanticCaseResult* wrong_token_case =
      wrong_token_precedence.mutable_cases(0);
  wrong_token_case->clear_response_segment_orders();
  wrong_token_case->set_response_token_matches_request(false);
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                wrong_token_precedence, corpus, capacity_config,
                *capacity_audit, MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeSemanticResultsTest,
     RejectsUnknownFieldsAndInvalidInputsBeforeRanking) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(1);
  const AjimeeCapacityAuditConfig capacity_config = MakeCapacityConfig(1);
  absl::StatusOr<AjimeeCapacityAudit> capacity_audit =
      MakeCapacityAudit(corpus, capacity_config);
  ASSERT_TRUE(capacity_audit.ok()) << capacity_audit.status();

  AjimeeSemanticRunnerConfig poisoned_config = MakeSemanticConfig();
  AddUnknownFieldPoison(&poisoned_config);
  int callback_count = 0;
  const auto callback =
      [&callback_count](const converter::CandidateRankerRequest& request) {
        ++callback_count;
        return MakeSuccessfulResponse(request);
      };
  EXPECT_EQ(BuildAjimeeSemanticResults(
                corpus, capacity_config, *capacity_audit, poisoned_config,
                MakeSemanticIdentity(), callback)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(callback_count, 0);

  AjimeeFrozenCorpus poisoned_corpus = corpus;
  AddUnknownFieldPoison(poisoned_corpus.mutable_cases(0)
                            ->mutable_request()
                            ->mutable_segments(0)
                            ->mutable_candidates(0));
  EXPECT_EQ(BuildAjimeeSemanticResults(
                poisoned_corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig(), MakeSemanticIdentity(), callback)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(callback_count, 0);

  AjimeeSemanticResultsIdentity wrong_identity = MakeSemanticIdentity();
  wrong_identity.set_capacity_audit_sha256(kManifestSha256);
  EXPECT_EQ(BuildAjimeeSemanticResults(
                corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig(), wrong_identity, callback)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(callback_count, 0);

  absl::StatusOr<AjimeeSemanticResults> valid =
      BuildAllSuccessfulResults(corpus, capacity_config, *capacity_audit);
  ASSERT_TRUE(valid.ok()) << valid.status();
  AjimeeSemanticResults poisoned_results = *valid;
  AddUnknownFieldPoison(poisoned_results.mutable_identity()
                            ->mutable_capacity_audit_identity());
  EXPECT_EQ(ValidateAjimeeSemanticResults(
                poisoned_results, corpus, capacity_config, *capacity_audit,
                MakeSemanticConfig())
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeSemanticResultsTest, MapsEveryCanonicalBackendError) {
  constexpr std::array<absl::StatusCode, 16> kStatusCodes = {
      absl::StatusCode::kCancelled,
      absl::StatusCode::kUnknown,
      absl::StatusCode::kInvalidArgument,
      absl::StatusCode::kDeadlineExceeded,
      absl::StatusCode::kNotFound,
      absl::StatusCode::kAlreadyExists,
      absl::StatusCode::kPermissionDenied,
      absl::StatusCode::kResourceExhausted,
      absl::StatusCode::kFailedPrecondition,
      absl::StatusCode::kAborted,
      absl::StatusCode::kOutOfRange,
      absl::StatusCode::kUnimplemented,
      absl::StatusCode::kInternal,
      absl::StatusCode::kUnavailable,
      absl::StatusCode::kDataLoss,
      absl::StatusCode::kUnauthenticated,
  };
  const AjimeeFrozenCorpus corpus = MakeCorpus(kStatusCodes.size());
  const AjimeeCapacityAuditConfig capacity_config =
      MakeCapacityConfig(kStatusCodes.size());
  absl::StatusOr<AjimeeCapacityAudit> capacity_audit =
      MakeCapacityAudit(corpus, capacity_config);
  ASSERT_TRUE(capacity_audit.ok()) << capacity_audit.status();
  absl::StatusOr<AjimeeSemanticResults> results =
      BuildAjimeeSemanticResults(
          corpus, capacity_config, *capacity_audit, MakeSemanticConfig(),
          MakeSemanticIdentity(),
          [&kStatusCodes](const converter::CandidateRankerRequest& request)
              -> absl::StatusOr<converter::CandidateRankerResponse> {
            return absl::Status(
                kStatusCodes[request.token.request_sequence - 1],
                kStatusPoison);
          });
  ASSERT_TRUE(results.ok()) << results.status();
  ASSERT_EQ(results->cases_size(), kStatusCodes.size());
  for (int i = 0; i < results->cases_size(); ++i) {
    EXPECT_EQ(results->cases(i).outcome(),
              AJIMEE_SEMANTIC_BACKEND_ERROR);
    EXPECT_EQ(static_cast<int>(results->cases(i).canonical_status_code()),
              static_cast<int>(kStatusCodes[i]));
  }
}

TEST(AjimeeSemanticResultsTest, CheckedProductionConfigPinsArtifactHashes) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation", "ajimee_semantic_runner_config.textproto"});
  absl::StatusOr<std::string> contents = FileUtil::GetContents(path);
  ASSERT_TRUE(contents.ok()) << contents.status();
  AjimeeSemanticRunnerConfig config;
  ASSERT_TRUE(ParseTextproto(*contents, &config).ok());

  EXPECT_EQ(config.schema_version(), 1);
  EXPECT_EQ(config.semantic_results_schema_version(), 1);
  EXPECT_EQ(config.capacity_audit_config_sha256(),
            "c1265cc3834522a5e9958950227c4b304"
            "21b36690e56be4d822ac66b61f6dc05");
  EXPECT_EQ(config.capacity_audit_sha256(),
            "4854ae79bc8a24e292be356180420215d"
            "d56e70a04c10b12d2f8b242a75e9a5d");
  EXPECT_EQ(config.numeric_profile(), AJIMEE_SEMANTIC_CONFIGURED);
  EXPECT_TRUE(ValidateAjimeeSemanticRunnerConfig(config).ok());
}

}  // namespace
}  // namespace mozc::engine::evaluation
