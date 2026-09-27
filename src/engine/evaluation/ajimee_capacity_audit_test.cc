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
#include <array>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/file_util.h"
#include "converter/candidate_ranker.h"
#include "engine/candidate_ranker_model.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
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
constexpr char kPrivacyPoison[] =
    "PRIVATE_AJIMEE_REQUEST_TEXT_MUST_NEVER_ENTER_THE_AUDIT_ARTIFACT";
constexpr char kStatusPoison[] =
    "PRIVATE_AJIMEE_STATUS_MESSAGE_MUST_NEVER_ENTER_THE_AUDIT_ARTIFACT";
constexpr char kUnknownFieldPoison[] =
    "PRIVATE_AJIMEE_UNKNOWN_FIELD_MUST_NEVER_ENTER_A_VALIDATED_ARTIFACT";

template <typename Message>
void AddUnknownFieldPoison(Message* message) {
  std::string serialized;
  ASSERT_TRUE(message->SerializeToString(&serialized));
  serialized.append("\xA2\x06", 2);
  static_assert(sizeof(kUnknownFieldPoison) - 1 < 128);
  serialized.push_back(
      static_cast<char>(sizeof(kUnknownFieldPoison) - 1));
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
                  const std::string& value) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key("key");
  candidate->set_value(value);
  candidate->set_cost(static_cast<int32_t>(id));
  candidate->set_attributes(static_cast<uint32_t>(id + 1));
  candidate->set_consumed_key_size(static_cast<uint32_t>(id + 2));
  candidate->set_is_protected(false);
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
    uint64_t candidate_id = 0;
    std::string baseline;
    for (uint64_t segment_id = 0; segment_id < 2; ++segment_id) {
      FrozenCandidateRankerSegment* segment = request->add_segments();
      segment->set_id(segment_id);
      segment->set_key("segment-key");
      const std::string first_value =
          segment_id == 0 ? kPrivacyPoison : "baseline-two";
      AddCandidate(segment, candidate_id++, first_value);
      AddCandidate(segment, candidate_id++, "candidate-two");
      AddCandidate(segment, candidate_id++, "candidate-three");
      baseline.append(first_value);
    }
    frozen_case->set_baseline_output(baseline);
  }
  return corpus;
}

AjimeeCapacityAuditConfig MakeConfig(uint32_t case_count) {
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

AjimeeCapacityAuditIdentity MakeIdentity() {
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

void FillCompleteRequestAndWindowUsage(
    const converter::CandidateRankerRequest& request,
    CandidateRankerCapacityUsage* usage) {
  usage->request_segment_count = request.segments.size();
  usage->request_string_bytes = RequestStringBytes(request);
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    usage->request_candidate_count += segment.candidates.size();
    uint64_t unprotected_candidate_count = 0;
    for (const converter::CandidateRankerCandidate& candidate :
         segment.candidates) {
      if (!candidate.is_protected) {
        ++unprotected_candidate_count;
      }
    }
    usage->request_unprotected_candidate_count +=
        unprotected_candidate_count;
    if (unprotected_candidate_count > 1) {
      ++usage->request_rankable_segment_count;
    }
    usage->max_unprotected_candidates_in_segment =
        std::max(usage->max_unprotected_candidates_in_segment,
                 unprotected_candidate_count);
    usage->window_candidate_count += unprotected_candidate_count;
  }
}

CandidateRankerCapacityResult MakePassingResult(
    const converter::CandidateRankerRequest& request,
    CandidateRankerCapacityLimit limiter =
        CandidateRankerCapacityLimit::kFullRecordTokens) {
  CandidateRankerCapacityResult result;
  CandidateRankerCapacityUsage& usage = result.usage;
  FillCompleteRequestAndWindowUsage(request, &usage);
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    ++usage.selected_segment_count;
    usage.selected_candidate_count += 2;
    ++usage.candidates_omitted_by_capacity;
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = segment.id,
        .unprotected_candidate_count = segment.candidates.size(),
        .window_candidate_count = segment.candidates.size(),
        .selected_candidate_count = 2,
        .omitted_by_window = 0,
        .omitted_by_capacity = 1,
        .disposition = CandidateRankerSegmentCapacityDisposition::kSelected,
        .first_limiter = CandidateRankerCapacityViolation{
            .limit = limiter,
            .observed = 4097,
            .allowed = 4096,
        },
    });
  }
  usage.decode_batch_count = 1;
  usage.max_selected_serialized_record_bytes = 80;
  usage.max_selected_normalized_record_bytes = 81;
  usage.max_selected_full_record_tokens = 20;
  usage.max_segments_in_decode = request.segments.size();
  usage.max_sequences_in_decode = usage.selected_candidate_count;
  usage.max_input_nodes_in_decode = 16;
  usage.max_output_rows_in_decode = 4;
  usage.max_reserved_output_rows_in_decode = 8;
  usage.max_output_logit_bytes_in_decode = 1024;
  return result;
}

CandidateRankerCapacityResult MakeOmittedCapacityResult(
    const converter::CandidateRankerRequest& request) {
  CandidateRankerCapacityResult result;
  CandidateRankerCapacityUsage& usage = result.usage;
  FillCompleteRequestAndWindowUsage(request, &usage);
  for (const converter::CandidateRankerSegment& segment : request.segments) {
    const uint64_t candidate_count = segment.candidates.size();
    ++usage.segments_omitted_by_capacity;
    usage.candidates_omitted_by_capacity += candidate_count;
    result.segments.push_back(CandidateRankerSegmentCapacityDiagnostic{
        .segment_id = segment.id,
        .unprotected_candidate_count = candidate_count,
        .window_candidate_count = candidate_count,
        .selected_candidate_count = 0,
        .omitted_by_window = 0,
        .omitted_by_capacity = candidate_count,
        .disposition =
            CandidateRankerSegmentCapacityDisposition::kOmittedCapacity,
        .first_limiter = CandidateRankerCapacityViolation{
            .limit = CandidateRankerCapacityLimit::kSerializedRecordBytes,
            .observed = 81,
            .allowed = 80,
        },
    });
  }
  return result;
}

CandidateRankerCapacityResult MakeRequestLimitResult(
    const converter::CandidateRankerRequest& request,
    CandidateRankerRequestLimit limit) {
  CandidateRankerCapacityResult result;
  result.usage.request_segment_count = request.segments.size();
  uint64_t observed = result.usage.request_segment_count;
  if (limit == CandidateRankerRequestLimit::kCandidates ||
      limit == CandidateRankerRequestLimit::kStringBytes) {
    for (const converter::CandidateRankerSegment& segment : request.segments) {
      result.usage.request_candidate_count += segment.candidates.size();
    }
    observed = result.usage.request_candidate_count;
  }
  if (limit == CandidateRankerRequestLimit::kStringBytes) {
    result.usage.request_string_bytes = RequestStringBytes(request);
    observed = result.usage.request_string_bytes;
  }
  result.request_limit_violation = CandidateRankerRequestLimitViolation{
      .limit = limit,
      .observed = observed,
      .allowed = observed - 1,
  };
  return result;
}

TEST(AjimeeCapacityAuditTest, BuildsDeterministicPrivacySafePassingArtifact) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(2);
  const AjimeeCapacityAuditConfig config = MakeConfig(2);
  const AjimeeCapacityAuditIdentity identity = MakeIdentity();
  int callback_count = 0;
  const auto callback =
      [&callback_count](const converter::CandidateRankerRequest& request) {
        ++callback_count;
        EXPECT_EQ(request.preceding_text, kPrivacyPoison);
        return MakePassingResult(request);
      };
  absl::StatusOr<AjimeeCapacityAudit> first =
      BuildAjimeeCapacityAudit(corpus, config, identity, callback);
  absl::StatusOr<AjimeeCapacityAudit> second =
      BuildAjimeeCapacityAudit(corpus, config, identity, callback);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(callback_count, 4);
  EXPECT_TRUE(first->all_passed());
  ASSERT_EQ(first->cases_size(), 2);
  const AjimeeCapacityAuditUsage& usage = first->cases(0).usage();
  EXPECT_EQ(usage.request_segment_count(), 2);
  EXPECT_EQ(usage.request_candidate_count(), 6);
  EXPECT_EQ(usage.request_unprotected_candidate_count(), 6);
  EXPECT_EQ(usage.request_rankable_segment_count(), 2);
  EXPECT_GT(usage.request_string_bytes(), 0);
  EXPECT_EQ(usage.max_unprotected_candidates_in_segment(), 3);
  EXPECT_EQ(usage.window_candidate_count(), 6);
  EXPECT_EQ(usage.candidates_omitted_by_window(), 0);
  EXPECT_EQ(usage.selected_segment_count(), 2);
  EXPECT_EQ(usage.selected_candidate_count(), 4);
  EXPECT_EQ(usage.candidates_omitted_by_capacity(), 2);
  EXPECT_EQ(usage.segments_omitted_by_capacity(), 0);
  EXPECT_EQ(usage.decode_batch_count(), 1);
  EXPECT_EQ(usage.max_selected_serialized_record_bytes(), 80);
  EXPECT_EQ(usage.max_selected_normalized_record_bytes(), 81);
  EXPECT_EQ(usage.max_selected_full_record_tokens(), 20);
  EXPECT_EQ(usage.max_segments_in_decode(), 2);
  EXPECT_EQ(usage.max_sequences_in_decode(), 4);
  EXPECT_EQ(usage.max_input_nodes_in_decode(), 16);
  EXPECT_EQ(usage.max_output_rows_in_decode(), 4);
  EXPECT_EQ(usage.max_reserved_output_rows_in_decode(), 8);
  EXPECT_EQ(usage.max_output_logit_bytes_in_decode(), 1024);

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
}

TEST(AjimeeCapacityAuditTest, RecordsCapacityOmissionsAsPassingPolicy) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(1);
  absl::StatusOr<AjimeeCapacityAudit> audit = BuildAjimeeCapacityAudit(
      corpus, MakeConfig(1), MakeIdentity(),
      [](const converter::CandidateRankerRequest& request) {
        return MakeOmittedCapacityResult(request);
      });
  ASSERT_TRUE(audit.ok()) << audit.status();
  EXPECT_TRUE(audit->all_passed());
  ASSERT_EQ(audit->cases_size(), 1);
  const AjimeeCapacityAuditCaseResult& result = audit->cases(0);
  EXPECT_EQ(result.outcome(), AJIMEE_CAPACITY_AUDIT_PASS);
  EXPECT_EQ(result.usage().selected_segment_count(), 0);
  EXPECT_EQ(result.usage().selected_candidate_count(), 0);
  EXPECT_EQ(result.usage().segments_omitted_by_capacity(), 2);
  EXPECT_EQ(result.usage().candidates_omitted_by_capacity(), 6);
  EXPECT_EQ(result.usage().decode_batch_count(), 0);
  EXPECT_EQ(result.usage().max_output_logit_bytes_in_decode(), 0);
  ASSERT_EQ(result.segments_size(), 2);
  EXPECT_EQ(result.segments(0).disposition(),
            AJIMEE_SEGMENT_CAPACITY_OMITTED_CAPACITY);
  EXPECT_TRUE(result.segments(0).has_first_limiter());
}

TEST(AjimeeCapacityAuditTest, RecordsAllOutcomesWithoutStatusMessages) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(4);
  const AjimeeCapacityAuditConfig config = MakeConfig(4);
  const AjimeeCapacityAuditIdentity identity = MakeIdentity();
  const auto callback = [](const converter::CandidateRankerRequest& request)
      -> absl::StatusOr<CandidateRankerCapacityResult> {
    switch (request.token.request_sequence) {
      case 1:
        return MakePassingResult(request);
      case 2:
        return MakeRequestLimitResult(
            request, CandidateRankerRequestLimit::kCandidates);
      case 3:
        return absl::InvalidArgumentError(kStatusPoison);
      case 4:
        return absl::InternalError(kStatusPoison);
    }
    return absl::UnknownError(kStatusPoison);
  };
  absl::StatusOr<AjimeeCapacityAudit> audit =
      BuildAjimeeCapacityAudit(corpus, config, identity, callback);
  ASSERT_TRUE(audit.ok()) << audit.status();
  EXPECT_FALSE(audit->all_passed());
  ASSERT_EQ(audit->cases_size(), 4);
  EXPECT_EQ(audit->cases(0).outcome(), AJIMEE_CAPACITY_AUDIT_PASS);
  EXPECT_EQ(static_cast<int>(audit->cases(0).outcome()), 1);
  EXPECT_EQ(audit->cases(0).canonical_status_code(), AJIMEE_STATUS_OK);
  EXPECT_EQ(audit->cases(1).outcome(),
            AJIMEE_CAPACITY_AUDIT_REQUEST_LIMIT_EXCEEDED);
  EXPECT_EQ(static_cast<int>(audit->cases(1).outcome()), 2);
  EXPECT_EQ(audit->cases(1).canonical_status_code(),
            AJIMEE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(audit->cases(2).outcome(),
            AJIMEE_CAPACITY_AUDIT_ERROR);
  EXPECT_EQ(audit->cases(2).canonical_status_code(),
            AJIMEE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(audit->cases(3).outcome(), AJIMEE_CAPACITY_AUDIT_ERROR);
  EXPECT_EQ(static_cast<int>(audit->cases(3).outcome()), 3);
  EXPECT_EQ(audit->cases(3).canonical_status_code(), AJIMEE_STATUS_INTERNAL);
  EXPECT_FALSE(audit->cases(2).has_usage());
  EXPECT_FALSE(audit->cases(3).has_usage());

  absl::StatusOr<std::string> binary = SerializeDeterministically(*audit);
  absl::StatusOr<std::string> text = ReviewTextproto(*audit);
  ASSERT_TRUE(binary.ok());
  ASSERT_TRUE(text.ok());
  EXPECT_EQ(binary->find(kStatusPoison), std::string::npos);
  EXPECT_EQ(text->find(kStatusPoison), std::string::npos);
}

TEST(AjimeeCapacityAuditTest, MapsAllCanonicalErrorStatusCodes) {
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
  constexpr std::array<AjimeeCanonicalStatusCode, 16> kExpectedCodes = {
      AJIMEE_STATUS_CANCELLED,
      AJIMEE_STATUS_UNKNOWN,
      AJIMEE_STATUS_INVALID_ARGUMENT,
      AJIMEE_STATUS_DEADLINE_EXCEEDED,
      AJIMEE_STATUS_NOT_FOUND,
      AJIMEE_STATUS_ALREADY_EXISTS,
      AJIMEE_STATUS_PERMISSION_DENIED,
      AJIMEE_STATUS_RESOURCE_EXHAUSTED,
      AJIMEE_STATUS_FAILED_PRECONDITION,
      AJIMEE_STATUS_ABORTED,
      AJIMEE_STATUS_OUT_OF_RANGE,
      AJIMEE_STATUS_UNIMPLEMENTED,
      AJIMEE_STATUS_INTERNAL,
      AJIMEE_STATUS_UNAVAILABLE,
      AJIMEE_STATUS_DATA_LOSS,
      AJIMEE_STATUS_UNAUTHENTICATED,
  };
  const AjimeeFrozenCorpus corpus =
      MakeCorpus(static_cast<int>(kStatusCodes.size()));
  absl::StatusOr<AjimeeCapacityAudit> audit = BuildAjimeeCapacityAudit(
      corpus, MakeConfig(static_cast<uint32_t>(kStatusCodes.size())),
      MakeIdentity(),
      [&kStatusCodes](const converter::CandidateRankerRequest& request)
          -> absl::StatusOr<CandidateRankerCapacityResult> {
        return absl::Status(
            kStatusCodes[request.token.request_sequence - 1], kStatusPoison);
      });
  ASSERT_TRUE(audit.ok()) << audit.status();
  ASSERT_EQ(audit->cases_size(), kExpectedCodes.size());
  for (int i = 0; i < audit->cases_size(); ++i) {
    EXPECT_EQ(audit->cases(i).outcome(), AJIMEE_CAPACITY_AUDIT_ERROR);
    EXPECT_EQ(audit->cases(i).canonical_status_code(), kExpectedCodes[i]);
  }
}

TEST(AjimeeCapacityAuditTest, MapsAllRequestAndCapacityLimitEnums) {
  const AjimeeFrozenCorpus request_corpus = MakeCorpus(3);
  const AjimeeCapacityAuditConfig request_config = MakeConfig(3);
  constexpr std::array<CandidateRankerRequestLimit, 3> kRequestLimits = {
      CandidateRankerRequestLimit::kSegments,
      CandidateRankerRequestLimit::kCandidates,
      CandidateRankerRequestLimit::kStringBytes,
  };
  const auto request_callback =
      [&kRequestLimits](const converter::CandidateRankerRequest& request) {
        return MakeRequestLimitResult(
            request, kRequestLimits[request.token.request_sequence - 1]);
      };
  absl::StatusOr<AjimeeCapacityAudit> request_audit =
      BuildAjimeeCapacityAudit(request_corpus, request_config, MakeIdentity(),
                               request_callback);
  ASSERT_TRUE(request_audit.ok()) << request_audit.status();
  EXPECT_EQ(request_audit->cases(0).request_limit_violation().limit(),
            AJIMEE_REQUEST_SEGMENTS);
  EXPECT_EQ(request_audit->cases(1).request_limit_violation().limit(),
            AJIMEE_REQUEST_CANDIDATES);
  EXPECT_EQ(request_audit->cases(2).request_limit_violation().limit(),
            AJIMEE_REQUEST_STRING_BYTES);

  const AjimeeFrozenCorpus capacity_corpus = MakeCorpus(9);
  const AjimeeCapacityAuditConfig capacity_config = MakeConfig(9);
  constexpr std::array<CandidateRankerCapacityLimit, 9> kCapacityLimits = {
      CandidateRankerCapacityLimit::kSerializedRecordBytes,
      CandidateRankerCapacityLimit::kNormalizedRecordBytes,
      CandidateRankerCapacityLimit::kFullRecordTokens,
      CandidateRankerCapacityLimit::kSegmentsPerDecode,
      CandidateRankerCapacityLimit::kSequencesPerDecode,
      CandidateRankerCapacityLimit::kInputNodesPerDecode,
      CandidateRankerCapacityLimit::kOutputRowsPerDecode,
      CandidateRankerCapacityLimit::kReservedOutputRowsPerDecode,
      CandidateRankerCapacityLimit::kOutputLogitBytesPerDecode,
  };
  constexpr std::array<AjimeeCapacityLimit, 9> kExpectedCapacityLimits = {
      AJIMEE_SERIALIZED_RECORD_BYTES,
      AJIMEE_NORMALIZED_RECORD_BYTES,
      AJIMEE_FULL_RECORD_TOKENS,
      AJIMEE_SEGMENTS_PER_DECODE,
      AJIMEE_SEQUENCES_PER_DECODE,
      AJIMEE_INPUT_NODES_PER_DECODE,
      AJIMEE_OUTPUT_ROWS_PER_DECODE,
      AJIMEE_RESERVED_OUTPUT_ROWS_PER_DECODE,
      AJIMEE_OUTPUT_LOGIT_BYTES_PER_DECODE,
  };
  const auto capacity_callback =
      [&kCapacityLimits](const converter::CandidateRankerRequest& request) {
        return MakePassingResult(
            request, kCapacityLimits[request.token.request_sequence - 1]);
      };
  absl::StatusOr<AjimeeCapacityAudit> capacity_audit =
      BuildAjimeeCapacityAudit(capacity_corpus, capacity_config,
                               MakeIdentity(), capacity_callback);
  ASSERT_TRUE(capacity_audit.ok()) << capacity_audit.status();
  for (int i = 0; i < capacity_audit->cases_size(); ++i) {
    EXPECT_EQ(capacity_audit->cases(i).segments(0).first_limiter().limit(),
              kExpectedCapacityLimits[i]);
  }
}

TEST(AjimeeCapacityAuditTest, RejectsUnknownFieldsRecursively) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(1);
  const AjimeeCapacityAuditConfig config = MakeConfig(1);

  AjimeeCapacityAuditConfig unknown_config = config;
  AddUnknownFieldPoison(&unknown_config);
  EXPECT_EQ(ValidateAjimeeCapacityAuditConfig(unknown_config).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeCapacityAuditIdentity unknown_identity = MakeIdentity();
  AddUnknownFieldPoison(&unknown_identity);
  int callback_count = 0;
  absl::StatusOr<AjimeeCapacityAudit> rejected = BuildAjimeeCapacityAudit(
      corpus, config, unknown_identity,
      [&callback_count](const converter::CandidateRankerRequest& request) {
        ++callback_count;
        return MakePassingResult(request);
      });
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(callback_count, 0);

  absl::StatusOr<AjimeeCapacityAudit> valid = BuildAjimeeCapacityAudit(
      corpus, config, MakeIdentity(),
      [](const converter::CandidateRankerRequest& request) {
        return MakePassingResult(request);
      });
  ASSERT_TRUE(valid.ok()) << valid.status();
  AjimeeCapacityAudit unknown_nested_artifact = *valid;
  AddUnknownFieldPoison(
      unknown_nested_artifact.mutable_cases(0)->mutable_segments(0));
  absl::StatusOr<std::string> poisoned_binary =
      SerializeDeterministically(unknown_nested_artifact);
  ASSERT_TRUE(poisoned_binary.ok());
  EXPECT_NE(poisoned_binary->find(kUnknownFieldPoison), std::string::npos);
  EXPECT_EQ(ValidateAjimeeCapacityAudit(unknown_nested_artifact, corpus, config)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeCapacityAuditTest, RejectsSchemaIdentityAndSemanticMismatches) {
  const AjimeeFrozenCorpus corpus = MakeCorpus(1);
  const AjimeeCapacityAuditConfig config = MakeConfig(1);
  EXPECT_TRUE(ValidateAjimeeCapacityAuditConfig(config).ok());
  for (int schema_field = 0; schema_field < 4; ++schema_field) {
    AjimeeCapacityAuditConfig invalid = config;
    switch (schema_field) {
      case 0:
        invalid.set_schema_version(2);
        break;
      case 1:
        invalid.set_frozen_corpus_schema_version(1);
        break;
      case 2:
        invalid.set_capacity_audit_schema_version(2);
        break;
      case 3:
        invalid.set_model_manifest_schema_version(3);
        break;
    }
    EXPECT_EQ(ValidateAjimeeCapacityAuditConfig(invalid).code(),
              absl::StatusCode::kInvalidArgument);
  }
  const std::array<std::string, 6> invalid_names = {
      ".", "..", "directory/model.gguf", "directory\\model.gguf",
      "C:model.gguf", std::string("model\0.gguf", 11),
  };
  for (const std::string& invalid_name : invalid_names) {
    AjimeeCapacityAuditConfig invalid = config;
    invalid.set_gguf_file_name(invalid_name);
    EXPECT_EQ(ValidateAjimeeCapacityAuditConfig(invalid).code(),
              absl::StatusCode::kInvalidArgument);
  }

  AjimeeFrozenCorpus malformed_corpus = corpus;
  malformed_corpus.mutable_cases(0)->mutable_request()->clear_segments();
  int malformed_callback_count = 0;
  absl::StatusOr<AjimeeCapacityAudit> malformed = BuildAjimeeCapacityAudit(
      malformed_corpus, config, MakeIdentity(),
      [&malformed_callback_count](
          const converter::CandidateRankerRequest& request) {
        ++malformed_callback_count;
        return MakePassingResult(request);
      });
  EXPECT_EQ(malformed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(malformed_callback_count, 0);

  int callback_count = 0;
  AjimeeCapacityAuditIdentity mismatched_identity = MakeIdentity();
  mismatched_identity.set_gguf_sha256(kManifestSha256);
  absl::StatusOr<AjimeeCapacityAudit> mismatch = BuildAjimeeCapacityAudit(
      corpus, config, mismatched_identity,
      [&callback_count](const converter::CandidateRankerRequest& request) {
        ++callback_count;
        return MakePassingResult(request);
      });
  EXPECT_EQ(mismatch.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(callback_count, 0);

  absl::StatusOr<AjimeeCapacityAudit> valid = BuildAjimeeCapacityAudit(
      corpus, config, MakeIdentity(),
      [](const converter::CandidateRankerRequest& request) {
        return MakePassingResult(request);
      });
  ASSERT_TRUE(valid.ok()) << valid.status();

  AjimeeCapacityAudit wrong_source = *valid;
  wrong_source.mutable_cases(0)->set_source_index(999);
  EXPECT_EQ(ValidateAjimeeCapacityAudit(wrong_source, corpus, config).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeCapacityAudit wrong_usage = *valid;
  wrong_usage.mutable_cases(0)->mutable_usage()->set_selected_candidate_count(
      3);
  EXPECT_EQ(ValidateAjimeeCapacityAudit(wrong_usage, corpus, config).code(),
            absl::StatusCode::kInvalidArgument);

  AjimeeCapacityAudit contradictory_limiter = *valid;
  AjimeeCapacityViolation* limiter = contradictory_limiter.mutable_cases(0)
                                         ->mutable_segments(0)
                                         ->mutable_first_limiter();
  limiter->set_observed(21);
  limiter->set_allowed(19);
  EXPECT_EQ(
      ValidateAjimeeCapacityAudit(contradictory_limiter, corpus, config).code(),
      absl::StatusCode::kInvalidArgument);

  AjimeeCapacityAudit reordered_segments = *valid;
  reordered_segments.mutable_cases(0)->mutable_segments()->SwapElements(0, 1);
  EXPECT_EQ(
      ValidateAjimeeCapacityAudit(reordered_segments, corpus, config).code(),
      absl::StatusCode::kInvalidArgument);

  AjimeeCapacityAudit wrong_all_passed = *valid;
  wrong_all_passed.set_all_passed(false);
  EXPECT_EQ(
      ValidateAjimeeCapacityAudit(wrong_all_passed, corpus, config).code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(AjimeeCapacityAuditTest, CheckedProductionConfigPinsEveryIdentity) {
  const std::string path = testing::GetSourceFileOrDie(
      {"engine", "evaluation", "ajimee_capacity_audit_config.textproto"});
  absl::StatusOr<std::string> contents = FileUtil::GetContents(path);
  ASSERT_TRUE(contents.ok()) << contents.status();
  AjimeeCapacityAuditConfig config;
  ASSERT_TRUE(ParseTextproto(*contents, &config).ok());

  EXPECT_EQ(config.schema_version(), 1);
  EXPECT_EQ(config.frozen_corpus_schema_version(), 2);
  EXPECT_EQ(config.capacity_audit_schema_version(), 1);
  EXPECT_EQ(config.frozen_corpus_sha256(),
            "9b1656a48ed3cfbf1577b7a404688aff"
            "b49b793f9c556ba2d2fb46b41238c61c");
  EXPECT_EQ(config.expected_case_count(), 200);
  EXPECT_EQ(config.model_manifest_schema_version(), 4);
  EXPECT_EQ(config.model_manifest_sha256(),
            "f4d7e53eb14110c855fe684f34ee536"
            "465f7eae6601f9555859ccce28edfb465");
  EXPECT_EQ(config.gguf_file_name(), "rinna-xsmall-q4_k_m.gguf");
  EXPECT_EQ(config.gguf_sha256(),
            "3c53d019a0ff0832ff5f7b08508e8b5"
            "c3b78b747871060deb2db3375fcfb6677");
  EXPECT_EQ(config.tokenizer_sha256(),
            "b5cbdfa8aa7c54c8c5af85b78c309c5"
            "4a5f2749a20468bf6f60eee007fe6fec1");
  EXPECT_EQ(config.quantization(), "Q4_K_M");
  EXPECT_EQ(config.source_revision(),
            "8e91527b3276e0565154935e84a08bf0137ed99f");
  EXPECT_EQ(config.runtime_revision(),
            "1511ce3bc3f087376c8526b4ad07100bfabb277f");
  EXPECT_TRUE(ValidateAjimeeCapacityAuditConfig(config).ok());
}

}  // namespace
}  // namespace mozc::engine::evaluation
