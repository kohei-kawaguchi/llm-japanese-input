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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "base/file/temp_dir.h"
#include "base/file_util.h"
#include "base/protobuf/text_format.h"
#include "engine/candidate_ranker_model.pb.h"
#include "gguf.h"
#include "testing/gunit.h"
#include "testing/mozctest.h"

extern "C" {
#include "sha256/sha256.h"
}

namespace mozc {
namespace engine {
namespace {

constexpr char kSha256[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr char kInvalidGguf[] = "not a gguf";
constexpr char kInvalidGgufSha256[] =
    "490392a613ac8386f8f21b2f785ad5989694b9acecdf27931df67f99cd157235";

struct TestGgufDeleter {
  void operator()(gguf_context* context) const { gguf_free(context); }
};

using TestGgufPointer =
    std::unique_ptr<gguf_context, TestGgufDeleter>;

enum class ArchitectureMetadataType : uint8_t {
  kString,
  kUint32,
};

enum class ContextMetadataType : uint8_t {
  kUint32,
  kString,
};

std::string MakeMetadataGguf(
    ArchitectureMetadataType architecture_type,
    ContextMetadataType context_type, uint32_t context_value) {
  const TestGgufPointer metadata(gguf_init_empty());
  if (architecture_type == ArchitectureMetadataType::kString) {
    gguf_set_val_str(metadata.get(), "general.architecture", "gpt2");
  } else {
    gguf_set_val_u32(metadata.get(), "general.architecture", 1);
  }
  if (context_type == ContextMetadataType::kUint32) {
    gguf_set_val_u32(metadata.get(), "gpt2.context_length", context_value);
  } else {
    gguf_set_val_str(metadata.get(), "gpt2.context_length", "1024");
  }
  std::string bytes(gguf_get_meta_size(metadata.get()), '\0');
  gguf_get_meta_data(metadata.get(), bytes.data());
  return bytes;
}

std::string ComputeTestSha256(const std::string& contents) {
  sha256_t hash;
  sha256_init(&hash);
  sha256_update(
      &hash, reinterpret_cast<const unsigned char*>(contents.data()),
      contents.size());
  std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
  sha256_final(&hash, digest.data());
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result(digest.size() * 2, '\0');
  for (size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = kHexDigits[digest[i] >> 4];
    result[2 * i + 1] = kHexDigits[digest[i] & 0x0f];
  }
  return result;
}

CandidateRankerModelManifest MakeManifest() {
  CandidateRankerModelManifest manifest;
  manifest.set_schema_version(5);

  CandidateRankerModelManifest::ScoringTemplate* scoring_template =
      manifest.mutable_scoring_template();
  scoring_template->set_version("test-v2");
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
  runtime->set_batch_thread_count(2);
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

std::array<LlamaCandidateRankerEvaluationIdentity, 7>
MakeIdentityMismatches(
    const LlamaCandidateRankerEvaluationIdentity& exact_identity) {
  std::array<LlamaCandidateRankerEvaluationIdentity, 7> mismatches = {
      exact_identity, exact_identity, exact_identity, exact_identity,
      exact_identity, exact_identity, exact_identity,
  };
  mismatches[0].manifest_sha256 = "mismatch";
  mismatches[1].gguf_file_name = "mismatch.gguf";
  mismatches[2].gguf_sha256 = "mismatch";
  mismatches[3].tokenizer_sha256 = "mismatch";
  mismatches[4].quantization = "mismatch";
  mismatches[5].source_revision = "mismatch";
  mismatches[6].runtime_revision = "mismatch";
  return mismatches;
}

class LlamaCandidateRankerTest : public ::testing::Test {
 protected:
  LlamaCandidateRankerTest()
      : directory_(mozc::testing::MakeTempDirectoryOrDie()),
        manifest_path_(
            FileUtil::JoinPath(directory_.path(), "manifest.textproto")),
        model_path_(FileUtil::JoinPath(directory_.path(), "model.gguf")) {}

  LlamaCandidateRankerEvaluationIdentity WriteManifest(
      const CandidateRankerModelManifest& manifest) {
    std::string text;
    EXPECT_TRUE(protobuf::TextFormat::PrintToString(manifest, &text));
    EXPECT_TRUE(FileUtil::SetContents(manifest_path_, text).ok());
    return BuildLlamaCandidateRankerEvaluationIdentity(manifest, text);
  }

  LlamaCandidateRankerEvaluationIdentity WriteModelAndManifest(
      CandidateRankerModelManifest manifest, const std::string& model) {
    EXPECT_TRUE(FileUtil::SetContents(model_path_, model).ok());
    manifest.mutable_artifacts()->set_gguf_sha256(
        ComputeTestSha256(model));
    return WriteManifest(manifest);
  }

  TempDirectory directory_;
  const std::string manifest_path_;
  const std::string model_path_;
};

TEST_F(LlamaCandidateRankerTest, ReportsMissingManifestWithoutAPath) {
  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(result.status().message(),
            "candidate ranking manifest is unavailable");
}

TEST_F(LlamaCandidateRankerTest,
       CapacityAuditorUsesTheCheckedManifestAndModelPath) {
  const auto missing_manifest = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), {});
  EXPECT_EQ(missing_manifest.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(missing_manifest.status().message(),
            "candidate ranking manifest is unavailable");

  const LlamaCandidateRankerEvaluationIdentity exact_identity =
      WriteManifest(MakeManifest());
  const auto mismatches = MakeIdentityMismatches(exact_identity);
  for (const LlamaCandidateRankerEvaluationIdentity& mismatch : mismatches) {
    const auto identity_mismatch =
        LlamaCandidateRankerCapacityAuditor::Create(
            manifest_path_, directory_.path(), mismatch);
    EXPECT_EQ(identity_mismatch.status().code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(identity_mismatch.status().message(),
              "candidate ranking evaluation identity does not match the "
              "expected identity");
  }

  const auto missing_model = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), exact_identity);
  EXPECT_EQ(missing_model.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(missing_model.status().message(),
            "candidate ranking model is unavailable");

  ASSERT_TRUE(FileUtil::SetContents(model_path_, kInvalidGguf).ok());
  const auto hash_mismatch = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), exact_identity);
  EXPECT_EQ(hash_mismatch.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(hash_mismatch.status().message(),
            "candidate ranking model hash does not match the manifest");

  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_artifacts()->set_gguf_sha256(kInvalidGgufSha256);
  const LlamaCandidateRankerEvaluationIdentity invalid_gguf_identity =
      WriteManifest(manifest);
  const auto invalid_gguf = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), invalid_gguf_identity);
  EXPECT_EQ(invalid_gguf.status().code(), absl::StatusCode::kInternal);
}

TEST_F(LlamaCandidateRankerTest,
       EvaluationCreateChecksIdentityBeforeOpeningModel) {
  const LlamaCandidateRankerEvaluationIdentity exact_identity =
      WriteManifest(MakeManifest());
  for (const LlamaCandidateRankerEvaluationIdentity& mismatch :
       MakeIdentityMismatches(exact_identity)) {
    const auto result = LlamaCandidateRanker::Create(
        manifest_path_, directory_.path(), mismatch);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(result.status().message(),
              "candidate ranking evaluation identity does not match the "
              "expected identity");
  }

  const auto exact = LlamaCandidateRanker::Create(
      manifest_path_, directory_.path(), exact_identity);
  EXPECT_EQ(exact.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(exact.status().message(),
            "candidate ranking model is unavailable");
}

TEST_F(LlamaCandidateRankerTest,
       CapacityAuditorRejectsWrongTypedContextBeforeModelLoading) {
  const std::string model = MakeMetadataGguf(
      ArchitectureMetadataType::kString, ContextMetadataType::kString, 1024);
  const LlamaCandidateRankerEvaluationIdentity identity =
      WriteModelAndManifest(MakeManifest(), model);

  const auto result = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), identity);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(),
            "candidate ranking model context metadata is invalid");
}

TEST_F(LlamaCandidateRankerTest,
       CapacityAuditorRejectsWrongTypedArchitectureBeforeModelLoading) {
  const std::string model = MakeMetadataGguf(
      ArchitectureMetadataType::kUint32, ContextMetadataType::kUint32, 1024);
  const LlamaCandidateRankerEvaluationIdentity identity =
      WriteModelAndManifest(MakeManifest(), model);

  const auto result = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), identity);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(),
            "candidate ranking model architecture metadata is invalid");
}

TEST_F(LlamaCandidateRankerTest,
       CapacityAuditorRejectsContextOutsideTheInt32Range) {
  for (const uint32_t context :
       {0u, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1}) {
    const std::string model = MakeMetadataGguf(
        ArchitectureMetadataType::kString, ContextMetadataType::kUint32,
        context);
    const LlamaCandidateRankerEvaluationIdentity identity =
        WriteModelAndManifest(MakeManifest(), model);

    const auto result = LlamaCandidateRankerCapacityAuditor::Create(
        manifest_path_, directory_.path(), identity);

    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(result.status().message(),
              "candidate ranking model context metadata is invalid");
  }
}

TEST_F(LlamaCandidateRankerTest,
       CapacityAuditorRewindsAfterValidTypedMetadata) {
  const std::string model = MakeMetadataGguf(
      ArchitectureMetadataType::kString, ContextMetadataType::kUint32, 1024);
  const LlamaCandidateRankerEvaluationIdentity identity =
      WriteModelAndManifest(MakeManifest(), model);

  const auto result = LlamaCandidateRankerCapacityAuditor::Create(
      manifest_path_, directory_.path(), identity);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(result.status().message(),
            "candidate ranking model could not be loaded");
}

TEST_F(LlamaCandidateRankerTest, RejectsMalformedManifest) {
  ASSERT_TRUE(FileUtil::SetContents(manifest_path_, "not a manifest").ok());
  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlamaCandidateRankerTest, RejectsSchemaVersionFourManifest) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.set_schema_version(4);
  WriteManifest(manifest);

  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlamaCandidateRankerTest, RejectsUnsafeModelBasenameBeforeLookup) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_artifacts()->set_gguf_file_name("../model.gguf");
  WriteManifest(manifest);

  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlamaCandidateRankerTest, ReportsMissingModelWithoutAPath) {
  WriteManifest(MakeManifest());

  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(result.status().message(),
            "candidate ranking model is unavailable");
}

TEST_F(LlamaCandidateRankerTest, RejectsModelHashMismatchBeforeLoading) {
  WriteManifest(MakeManifest());
  ASSERT_TRUE(FileUtil::SetContents(model_path_, kInvalidGguf).ok());

  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDataLoss);
}

TEST_F(LlamaCandidateRankerTest, RejectsInvalidGgufAfterHashVerification) {
  CandidateRankerModelManifest manifest = MakeManifest();
  manifest.mutable_artifacts()->set_gguf_sha256(kInvalidGgufSha256);
  WriteManifest(manifest);
  ASSERT_TRUE(FileUtil::SetContents(model_path_, kInvalidGguf).ok());

  const absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> result =
      LlamaCandidateRanker::Create(manifest_path_, directory_.path());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
}

TEST(LlamaCandidateRankerEvaluationTest,
     BuildsIdentityFromExactManifestBytes) {
  const LlamaCandidateRankerEvaluationIdentity identity =
      BuildLlamaCandidateRankerEvaluationIdentity(MakeManifest(),
                                                  "manifest bytes");

  EXPECT_EQ(identity.manifest_sha256,
            "66444334cfc47d81840f88c1e690564d3c130e9237482a58d7631124e9b9a815");
  EXPECT_EQ(identity.gguf_file_name, "model.gguf");
  EXPECT_EQ(identity.gguf_sha256, kSha256);
  EXPECT_EQ(identity.tokenizer_sha256, kSha256);
  EXPECT_EQ(identity.quantization, "Q4_K_M");
  EXPECT_EQ(identity.source_revision,
            "0123456789abcdef0123456789abcdef01234567");
  EXPECT_EQ(identity.runtime_revision,
            "fedcba9876543210fedcba9876543210fedcba98");
}

TEST(LlamaCandidateRankerEvaluationTest,
     ValidatesTheFedRecordTokenBoundaryAgainstModelContext) {
  CandidateRankerModelManifest manifest = MakeManifest();
  EXPECT_TRUE(
      ValidateLlamaCandidateRankerModelContextCapacity(manifest, 63).ok());
  EXPECT_EQ(ValidateLlamaCandidateRankerModelContextCapacity(manifest, 62)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateLlamaCandidateRankerModelContextCapacity(manifest, 0)
                .code(),
            absl::StatusCode::kInvalidArgument);
  manifest.mutable_limits()->clear_per_record_token_limit();
  EXPECT_EQ(ValidateLlamaCandidateRankerModelContextCapacity(manifest, 63)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(LlamaCandidateRankerEvaluationTest, WritesReportProvenance) {
  std::ostringstream output;
  WriteLlamaCandidateRankerEvaluationIdentity(
      output, LlamaCandidateRankerEvaluationIdentity{
                  .manifest_sha256 = "manifest",
                  .gguf_file_name = "model.gguf",
                  .gguf_sha256 = "gguf",
                  .tokenizer_sha256 = "tokenizer",
                  .quantization = "Q4_K_M",
                  .source_revision = "source",
                  .runtime_revision = "runtime",
              });
  WriteLlamaCandidateRankerEvaluationNumericProfile(
      output, LlamaCandidateRankerEvaluationProfile::kConfigured);
  WriteLlamaCandidateRankerEvaluationNumericProfile(
      output, LlamaCandidateRankerEvaluationProfile::kF32KvFlashDisabled);

  EXPECT_EQ(output.str(),
            "identity\tmanifest_sha256\tmanifest\tgguf_file_name\tmodel.gguf"
            "\tgguf_sha256\tgguf\ttokenizer_sha256\ttokenizer"
            "\tquantization\tQ4_K_M\tsource_revision\tsource"
            "\truntime_revision\truntime\n"
            "numeric_profile\tconfigured\tkv_key\tf16\tkv_value\tf16"
            "\tflash_attention\tauto\n"
            "numeric_profile\tf32_kv_flash_disabled\tkv_key\tf32\tkv_value"
            "\tf32\tflash_attention\tdisabled\n");
}

TEST(LlamaCandidateRankerEvaluationTest, ClassifiesScoredEdgeContributions) {
  EXPECT_EQ(ClassifyLlamaCandidateRankerEdgeContribution(4, false, 5),
            LlamaCandidateRankerEdgeContribution::kCandidateOrBoundary);
  EXPECT_EQ(ClassifyLlamaCandidateRankerEdgeContribution(5, false, 5),
            LlamaCandidateRankerEdgeContribution::kFollowingContext);
  EXPECT_EQ(ClassifyLlamaCandidateRankerEdgeContribution(4, true, 5),
            LlamaCandidateRankerEdgeContribution::kTerminal);
}

}  // namespace
}  // namespace engine
}  // namespace mozc
