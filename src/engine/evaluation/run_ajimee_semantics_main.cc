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

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/evaluation/ajimee_semantic_results.h"
#include "engine/evaluation/ajimee_semantic_results.pb.h"
#include "engine/llama_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "", "Checked AJIMEE semantic-runner textproto");
ABSL_FLAG(std::string, capacity_audit_config, "",
          "Checked AJIMEE capacity-audit textproto");
ABSL_FLAG(std::string, frozen_corpus, "", "Deterministic frozen AJIMEE corpus");
ABSL_FLAG(std::string, capacity_audit, "", "Accepted AJIMEE capacity audit");
ABSL_FLAG(std::string, manifest, "", "Checked model manifest");
ABSL_FLAG(std::string, model_directory, "", "Directory containing the GGUF");
ABSL_FLAG(std::string, output_binary, "", "Semantic-results binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Semantic-results review textproto output");

namespace mozc::engine::evaluation {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

AjimeeCapacityAuditIdentity MakeAuditIdentity(
    absl::string_view frozen_corpus_sha256,
    const LlamaCandidateRankerEvaluationIdentity& backend_identity) {
  AjimeeCapacityAuditIdentity identity;
  identity.set_frozen_corpus_sha256(frozen_corpus_sha256);
  identity.set_model_manifest_sha256(backend_identity.manifest_sha256);
  identity.set_gguf_file_name(backend_identity.gguf_file_name);
  identity.set_gguf_sha256(backend_identity.gguf_sha256);
  identity.set_tokenizer_sha256(backend_identity.tokenizer_sha256);
  identity.set_quantization(backend_identity.quantization);
  identity.set_source_revision(backend_identity.source_revision);
  identity.set_runtime_revision(backend_identity.runtime_revision);
  return identity;
}

LlamaCandidateRankerEvaluationIdentity MakeExpectedBackendIdentity(
    const AjimeeCapacityAuditConfig& config) {
  return LlamaCandidateRankerEvaluationIdentity{
      .manifest_sha256 = config.model_manifest_sha256(),
      .gguf_file_name = config.gguf_file_name(),
      .gguf_sha256 = config.gguf_sha256(),
      .tokenizer_sha256 = config.tokenizer_sha256(),
      .quantization = config.quantization(),
      .source_revision = config.source_revision(),
      .runtime_revision = config.runtime_revision(),
  };
}

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string capacity_config_path =
      absl::GetFlag(FLAGS_capacity_audit_config);
  const std::string frozen_corpus_path = absl::GetFlag(FLAGS_frozen_corpus);
  const std::string capacity_audit_path = absl::GetFlag(FLAGS_capacity_audit);
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest);
  const std::string model_directory = absl::GetFlag(FLAGS_model_directory);
  const std::string output_binary_path = absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);
  if (config_path.empty() || capacity_config_path.empty() ||
      frozen_corpus_path.empty() || capacity_audit_path.empty() ||
      manifest_path.empty() || model_directory.empty() ||
      output_binary_path.empty() || output_textproto_path.empty()) {
    return absl::InvalidArgumentError(
        "config, capacity_audit_config, frozen_corpus, capacity_audit, "
        "manifest, model_directory, output_binary, and output_textproto are "
        "required");
  }

  absl::StatusOr<std::string> config_bytes = FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  AjimeeSemanticRunnerConfig config;
  absl::Status status = ParseTextproto(*config_bytes, &config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeSemanticRunnerConfig(config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> capacity_config_bytes =
      FileUtil::GetContents(capacity_config_path);
  if (!capacity_config_bytes.ok()) {
    return capacity_config_bytes.status();
  }
  if (Sha256Bytes(*capacity_config_bytes) !=
      config.capacity_audit_config_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity-audit config SHA256 mismatch");
  }
  AjimeeCapacityAuditConfig capacity_config;
  status = ParseTextproto(*capacity_config_bytes, &capacity_config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeCapacityAuditConfig(capacity_config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> frozen_bytes =
      FileUtil::GetContents(frozen_corpus_path);
  if (!frozen_bytes.ok()) {
    return frozen_bytes.status();
  }
  const std::string frozen_sha256 = Sha256Bytes(*frozen_bytes);
  if (frozen_sha256 != capacity_config.frozen_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE frozen corpus SHA256 mismatch");
  }
  AjimeeFrozenCorpus corpus;
  if (!corpus.ParseFromString(*frozen_bytes) || !corpus.IsInitialized()) {
    return absl::InvalidArgumentError("AJIMEE frozen corpus binary is invalid");
  }
  if (HasUnknownFieldsRecursively(corpus)) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus contains unknown fields");
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  if (corpus.cases_size() != capacity_config.expected_case_count()) {
    return absl::FailedPreconditionError(
        "AJIMEE frozen corpus case count mismatch");
  }

  absl::StatusOr<std::string> capacity_bytes =
      FileUtil::GetContents(capacity_audit_path);
  if (!capacity_bytes.ok()) {
    return capacity_bytes.status();
  }
  if (Sha256Bytes(*capacity_bytes) != config.capacity_audit_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity-audit SHA256 mismatch");
  }
  AjimeeCapacityAudit capacity_audit;
  if (!capacity_audit.ParseFromString(*capacity_bytes) ||
      !capacity_audit.IsInitialized()) {
    return absl::InvalidArgumentError(
        "AJIMEE capacity-audit binary is invalid");
  }
  status = ValidateAjimeeCapacityAudit(capacity_audit, corpus, capacity_config);
  if (!status.ok()) {
    return status;
  }
  if (!capacity_audit.all_passed()) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity audit did not pass every case");
  }

  absl::StatusOr<std::string> manifest_bytes =
      FileUtil::GetContents(manifest_path);
  if (!manifest_bytes.ok()) {
    return manifest_bytes.status();
  }
  if (Sha256Bytes(*manifest_bytes) != capacity_config.model_manifest_sha256()) {
    return absl::FailedPreconditionError(
        "candidate ranking manifest SHA256 mismatch");
  }

  const LlamaCandidateRankerEvaluationIdentity expected_identity =
      MakeExpectedBackendIdentity(capacity_config);
  absl::StatusOr<std::shared_ptr<LlamaCandidateRanker>> created_backend =
      LlamaCandidateRanker::Create(manifest_path, model_directory,
                                   expected_identity);
  if (!created_backend.ok()) {
    return created_backend.status();
  }
  std::shared_ptr<LlamaCandidateRanker> backend = std::move(*created_backend);
  const LlamaCandidateRankerEvaluationIdentity& verified_identity =
      backend->EvaluationIdentity();
  if (verified_identity.manifest_sha256 != expected_identity.manifest_sha256 ||
      verified_identity.gguf_file_name != expected_identity.gguf_file_name ||
      verified_identity.gguf_sha256 != expected_identity.gguf_sha256 ||
      verified_identity.tokenizer_sha256 !=
          expected_identity.tokenizer_sha256 ||
      verified_identity.quantization != expected_identity.quantization ||
      verified_identity.source_revision != expected_identity.source_revision ||
      verified_identity.runtime_revision !=
          expected_identity.runtime_revision) {
    return absl::FailedPreconditionError(
        "candidate ranking backend identity changed after creation");
  }

  AjimeeSemanticResultsIdentity semantic_identity;
  semantic_identity.set_frozen_corpus_sha256(frozen_sha256);
  semantic_identity.set_capacity_audit_config_sha256(
      config.capacity_audit_config_sha256());
  semantic_identity.set_capacity_audit_sha256(config.capacity_audit_sha256());
  *semantic_identity.mutable_capacity_audit_identity() =
      MakeAuditIdentity(frozen_sha256, verified_identity);
  semantic_identity.set_numeric_profile(config.numeric_profile());

  NeverCancelled cancellation;
  const auto rank_callback =
      [&backend,
       &cancellation](const converter::CandidateRankerRequest& request) {
        return backend->Rank(request, cancellation);
      };
  absl::StatusOr<AjimeeSemanticResults> results =
      BuildAjimeeSemanticResults(corpus, capacity_config, capacity_audit,
                                 config, semantic_identity, rank_callback);
  if (!results.ok()) {
    return results.status();
  }
  backend.reset();

  absl::StatusOr<std::string> binary = SerializeDeterministically(*results);
  if (!binary.ok()) {
    return binary.status();
  }
  absl::StatusOr<std::string> review_text = ReviewTextproto(*results);
  if (!review_text.ok()) {
    return review_text.status();
  }
  status = FileUtil::SetContents(output_binary_path, *binary);
  if (!status.ok()) {
    return status;
  }
  status = FileUtil::SetContents(output_textproto_path, *review_text);
  if (!status.ok()) {
    return status;
  }
  for (const AjimeeSemanticCaseResult& result : results->cases()) {
    if (result.outcome() != AJIMEE_SEMANTIC_SUCCESS) {
      return absl::FailedPreconditionError(
          "AJIMEE semantic run did not succeed for every case");
    }
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace mozc::engine::evaluation

namespace {

int MainUtf8(int argc, char** argv) {
  mozc::InitMozc(argv[0], &argc, &argv);
  const absl::Status status = mozc::engine::evaluation::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return static_cast<int>(status.code());
  }
  return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  mozc::win32::Utf8ConsoleArgv utf8_argv(argc, argv);
  return MainUtf8(utf8_argv.argc(), utf8_argv.argv());
}
#else
int main(int argc, char** argv) { return MainUtf8(argc, argv); }
#endif
