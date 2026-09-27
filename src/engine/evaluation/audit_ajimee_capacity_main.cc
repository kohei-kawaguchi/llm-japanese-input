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
#include "base/file_util.h"
#include "base/init_mozc.h"
#include "converter/candidate_ranker.h"
#include "engine/evaluation/ajimee_artifact_util.h"
#include "engine/evaluation/ajimee_capacity_audit.h"
#include "engine/evaluation/ajimee_capacity_audit.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/ajimee_frozen_corpus_util.h"
#include "engine/llama_candidate_ranker.h"

#ifdef _WIN32
#include "base/win32/utf8_console_argv.h"
#endif

ABSL_FLAG(std::string, config, "", "Checked AJIMEE capacity-audit textproto");
ABSL_FLAG(std::string, frozen_corpus, "", "Deterministic frozen AJIMEE corpus");
ABSL_FLAG(std::string, manifest, "", "Checked model manifest");
ABSL_FLAG(std::string, model_directory, "", "Directory containing the GGUF");
ABSL_FLAG(std::string, output_binary, "", "Capacity-audit binary output");
ABSL_FLAG(std::string, output_textproto, "",
          "Capacity-audit review textproto output");

namespace mozc::engine::evaluation {
namespace {

class NeverCancelled final : public converter::CandidateRankerCancellation {
 public:
  bool IsCancellationRequested() const override { return false; }
};

absl::Status Run() {
  const std::string config_path = absl::GetFlag(FLAGS_config);
  const std::string frozen_corpus_path = absl::GetFlag(FLAGS_frozen_corpus);
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest);
  const std::string model_directory =
      absl::GetFlag(FLAGS_model_directory);
  const std::string output_binary_path =
      absl::GetFlag(FLAGS_output_binary);
  const std::string output_textproto_path =
      absl::GetFlag(FLAGS_output_textproto);
  if (config_path.empty() || frozen_corpus_path.empty() ||
      manifest_path.empty() || model_directory.empty() ||
      output_binary_path.empty() || output_textproto_path.empty()) {
    return absl::InvalidArgumentError(
        "config, frozen_corpus, manifest, model_directory, output_binary, "
        "and output_textproto are required");
  }

  absl::StatusOr<std::string> config_bytes =
      FileUtil::GetContents(config_path);
  if (!config_bytes.ok()) {
    return config_bytes.status();
  }
  AjimeeCapacityAuditConfig config;
  absl::Status status = ParseTextproto(*config_bytes, &config);
  if (!status.ok()) {
    return status;
  }
  status = ValidateAjimeeCapacityAuditConfig(config);
  if (!status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> frozen_bytes =
      FileUtil::GetContents(frozen_corpus_path);
  if (!frozen_bytes.ok()) {
    return frozen_bytes.status();
  }
  const std::string frozen_sha256 = Sha256Bytes(*frozen_bytes);
  if (frozen_sha256 != config.frozen_corpus_sha256()) {
    return absl::FailedPreconditionError(
        "AJIMEE frozen corpus SHA256 mismatch");
  }
  AjimeeFrozenCorpus corpus;
  if (!corpus.ParseFromString(*frozen_bytes) || !corpus.IsInitialized()) {
    return absl::InvalidArgumentError(
        "AJIMEE frozen corpus binary is invalid");
  }
  status = ValidateAjimeeFrozenCorpusStructure(corpus);
  if (!status.ok()) {
    return status;
  }
  if (corpus.cases_size() != config.expected_case_count()) {
    return absl::FailedPreconditionError(
        "AJIMEE frozen corpus case count mismatch");
  }

  absl::StatusOr<std::string> manifest_bytes =
      FileUtil::GetContents(manifest_path);
  if (!manifest_bytes.ok()) {
    return manifest_bytes.status();
  }
  if (Sha256Bytes(*manifest_bytes) != config.model_manifest_sha256()) {
    return absl::FailedPreconditionError(
        "candidate ranking manifest SHA256 mismatch");
  }
  const LlamaCandidateRankerEvaluationIdentity expected_identity{
      .manifest_sha256 = config.model_manifest_sha256(),
      .gguf_file_name = config.gguf_file_name(),
      .gguf_sha256 = config.gguf_sha256(),
      .tokenizer_sha256 = config.tokenizer_sha256(),
      .quantization = config.quantization(),
      .source_revision = config.source_revision(),
      .runtime_revision = config.runtime_revision(),
  };
  absl::StatusOr<std::unique_ptr<LlamaCandidateRankerCapacityAuditor>>
      created_auditor = LlamaCandidateRankerCapacityAuditor::Create(
          manifest_path, model_directory, expected_identity);
  if (!created_auditor.ok()) {
    return created_auditor.status();
  }
  std::unique_ptr<LlamaCandidateRankerCapacityAuditor> auditor =
      std::move(*created_auditor);

  const LlamaCandidateRankerEvaluationIdentity& verified_identity =
      auditor->EvaluationIdentity();
  AjimeeCapacityAuditIdentity audit_identity;
  audit_identity.set_frozen_corpus_sha256(frozen_sha256);
  audit_identity.set_model_manifest_sha256(
      verified_identity.manifest_sha256);
  audit_identity.set_gguf_file_name(verified_identity.gguf_file_name);
  audit_identity.set_gguf_sha256(verified_identity.gguf_sha256);
  audit_identity.set_tokenizer_sha256(verified_identity.tokenizer_sha256);
  audit_identity.set_quantization(verified_identity.quantization);
  audit_identity.set_source_revision(verified_identity.source_revision);
  audit_identity.set_runtime_revision(verified_identity.runtime_revision);

  NeverCancelled cancellation;
  const auto audit_callback =
      [&auditor, &cancellation](
          const converter::CandidateRankerRequest& request) {
        return auditor->Audit(request, cancellation);
      };
  absl::StatusOr<AjimeeCapacityAudit> audit = BuildAjimeeCapacityAudit(
      corpus, config, audit_identity, audit_callback);
  if (!audit.ok()) {
    return audit.status();
  }
  auditor.reset();

  absl::StatusOr<std::string> binary = SerializeDeterministically(*audit);
  if (!binary.ok()) {
    return binary.status();
  }
  absl::StatusOr<std::string> review_text = ReviewTextproto(*audit);
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
  if (!audit->all_passed()) {
    return absl::FailedPreconditionError(
        "AJIMEE capacity audit did not pass every case");
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
