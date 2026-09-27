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

#ifndef MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_IMPORTER_H_
#define MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_IMPORTER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "engine/evaluation/quality_regression_corpus.pb.h"
#include "engine/evaluation/quality_regression_corpus_util.h"

namespace mozc::engine::evaluation {

struct QualityRegressionCorpora {
  QualityRegressionInputCorpus development_inputs;
  QualityRegressionAnswerCorpus development_answers;
  QualityRegressionInputCorpus holdout_inputs;
  QualityRegressionAnswerCorpus holdout_answers;
};

absl::Status ValidateQualityRegressionImporterConfig(
    const QualityRegressionImporterConfig& config);

absl::StatusOr<QualityRegressionCorpora> ImportQualityRegressionCorpora(
    absl::string_view development_source,
    absl::string_view holdout_source,
    const QualityRegressionImporterConfig& config,
    absl::string_view import_config_sha256);

}  // namespace mozc::engine::evaluation

#endif  // MOZC_ENGINE_EVALUATION_QUALITY_REGRESSION_IMPORTER_H_
