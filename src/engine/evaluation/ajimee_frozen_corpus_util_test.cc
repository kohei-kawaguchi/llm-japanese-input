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

#include "engine/evaluation/ajimee_frozen_corpus_util.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "engine/evaluation/ajimee_frozen_corpus.pb.h"
#include "engine/evaluation/evaluation_artifact.pb.h"
#include "engine/evaluation/frozen_candidate_ranker.pb.h"
#include "testing/gunit.h"

namespace mozc::engine::evaluation {
namespace {

constexpr char kHex64[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kRevision40[] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

void FillSourceIdentity(EvaluationSourceIdentity* source) {
  source->set_benchmark_name("AJIMEE-Bench synthetic");
  source->set_source_revision(kRevision40);
  source->set_source_relative_path("synthetic/evaluation_items.json");
  source->set_source_sha256(kHex64);
  source->set_creator("Synthetic test creator");
  source->set_source_url("https://example.test/source");
  source->set_license_identifier("CC-BY-SA-3.0");
  source->set_license_url("https://creativecommons.org/licenses/by-sa/3.0/");
  source->set_upstream_dataset_url("https://example.test/upstream");
  source->set_changes_notice("Synthetic transformed corpus.");
}

FrozenCandidateRankerCandidate* AddCandidate(
    FrozenCandidateRankerSegment* segment, uint64_t id, const std::string& key,
    const std::string& value, int32_t cost, uint32_t attributes,
    uint32_t consumed_key_size, bool is_protected) {
  FrozenCandidateRankerCandidate* candidate = segment->add_candidates();
  candidate->set_id(id);
  candidate->set_key(key);
  candidate->set_value(value);
  candidate->set_cost(cost);
  candidate->set_attributes(attributes);
  candidate->set_consumed_key_size(consumed_key_size);
  candidate->set_is_protected(is_protected);
  return candidate;
}

AjimeeFrozenCorpus MakeValidCorpus() {
  AjimeeFrozenCorpus corpus;
  corpus.set_schema_version(kAjimeeFrozenCorpusSchemaVersion);
  FillSourceIdentity(corpus.mutable_source());
  corpus.set_input_corpus_sha256(kHex64);
  EvaluationMozcIdentity* mozc = corpus.mutable_mozc();
  mozc->set_source_revision(kRevision40);
  mozc->set_data_type("oss");
  mozc->set_data_sha256(kHex64);
  mozc->set_default_desktop_request_sha256(kHex64);
  mozc->set_default_desktop_config_sha256(kHex64);
  mozc->set_evaluation_clock_utc_rfc3339("2000-01-01T00:00:00Z");

  AjimeeFrozenCase* frozen_case = corpus.add_cases();
  frozen_case->set_source_index(17);
  frozen_case->set_normalized_hiragana_reading("よみ");
  frozen_case->set_history_reconstructed(false);
  frozen_case->set_baseline_output("甲乙");
  FrozenCandidateRankerRequest* request = frozen_case->mutable_request();
  request->mutable_token()->set_session_generation(0);
  request->mutable_token()->set_state_revision(0);
  request->mutable_token()->set_request_sequence(1);
  request->set_mode(FrozenCandidateRankerRequest::MODE_CONVERSION);
  request->set_preceding_text("左文脈");
  request->set_following_text("");
  request->set_reading("よみ");
  request->set_focused_segment_id(0);

  FrozenCandidateRankerSegment* first = request->add_segments();
  first->set_id(0);
  first->set_key("よ");
  AddCandidate(first, 0, "よ", "甲", 1, 2, 3, false);
  AddCandidate(first, 1, "よ", "丙", 4, 5, 6, true);
  FrozenCandidateRankerSegment* second = request->add_segments();
  second->set_id(1);
  second->set_key("み");
  AddCandidate(second, 2, "み", "乙", 7, 8, 9, false);
  return corpus;
}

TEST(AjimeeFrozenCorpusUtilTest, ValidatesFrozenCorpusWithoutFreezerConfig) {
  const AjimeeFrozenCorpus valid = MakeValidCorpus();
  EXPECT_TRUE(ValidateAjimeeFrozenCorpusStructure(valid).ok());

  const auto expect_invalid = [](const AjimeeFrozenCorpus& corpus) {
    EXPECT_EQ(ValidateAjimeeFrozenCorpusStructure(corpus).code(),
              absl::StatusCode::kInvalidArgument);
  };

  AjimeeFrozenCorpus schema = valid;
  schema.set_schema_version(kAjimeeFrozenCorpusSchemaVersion + 1);
  expect_invalid(schema);

  AjimeeFrozenCorpus source = valid;
  source.mutable_source()->set_source_sha256("A");
  expect_invalid(source);

  AjimeeFrozenCorpus input_hash = valid;
  input_hash.set_input_corpus_sha256("A");
  expect_invalid(input_hash);

  AjimeeFrozenCorpus clock = valid;
  clock.mutable_mozc()->set_evaluation_clock_utc_rfc3339(
      "2000-01-01T00:00:00+00:00");
  expect_invalid(clock);

  AjimeeFrozenCorpus unordered = valid;
  AjimeeFrozenCase* duplicate_case = unordered.add_cases();
  *duplicate_case = unordered.cases(0);
  duplicate_case->mutable_request()->mutable_token()->set_request_sequence(2);
  expect_invalid(unordered);

  AjimeeFrozenCorpus reading = valid;
  reading.mutable_cases(0)->mutable_request()->set_reading("べつ");
  expect_invalid(reading);

  AjimeeFrozenCorpus segment_id = valid;
  segment_id.mutable_cases(0)->mutable_request()->mutable_segments(0)->set_id(
      1);
  expect_invalid(segment_id);

  AjimeeFrozenCorpus candidate_id = valid;
  candidate_id.mutable_cases(0)
      ->mutable_request()
      ->mutable_segments(0)
      ->mutable_candidates(1)
      ->set_id(2);
  expect_invalid(candidate_id);

  AjimeeFrozenCorpus baseline = valid;
  baseline.mutable_cases(0)->set_baseline_output("不一致");
  expect_invalid(baseline);
}

}  // namespace
}  // namespace mozc::engine::evaluation
