#!/usr/bin/env python3

# Copyright 2026 LLM Japanese Input Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#     * Neither the name of LLM Japanese Input Authors nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from google.protobuf import text_format

from engine import candidate_ranker_model_pb2
from engine.evaluation import frozen_candidate_ranker_pb2
from engine.evaluation import quality_regression_corpus_pb2
from engine.evaluation import quality_regression_development_report
from engine.evaluation import quality_regression_development_report_pb2
from engine.evaluation import quality_regression_frozen_corpus_pb2
from engine.evaluation import quality_regression_objective_evaluation_pb2
from engine.evaluation import report_quality_regression_development_main


_HEX_40 = "a" * 40
_HEX_64_A = "1" * 64
_HEX_64_B = "2" * 64
_HEX_64_C = "3" * 64
_HEX_64_D = "4" * 64
_HEX_64_E = "5" * 64
_STRUCTURED_WITH_TARGET = (
    candidate_ranker_model_pb2.CandidateRankerModelManifest.ScoringTemplate
    .STRUCTURED_WITH_TARGET
)
_STRUCTURED_WITHOUT_TARGET = (
    candidate_ranker_model_pb2.CandidateRankerModelManifest.ScoringTemplate
    .STRUCTURED_WITHOUT_TARGET
)
_NATURAL_TEXT_ONLY = (
    candidate_ranker_model_pb2.CandidateRankerModelManifest.ScoringTemplate
    .NATURAL_TEXT_ONLY
)


def _binary(message) -> bytes:
  return message.SerializeToString(deterministic=True)


def _textproto(message) -> bytes:
  return text_format.MessageToString(
      message, as_utf8=True, use_index_order=True
  ).encode("utf-8")


def _copy_source(destination) -> None:
  destination.benchmark_name = "Synthetic quality regression"
  destination.source_revision = _HEX_40
  destination.source_relative_path = "synthetic/development.tsv"
  destination.source_sha256 = _HEX_64_A
  destination.creator = "Synthetic creator"
  destination.source_url = "https://example.test/source"
  destination.license_identifier = "BSD-3-Clause"
  destination.license_url = "https://example.test/license"
  destination.upstream_dataset_url = "https://example.test/upstream"
  destination.changes_notice = "Synthetic reporter test."


def _copy_identity(destination) -> None:
  destination.role = quality_regression_corpus_pb2.DEVELOPMENT
  _copy_source(destination.source)
  destination.parser_definition_version = 1
  destination.normalization_definition_version = 1
  destination.import_config_sha256 = _HEX_64_B


def _make_corpora():
  frozen = quality_regression_frozen_corpus_pb2.QualityRegressionFrozenCorpus()
  frozen.schema_version = 1
  _copy_identity(frozen.identity)
  frozen.input_corpus_sha256 = _HEX_64_C
  frozen.mozc.source_revision = _HEX_40
  frozen.mozc.data_type = "oss"
  frozen.mozc.data_sha256 = _HEX_64_A
  frozen.mozc.default_desktop_request_sha256 = _HEX_64_B
  frozen.mozc.default_desktop_config_sha256 = _HEX_64_C
  frozen.mozc.evaluation_clock_utc_rfc3339 = "2000-01-01T00:00:00Z"
  frozen.freezer_config_sha256 = _HEX_64_D

  answers = quality_regression_corpus_pb2.QualityRegressionAnswerCorpus()
  answers.schema_version = 1
  _copy_identity(answers.identity)
  for ordinal in range(5):
    frozen_case = frozen.cases.add()
    frozen_case.source_line = ordinal + 1
    frozen_case.conversion_reading = "よみ"
    frozen_case.mozc_baseline_output = "誤"
    request = frozen_case.request
    request.token.session_generation = 0
    request.token.state_revision = 0
    request.token.request_sequence = ordinal + 1
    request.mode = (
        frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest.MODE_CONVERSION
    )
    request.preceding_text = "文脈"
    request.following_text = ""
    request.reading = "よみ"
    request.focused_segment_id = 0
    segment = request.segments.add()
    segment.id = 0
    segment.key = "キー"
    for candidate_id, value in enumerate(("誤", "キー", "正")):
      candidate = segment.candidates.add()
      candidate.id = candidate_id
      candidate.key = "よみ"
      candidate.value = value
      candidate.cost = candidate_id
      candidate.attributes = 0
      candidate.consumed_key_size = 0
      candidate.is_protected = False

    answer = answers.cases.add()
    answer.source_line = ordinal + 1
    answer.normalized_whole_output = "正"
  return frozen, answers


def _add_candidate_score(
    segment, candidate_id: int, score_bits: int, token_count: int = 1
) -> None:
  candidate = segment.candidates.add()
  candidate.candidate_id = candidate_id
  candidate.original_candidate_index = candidate_id
  candidate.mozc_cost = candidate_id
  candidate.full_continuation_score_bits = score_bits
  candidate.scored_continuation_token_count = token_count


def _make_scores(
    frozen, model_ordinal: int
) -> quality_regression_objective_evaluation_pb2.DevelopmentObjectiveScores:
  scores = quality_regression_objective_evaluation_pb2.DevelopmentObjectiveScores()
  scores.schema_version = 2
  scores.identity.score_config_sha256 = hashlib.sha256(
      f"score-config-{model_ordinal}".encode()
  ).hexdigest()
  scores.identity.execution_config_sha256 = hashlib.sha256(
      f"execution-config-{model_ordinal}".encode()
  ).hexdigest()
  scores.identity.frozen_corpus_sha256 = hashlib.sha256(_binary(frozen)).hexdigest()
  scores.identity.native_coverage_suite_sha256 = hashlib.sha256(
      f"coverage-{model_ordinal}".encode()
  ).hexdigest()
  scores.identity.score_definition_version = 2
  scores.identity.numeric_profile = (
      quality_regression_objective_evaluation_pb2
      .DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED
  )
  scores.identity.layout = (
      quality_regression_objective_evaluation_pb2
      .DEVELOPMENT_EVALUATION_SHARED_TRIE
  )
  objectives = (
      (_STRUCTURED_WITH_TARGET, ((1, 300.0), (2, 99.0), (0, 0.0))),
      (_STRUCTURED_WITHOUT_TARGET, ((2, 100.0), (1, 50.0), (0, 0.0))),
      (_NATURAL_TEXT_ONLY, ((2, 100.0), (1, 50.0), (0, 0.0))),
  )
  for objective, candidate_scores in objectives:
    objective_scores = scores.objectives.add()
    objective_scores.objective = objective
    for frozen_case in frozen.cases:
      case_scores = objective_scores.cases.add()
      case_scores.source_line = frozen_case.source_line
      segment_scores = case_scores.segments.add()
      segment_scores.segment_id = 0
      for candidate_id, score in candidate_scores:
        _add_candidate_score(
            segment_scores,
            candidate_id,
            int.from_bytes(struct.pack("<d", score), "little"),
        )
  return scores


def _make_config(
    frozen_binary: bytes,
    answer_binary: bytes,
    scores_binaries: tuple[bytes, ...],
    scores: tuple[
        quality_regression_objective_evaluation_pb2.DevelopmentObjectiveScores,
        ...,
    ],
):
  config = (
      quality_regression_development_report_pb2
      .QualityRegressionDevelopmentReportConfig()
  )
  config.schema_version = 2
  config.frozen_corpus_schema_version = 1
  config.answer_corpus_schema_version = 1
  config.score_artifact_schema_version = 2
  config.report_schema_version = 2
  _copy_identity(config.expected_corpus_identity)
  config.frozen_corpus_sha256 = hashlib.sha256(frozen_binary).hexdigest()
  config.answer_corpus_sha256 = hashlib.sha256(answer_binary).hexdigest()
  config.expected_case_count = 5
  config.expected_baseline_correct_count = 0
  config.minimum_net_gain_cases = 1
  config.maximum_baseline_correct_regressions = 0
  config.metric_definition_version = 2
  config.exact_bootstrap_definition_version = 1
  config.clean_key_copy_definition_version = 1
  config.selection_policy_version = 2
  for model_ordinal, (scores_binary, model_scores, parameter_count) in enumerate(
      zip(
          scores_binaries,
          scores,
          (37_000_000, 494_000_000),
          strict=True,
      )
  ):
    model = config.models.add()
    model.ordinal = model_ordinal
    model.source_model = f"synthetic/model-{model_ordinal}"
    model.source_revision = f"{model_ordinal + 1:x}" * 40
    model.parameter_count = parameter_count
    model.objective_scores_sha256 = hashlib.sha256(scores_binary).hexdigest()
    model.score_identity.CopyFrom(model_scores.identity)
    for objective, eligible in (
        (_STRUCTURED_WITH_TARGET, False),
        (_STRUCTURED_WITHOUT_TARGET, True),
        (_NATURAL_TEXT_ONLY, True),
    ):
      spec = model.objectives.add()
      spec.objective = objective
      spec.manifest_filename = (
          f"manifest-{model_ordinal}-{objective}.textproto"
      )
      spec.manifest_sha256 = hashlib.sha256(
          spec.manifest_filename.encode()
      ).hexdigest()
      spec.selection_eligible = eligible
  ordinal = 0
  for aggregation in (
      quality_regression_development_report_pb2
      .DEVELOPMENT_FULL_CONTINUATION_SUM,
      quality_regression_development_report_pb2.DEVELOPMENT_CONTINUATION_MEAN,
  ):
    for order_prior in (0, 250000, 500000, 1000000):
      for exact_key_penalty in (0, 100000000):
        calibration = config.calibrations.add()
        calibration.ordinal = ordinal
        calibration.aggregation = aggregation
        calibration.mozc_order_prior_microlog_probability = order_prior
        calibration.exact_key_copy_penalty_microlog_probability = (
            exact_key_penalty
        )
        ordinal += 1
  return config


class QualityRegressionDevelopmentReportTest(unittest.TestCase):

  def test_calibration_grid_applies_mean_prior_and_key_penalty(self):
    frozen, _ = _make_corpora()
    segment = frozen.cases[0].request.segments[0]

    def ranked_ids(
        scores,
        aggregation,
        order_prior=0,
        exact_key_penalty=0,
    ):
      segment_scores = (
          quality_regression_objective_evaluation_pb2.DevelopmentSegmentScores()
      )
      segment_scores.segment_id = segment.id
      for candidate_id, score, token_count in scores:
        _add_candidate_score(
            segment_scores,
            candidate_id,
            int.from_bytes(struct.pack("<d", score), "little"),
            token_count,
        )
      calibration = (
          quality_regression_development_report_pb2.DevelopmentCalibrationSpec()
      )
      calibration.ordinal = 0
      calibration.aggregation = aggregation
      calibration.mozc_order_prior_microlog_probability = order_prior
      calibration.exact_key_copy_penalty_microlog_probability = (
          exact_key_penalty
      )
      return quality_regression_development_report._calibrated_ranked_ids(
          segment, segment_scores, calibration
      )

    normalization_scores = ((0, -6.0, 3), (1, -100.0, 1), (2, -3.0, 1))
    self.assertEqual(
        ranked_ids(
            normalization_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_FULL_CONTINUATION_SUM,
        )[0],
        2,
    )
    self.assertEqual(
        ranked_ids(
            normalization_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_CONTINUATION_MEAN,
        )[0],
        0,
    )

    prior_scores = ((0, 0.0, 1), (1, -100.0, 1), (2, 0.4, 1))
    self.assertEqual(
        ranked_ids(
            prior_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_FULL_CONTINUATION_SUM,
        )[0],
        2,
    )
    self.assertEqual(
        ranked_ids(
            prior_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_FULL_CONTINUATION_SUM,
            order_prior=250_000,
        )[0],
        0,
    )

    penalty_scores = ((0, 0.0, 1), (1, 2.0, 1), (2, 1.0, 1))
    self.assertEqual(
        ranked_ids(
            penalty_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_FULL_CONTINUATION_SUM,
        )[0],
        1,
    )
    self.assertEqual(
        ranked_ids(
            penalty_scores,
            quality_regression_development_report_pb2
            .DEVELOPMENT_FULL_CONTINUATION_SUM,
            exact_key_penalty=100_000_000,
        )[0],
        2,
    )

  def test_builds_selection_report_and_serializes_deterministically(self):
    frozen, answers = _make_corpora()
    scores = tuple(_make_scores(frozen, index) for index in range(2))
    frozen_binary = _binary(frozen)
    answer_binary = _binary(answers)
    scores_binaries = tuple(_binary(model_scores) for model_scores in scores)
    config = _make_config(
        frozen_binary, answer_binary, scores_binaries, scores
    )
    inputs = (
        quality_regression_development_report
        .QualityRegressionDevelopmentReportInputs(
            report_config_textproto=_textproto(config),
            frozen_corpus_binary=frozen_binary,
            answer_corpus_binary=answer_binary,
            objective_scores_binaries=scores_binaries,
        )
    )

    report = quality_regression_development_report.build_development_report(inputs)
    self.assertTrue(report.selected)
    self.assertEqual(report.selected_model_ordinal, 0)
    self.assertEqual(report.selected_objective, _STRUCTURED_WITHOUT_TARGET)
    self.assertEqual(
        report.selected_manifest_sha256,
        config.models[0].objectives[1].manifest_sha256,
    )
    self.assertEqual(report.selected_calibration.ordinal, 0)
    self.assertFalse(report.objectives[0].selection_eligible)
    self.assertFalse(report.objectives[0].development_gate_passed)
    self.assertTrue(report.objectives[16].development_gate_passed)
    self.assertTrue(report.objectives[32].development_gate_passed)
    self.assertEqual(report.objectives[16].win_count, 5)
    self.assertEqual(report.objectives[16].clean_key_copy_removed_vs_reference, 5)
    self.assertEqual(report.objectives[16].clean_key_copy_added_vs_reference, 0)
    self.assertEqual(report.objectives[48].model_ordinal, 1)
    self.assertEqual(len(report.objectives), 96)

    first = quality_regression_development_report.serialize_development_report(report)
    second = quality_regression_development_report.serialize_development_report(
        quality_regression_development_report.build_development_report(inputs)
    )
    self.assertEqual(first.binary, second.binary)
    self.assertEqual(first.canonical_json, second.canonical_json)
    self.assertNotIn("よみ".encode("utf-8"), first.binary)
    self.assertNotIn("正".encode("utf-8"), first.canonical_json)

  def test_cli_writes_same_artifacts(self):
    frozen, answers = _make_corpora()
    scores = tuple(_make_scores(frozen, index) for index in range(2))
    frozen_binary = _binary(frozen)
    answer_binary = _binary(answers)
    scores_binaries = tuple(_binary(model_scores) for model_scores in scores)
    config = _make_config(
        frozen_binary, answer_binary, scores_binaries, scores
    )
    inputs = (
        quality_regression_development_report
        .QualityRegressionDevelopmentReportInputs(
            report_config_textproto=_textproto(config),
            frozen_corpus_binary=frozen_binary,
            answer_corpus_binary=answer_binary,
            objective_scores_binaries=scores_binaries,
        )
    )
    expected = quality_regression_development_report.serialize_development_report(
        quality_regression_development_report.build_development_report(inputs)
    )

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      config_path = root / "config.textproto"
      frozen_path = root / "frozen.pb"
      answer_path = root / "answer.pb"
      scores_paths = tuple(root / f"scores-{index}.pb" for index in range(2))
      binary_path = root / "report.pb"
      json_path = root / "report.json"
      config_path.write_bytes(_textproto(config))
      frozen_path.write_bytes(frozen_binary)
      answer_path.write_bytes(answer_binary)
      for path, data in zip(scores_paths, scores_binaries, strict=True):
        path.write_bytes(data)
      report_quality_regression_development_main.main(
          [
              f"--config={config_path}",
              f"--frozen_corpus={frozen_path}",
              f"--answer_corpus={answer_path}",
              *(f"--objective_scores={path}" for path in scores_paths),
              f"--output_binary={binary_path}",
              f"--output_json={json_path}",
          ]
      )
      self.assertEqual(binary_path.read_bytes(), expected.binary)
      self.assertEqual(json_path.read_bytes(), expected.canonical_json)

  def test_checked_config_parses(self):
    config_path = Path(__file__).with_name(
        "quality_regression_development_report_config.textproto"
    )
    config = (
        quality_regression_development_report_pb2
        .QualityRegressionDevelopmentReportConfig()
    )
    text_format.Parse(config_path.read_text(encoding="utf-8"), config)
    self.assertEqual(config.expected_case_count, 219)
    self.assertEqual(config.expected_baseline_correct_count, 164)
    self.assertEqual(len(config.models), 2)
    self.assertEqual(len(config.calibrations), 16)


if __name__ == "__main__":
  unittest.main()
