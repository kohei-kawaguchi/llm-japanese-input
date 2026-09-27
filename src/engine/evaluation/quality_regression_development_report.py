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

"""Builds deterministic quality-regression development selection reports."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import struct
from typing import Any, Iterable

from google.protobuf import text_format
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from engine import candidate_ranker_model_pb2
from engine.evaluation import exact_quality_metrics_pb2
from engine.evaluation import frozen_candidate_ranker_pb2
from engine.evaluation import quality_regression_corpus_pb2
from engine.evaluation import quality_regression_development_report_pb2
from engine.evaluation import quality_regression_frozen_corpus_pb2
from engine.evaluation import quality_regression_objective_evaluation_pb2


_CONFIG_SCHEMA_VERSION = 2
_FROZEN_CORPUS_SCHEMA_VERSION = 1
_ANSWER_CORPUS_SCHEMA_VERSION = 1
_SCORE_ARTIFACT_SCHEMA_VERSION = 2
_REPORT_SCHEMA_VERSION = 2
_SCORE_DEFINITION_VERSION = 2
_METRIC_DEFINITION_VERSION = 2
_EXACT_BOOTSTRAP_DEFINITION_VERSION = 1
_CLEAN_KEY_COPY_DEFINITION_VERSION = 1
_SELECTION_POLICY_VERSION = 2
_EXPECTED_MODEL_COUNT = 2
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")

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
_OBJECTIVE_ORDER = (
    _STRUCTURED_WITH_TARGET,
    _STRUCTURED_WITHOUT_TARGET,
    _NATURAL_TEXT_ONLY,
)
_CONFIGURED_NUMERIC_PROFILE = (
    quality_regression_objective_evaluation_pb2
    .DEVELOPMENT_NUMERIC_PROFILE_CONFIGURED
)
_SHARED_TRIE_LAYOUT = (
    quality_regression_objective_evaluation_pb2
    .DEVELOPMENT_EVALUATION_SHARED_TRIE
)
_FULL_CONTINUATION_SUM = (
    quality_regression_development_report_pb2.DEVELOPMENT_FULL_CONTINUATION_SUM
)
_CONTINUATION_MEAN = (
    quality_regression_development_report_pb2.DEVELOPMENT_CONTINUATION_MEAN
)
_ORDER_PRIORS = (0, 250_000, 500_000, 1_000_000)
_EXACT_KEY_PENALTIES = (0, 100_000_000)
_EXPECTED_CALIBRATIONS = tuple(
    (ordinal, aggregation, order_prior, exact_key_penalty)
    for ordinal, (aggregation, order_prior, exact_key_penalty) in enumerate(
        (
            (aggregation, order_prior, exact_key_penalty)
            for aggregation in (_FULL_CONTINUATION_SUM, _CONTINUATION_MEAN)
            for order_prior in _ORDER_PRIORS
            for exact_key_penalty in _EXACT_KEY_PENALTIES
        )
    )
)


@dataclasses.dataclass(frozen=True)
class QualityRegressionDevelopmentReportInputs:
  report_config_textproto: bytes
  frozen_corpus_binary: bytes
  answer_corpus_binary: bytes
  objective_scores_binaries: tuple[bytes, ...]


@dataclasses.dataclass(frozen=True)
class QualityRegressionSerializedDevelopmentReport:
  binary: bytes
  canonical_json: bytes


@dataclasses.dataclass(frozen=True)
class _CaseOutput:
  source_line: int
  baseline_output: str
  model_output: str
  answer: str
  exact_key_regression: bool
  clean_key_copy_promotions: frozenset[int]


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _require_initialized_without_unknown_fields(
    message: Message, label: str
) -> None:
  if not message.IsInitialized():
    raise ValueError(f"{label} is incomplete")
  serialized = message.SerializePartialToString(deterministic=True)
  known = message.__class__()
  known.CopyFrom(message)
  known.DiscardUnknownFields()
  if known.SerializePartialToString(deterministic=True) != serialized:
    raise ValueError(f"{label} contains unknown fields")


def _parse_textproto(message_type: type[Message], data: bytes, label: str) -> Message:
  message = message_type()
  text_format.Parse(data.decode("utf-8"), message)
  _require_initialized_without_unknown_fields(message, label)
  return message


def _parse_binary(message_type: type[Message], data: bytes, label: str) -> Message:
  message = message_type()
  message.ParseFromString(data)
  _require_initialized_without_unknown_fields(message, label)
  return message


def _require_lower_hex64(value: str, label: str) -> None:
  if _LOWER_HEX_64.fullmatch(value) is None:
    raise ValueError(f"{label} is not a lowercase SHA256")


def _require_lower_hex40(value: str, label: str) -> None:
  if _LOWER_HEX_40.fullmatch(value) is None:
    raise ValueError(f"{label} is not a lowercase 40-digit revision")


def _require_same_message(actual: Message, expected: Message, label: str) -> None:
  if (
      actual.SerializeToString(deterministic=True)
      != expected.SerializeToString(deterministic=True)
  ):
    raise ValueError(label)


def _validate_config(
    config: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReportConfig,
) -> None:
  if (
      config.schema_version != _CONFIG_SCHEMA_VERSION
      or config.frozen_corpus_schema_version != _FROZEN_CORPUS_SCHEMA_VERSION
      or config.answer_corpus_schema_version != _ANSWER_CORPUS_SCHEMA_VERSION
      or config.score_artifact_schema_version != _SCORE_ARTIFACT_SCHEMA_VERSION
      or config.report_schema_version != _REPORT_SCHEMA_VERSION
      or config.metric_definition_version != _METRIC_DEFINITION_VERSION
      or config.exact_bootstrap_definition_version
      != _EXACT_BOOTSTRAP_DEFINITION_VERSION
      or config.clean_key_copy_definition_version
      != _CLEAN_KEY_COPY_DEFINITION_VERSION
      or config.selection_policy_version != _SELECTION_POLICY_VERSION
  ):
    raise ValueError("quality-regression development report config schema is invalid")
  if (
      config.expected_corpus_identity.role
      != quality_regression_corpus_pb2.DEVELOPMENT
      or config.expected_case_count == 0
      or config.expected_baseline_correct_count > config.expected_case_count
      or len(config.models) != _EXPECTED_MODEL_COUNT
      or len(config.calibrations) != len(_EXPECTED_CALIBRATIONS)
  ):
    raise ValueError("quality-regression development report config counts are invalid")
  for name in ("frozen_corpus_sha256", "answer_corpus_sha256"):
    _require_lower_hex64(getattr(config, name), f"config {name}")
  seen_model_identities: set[tuple[str, str]] = set()
  seen_score_hashes: set[str] = set()
  seen_manifest_hashes: set[str] = set()
  for model_index, model in enumerate(config.models):
    model_identity = (model.source_model, model.source_revision)
    if (
        model.ordinal != model_index
        or not model.source_model
        or model.parameter_count == 0
        or model_identity in seen_model_identities
        or model.objective_scores_sha256 in seen_score_hashes
        or len(model.objectives) != len(_OBJECTIVE_ORDER)
    ):
      raise ValueError("quality-regression development model config is invalid")
    _require_lower_hex40(model.source_revision, "model source revision")
    _require_lower_hex64(
        model.objective_scores_sha256, "model objective scores SHA256"
    )
    score_identity = model.score_identity
    for name in (
        "score_config_sha256",
        "execution_config_sha256",
        "native_coverage_suite_sha256",
    ):
      _require_lower_hex64(getattr(score_identity, name), f"model {name}")
    if (
        score_identity.frozen_corpus_sha256 != config.frozen_corpus_sha256
        or score_identity.score_definition_version != _SCORE_DEFINITION_VERSION
        or score_identity.numeric_profile != _CONFIGURED_NUMERIC_PROFILE
        or score_identity.layout != _SHARED_TRIE_LAYOUT
    ):
      raise ValueError("quality-regression development score config is invalid")
    for objective_index, objective in enumerate(model.objectives):
      if (
          objective.objective != _OBJECTIVE_ORDER[objective_index]
          or objective.selection_eligible
          != (objective.objective != _STRUCTURED_WITH_TARGET)
          or not objective.manifest_filename
          or objective.manifest_sha256 in seen_manifest_hashes
      ):
        raise ValueError("quality-regression development objective config is invalid")
      _require_lower_hex64(objective.manifest_sha256, "objective manifest SHA256")
      seen_manifest_hashes.add(objective.manifest_sha256)
    seen_model_identities.add(model_identity)
    seen_score_hashes.add(model.objective_scores_sha256)
  for calibration, expected in zip(
      config.calibrations, _EXPECTED_CALIBRATIONS, strict=True
  ):
    actual = (
        calibration.ordinal,
        calibration.aggregation,
        calibration.mozc_order_prior_microlog_probability,
        calibration.exact_key_copy_penalty_microlog_probability,
    )
    if actual != expected:
      raise ValueError("quality-regression development calibration grid is invalid")


def _validate_inputs(
    config: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReportConfig,
    frozen_corpus: quality_regression_frozen_corpus_pb2.QualityRegressionFrozenCorpus,
    answer_corpus: quality_regression_corpus_pb2.QualityRegressionAnswerCorpus,
    objective_scores: tuple[
        quality_regression_objective_evaluation_pb2.DevelopmentObjectiveScores,
        ...,
    ],
    objective_score_hashes: tuple[str, ...],
    hashes: dict[str, str],
) -> None:
  if (
      hashes["frozen_corpus_sha256"] != config.frozen_corpus_sha256
      or hashes["answer_corpus_sha256"] != config.answer_corpus_sha256
      or len(objective_scores) != len(config.models)
      or len(objective_score_hashes) != len(config.models)
  ):
    raise ValueError("quality-regression development report input hash mismatch")
  if (
      frozen_corpus.schema_version != config.frozen_corpus_schema_version
      or answer_corpus.schema_version != config.answer_corpus_schema_version
      or len(frozen_corpus.cases) != config.expected_case_count
      or len(answer_corpus.cases) != config.expected_case_count
  ):
    raise ValueError("quality-regression development report input shape mismatch")
  _require_same_message(
      frozen_corpus.identity,
      config.expected_corpus_identity,
      "quality-regression frozen corpus identity mismatch",
  )
  _require_same_message(
      answer_corpus.identity,
      config.expected_corpus_identity,
      "quality-regression answer corpus identity mismatch",
  )
  for model, scores, score_hash in zip(
      config.models, objective_scores, objective_score_hashes, strict=True
  ):
    if (
        score_hash != model.objective_scores_sha256
        or scores.schema_version != config.score_artifact_schema_version
        or len(scores.objectives) != len(_OBJECTIVE_ORDER)
    ):
      raise ValueError("quality-regression development score identity mismatch")
    _require_same_message(
        scores.identity,
        model.score_identity,
        "quality-regression development score identity mismatch",
    )
    for index, objective in enumerate(scores.objectives):
      if (
          objective.objective != _OBJECTIVE_ORDER[index]
          or objective.objective != model.objectives[index].objective
          or len(objective.cases) != config.expected_case_count
      ):
        raise ValueError("quality-regression objective score order mismatch")
  baseline_correct = sum(
      frozen_case.mozc_baseline_output == answer_case.normalized_whole_output
      for frozen_case, answer_case in zip(
          frozen_corpus.cases, answer_corpus.cases, strict=True
      )
  )
  if baseline_correct != config.expected_baseline_correct_count:
    raise ValueError("quality-regression baseline correct count mismatch")


def _candidate_by_id(segment) -> dict[int, Any]:
  return {candidate.id: candidate for candidate in segment.candidates}


def _ranked_segment_order(segment, ranked_candidate_ids: Iterable[int]) -> list[int]:
  candidate_by_id = _candidate_by_id(segment)
  ranked: list[int] = []
  seen: set[int] = set()
  for candidate_id in ranked_candidate_ids:
    if candidate_id not in candidate_by_id or candidate_id in seen:
      raise ValueError("quality-regression score candidate order is invalid")
    if candidate_by_id[candidate_id].is_protected:
      raise ValueError("quality-regression score candidate is protected")
    ranked.append(candidate_id)
    seen.add(candidate_id)
  ranked.extend(
      candidate.id
      for candidate in segment.candidates
      if not candidate.is_protected and candidate.id not in seen
  )
  ranked_index = 0
  merged: list[int] = []
  for candidate in segment.candidates:
    if candidate.is_protected:
      merged.append(candidate.id)
    else:
      merged.append(ranked[ranked_index])
      ranked_index += 1
  return merged


def _score_bits_to_double(score_bits: int) -> float:
  score = struct.unpack("<d", struct.pack("<Q", score_bits))[0]
  if not math.isfinite(score) or (score == 0.0 and score_bits != 0):
    raise ValueError("quality-regression persisted score bits are invalid")
  return score


def _calibrated_ranked_ids(
    segment,
    segment_scores: quality_regression_objective_evaluation_pb2
    .DevelopmentSegmentScores,
    calibration: quality_regression_development_report_pb2
    .DevelopmentCalibrationSpec,
) -> list[int]:
  candidate_by_id = _candidate_by_id(segment)
  adjusted: list[tuple[float, int, int]] = []
  for candidate_score in segment_scores.candidates:
    candidate = candidate_by_id[candidate_score.candidate_id]
    if calibration.aggregation == _FULL_CONTINUATION_SUM:
      score = _score_bits_to_double(
          candidate_score.full_continuation_score_bits
      )
    elif calibration.aggregation == _CONTINUATION_MEAN:
      if candidate_score.scored_continuation_token_count == 0:
        raise ValueError("quality-regression continuation score has no token")
      score = (
          _score_bits_to_double(candidate_score.full_continuation_score_bits)
          / candidate_score.scored_continuation_token_count
      )
    else:
      raise ValueError("quality-regression score aggregation is invalid")
    score -= (
        calibration.mozc_order_prior_microlog_probability
        * candidate_score.original_candidate_index
        / 1_000_000.0
    )
    if segment.candidates[0].value != segment.key and candidate.value == segment.key:
      score -= (
          calibration.exact_key_copy_penalty_microlog_probability
          / 1_000_000.0
      )
    adjusted.append(
        (
            -score,
            candidate_score.original_candidate_index,
            candidate_score.candidate_id,
        )
    )
  return [candidate_id for _, _, candidate_id in sorted(adjusted)]


def _model_top_output(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    case_scores: quality_regression_objective_evaluation_pb2
    .DevelopmentCaseScores,
    calibration: quality_regression_development_report_pb2
    .DevelopmentCalibrationSpec,
) -> tuple[str, dict[int, int]]:
  scores_by_segment = {segment.segment_id: segment for segment in case_scores.segments}
  values: list[str] = []
  top_by_segment: dict[int, int] = {}
  for segment in request.segments:
    if segment.id in scores_by_segment:
      ranked_ids = _calibrated_ranked_ids(
          segment, scores_by_segment[segment.id], calibration
      )
      order = _ranked_segment_order(segment, ranked_ids)
    else:
      order = [candidate.id for candidate in segment.candidates]
    top_id = order[0]
    top_by_segment[segment.id] = top_id
    values.append(_candidate_by_id(segment)[top_id].value)
  return "".join(values), top_by_segment


def _levenshtein(left: str, right: str) -> int:
  left_scalars = tuple(left)
  right_scalars = tuple(right)
  if len(left_scalars) < len(right_scalars):
    left_scalars, right_scalars = right_scalars, left_scalars
  previous = list(range(len(right_scalars) + 1))
  for left_index, left_scalar in enumerate(left_scalars, start=1):
    current = [left_index]
    for right_index, right_scalar in enumerate(right_scalars, start=1):
      current.append(
          min(
              current[-1] + 1,
              previous[right_index] + 1,
              previous[right_index - 1] + (left_scalar != right_scalar),
          )
      )
    previous = current
  return previous[-1]


def _exact_bootstrap_endpoints(
    win_count: int, regression_count: int, tie_count: int
) -> tuple[int, int]:
  sample_size = win_count + regression_count + tie_count
  if sample_size <= 0:
    raise ValueError("quality-regression bootstrap sample is empty")
  mass = {0: 1}
  for _ in range(sample_size):
    next_mass: dict[int, int] = {}
    for value, count in mass.items():
      for delta, multiplier in (
          (-1, regression_count),
          (0, tie_count),
          (1, win_count),
      ):
        if multiplier:
          next_mass[value + delta] = (
              next_mass.get(value + delta, 0) + count * multiplier
          )
    mass = next_mass
  total_mass = sample_size**sample_size
  cumulative = 0
  lower: int | None = None
  upper: int | None = None
  for value in sorted(mass):
    cumulative += mass[value]
    if lower is None and 40 * cumulative >= total_mass:
      lower = value
    if upper is None and 40 * cumulative >= 39 * total_mass:
      upper = value
  if lower is None or upper is None:
    raise ValueError("quality-regression bootstrap quantiles are incomplete")
  return lower, upper


def _set_ratio(
    ratio: exact_quality_metrics_pb2.ExactRatio,
    numerator: int,
    denominator: int,
) -> None:
  if denominator <= 0:
    raise ValueError("quality-regression ratio denominator is invalid")
  ratio.numerator = numerator
  ratio.denominator = denominator


def _clean_key_copy_promotions(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    top_by_segment: dict[int, int],
) -> frozenset[int]:
  baseline_left: list[str] = []
  result: set[int] = set()
  for segment in request.segments:
    candidate_by_id = _candidate_by_id(segment)
    baseline_value = segment.candidates[0].value
    model_value = candidate_by_id[top_by_segment[segment.id]].value
    selected_values = {candidate.value for candidate in segment.candidates}
    clean_context = request.preceding_text + "".join(baseline_left)
    if (
        baseline_value != segment.key
        and model_value == segment.key
        and segment.key in selected_values
        and segment.key not in clean_context
    ):
      result.add(segment.id)
    baseline_left.append(baseline_value)
  return frozenset(result)


def _has_exact_key_regression(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    top_by_segment: dict[int, int],
) -> bool:
  for segment in request.segments:
    candidate_by_id = _candidate_by_id(segment)
    if (
        segment.candidates[0].value != segment.key
        and candidate_by_id[top_by_segment[segment.id]].value == segment.key
    ):
      return True
  return False


def _case_outputs(
    frozen_corpus: quality_regression_frozen_corpus_pb2.QualityRegressionFrozenCorpus,
    answer_corpus: quality_regression_corpus_pb2.QualityRegressionAnswerCorpus,
    objective_scores: quality_regression_objective_evaluation_pb2
    .DevelopmentObjectiveScoreSet,
    calibration: quality_regression_development_report_pb2
    .DevelopmentCalibrationSpec,
) -> tuple[_CaseOutput, ...]:
  outputs: list[_CaseOutput] = []
  for frozen_case, answer_case, case_scores in zip(
      frozen_corpus.cases,
      answer_corpus.cases,
      objective_scores.cases,
      strict=True,
  ):
    if (
        frozen_case.source_line != answer_case.source_line
        or frozen_case.source_line != case_scores.source_line
    ):
      raise ValueError("quality-regression source-line order mismatch")
    model_output, top_by_segment = _model_top_output(
        frozen_case.request,
        case_scores,
        calibration,
    )
    baseline_correct = (
        frozen_case.mozc_baseline_output == answer_case.normalized_whole_output
    )
    model_correct = model_output == answer_case.normalized_whole_output
    outputs.append(
        _CaseOutput(
            source_line=frozen_case.source_line,
            baseline_output=frozen_case.mozc_baseline_output,
            model_output=model_output,
            answer=answer_case.normalized_whole_output,
            exact_key_regression=(
                baseline_correct
                and not model_correct
                and _has_exact_key_regression(frozen_case.request, top_by_segment)
            ),
            clean_key_copy_promotions=_clean_key_copy_promotions(
                frozen_case.request, top_by_segment
            ),
        )
    )
  return tuple(outputs)


def _build_objective_metric(
    config: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReportConfig,
    model_ordinal: int,
    objective: int,
    selection_eligible: bool,
    calibration: quality_regression_development_report_pb2
    .DevelopmentCalibrationSpec,
    outputs: tuple[_CaseOutput, ...],
    reference_outputs: tuple[_CaseOutput, ...] | None,
) -> quality_regression_development_report_pb2.QualityRegressionDevelopmentObjectiveMetric:
  metric = (
      quality_regression_development_report_pb2
      .QualityRegressionDevelopmentObjectiveMetric()
  )
  metric.model_ordinal = model_ordinal
  metric.objective = objective
  metric.selection_eligible = selection_eligible
  metric.calibration.CopyFrom(calibration)
  metric.case_count = len(outputs)
  for name in (
      "baseline_correct_count",
      "model_correct_count",
      "retained_correct_count",
      "win_count",
      "regression_count",
      "unchanged_incorrect_count",
      "mozc_correct_exact_key_regression_count",
      "clean_key_copy_promotion_count",
      "clean_key_copy_removed_vs_reference",
      "clean_key_copy_added_vs_reference",
  ):
    setattr(metric, name, 0)
  baseline_edit_distance = 0
  model_edit_distance = 0
  reference_scalars = 0
  for case in outputs:
    baseline_correct = case.baseline_output == case.answer
    model_correct = case.model_output == case.answer
    metric.baseline_correct_count += baseline_correct
    metric.model_correct_count += model_correct
    if baseline_correct and model_correct:
      metric.retained_correct_count += 1
    elif not baseline_correct and model_correct:
      metric.win_count += 1
    elif baseline_correct and not model_correct:
      metric.regression_count += 1
    else:
      metric.unchanged_incorrect_count += 1
    metric.mozc_correct_exact_key_regression_count += case.exact_key_regression
    metric.clean_key_copy_promotion_count += len(case.clean_key_copy_promotions)
    baseline_edit_distance += _levenshtein(case.baseline_output, case.answer)
    model_edit_distance += _levenshtein(case.model_output, case.answer)
    reference_scalars += len(tuple(case.answer))
  metric.baseline_character_error.edit_distance_sum = baseline_edit_distance
  metric.baseline_character_error.reference_unicode_scalar_count = reference_scalars
  metric.model_character_error.edit_distance_sum = model_edit_distance
  metric.model_character_error.reference_unicode_scalar_count = reference_scalars
  tie_count = metric.case_count - metric.win_count - metric.regression_count
  lower, upper = _exact_bootstrap_endpoints(
      metric.win_count, metric.regression_count, tie_count
  )
  metric.paired_bootstrap.definition_version = (
      config.exact_bootstrap_definition_version
  )
  metric.paired_bootstrap.sample_size = metric.case_count
  metric.paired_bootstrap.win_count = metric.win_count
  metric.paired_bootstrap.regression_count = metric.regression_count
  metric.paired_bootstrap.tie_count = tie_count
  metric.paired_bootstrap.lower_gain_cases = lower
  metric.paired_bootstrap.upper_gain_cases = upper
  metric.paired_bootstrap.gain_denominator_cases = metric.case_count
  _set_ratio(metric.baseline_accuracy, metric.baseline_correct_count, metric.case_count)
  _set_ratio(metric.model_accuracy, metric.model_correct_count, metric.case_count)
  _set_ratio(
      metric.accuracy_gain,
      metric.win_count - metric.regression_count,
      metric.case_count,
  )
  if metric.baseline_correct_count:
    _set_ratio(
        metric.retention,
        metric.retained_correct_count,
        metric.baseline_correct_count,
    )
  if reference_outputs is not None:
    for case, reference in zip(outputs, reference_outputs, strict=True):
      removed = reference.clean_key_copy_promotions - case.clean_key_copy_promotions
      added = case.clean_key_copy_promotions - reference.clean_key_copy_promotions
      metric.clean_key_copy_removed_vs_reference += len(removed)
      metric.clean_key_copy_added_vs_reference += len(added)
  metric.development_gate_passed = (
      selection_eligible
      and metric.win_count - metric.regression_count
      >= config.minimum_net_gain_cases
      and metric.paired_bootstrap.lower_gain_cases > 0
      and metric.regression_count <= config.maximum_baseline_correct_regressions
      and metric.clean_key_copy_removed_vs_reference
      > metric.clean_key_copy_added_vs_reference
      and metric.mozc_correct_exact_key_regression_count == 0
  )
  return metric


def _select_objective(
    metrics: Iterable[
        quality_regression_development_report_pb2
        .QualityRegressionDevelopmentObjectiveMetric
    ],
    models: Iterable[
        quality_regression_development_report_pb2.DevelopmentModelSpec
    ],
) -> quality_regression_development_report_pb2.QualityRegressionDevelopmentObjectiveMetric | None:
  passing = [metric for metric in metrics if metric.development_gate_passed]
  if not passing:
    return None
  model_by_ordinal = {model.ordinal: model for model in models}
  return sorted(
      passing,
      key=lambda metric: (
          -(metric.win_count - metric.regression_count),
          -metric.retained_correct_count,
          metric.model_character_error.edit_distance_sum,
          metric.clean_key_copy_promotion_count,
          model_by_ordinal[metric.model_ordinal].parameter_count,
          metric.model_ordinal,
          metric.objective,
          metric.calibration.aggregation,
          metric.calibration.mozc_order_prior_microlog_probability,
          metric.calibration.exact_key_copy_penalty_microlog_probability,
          metric.calibration.ordinal,
      ),
  )[0]


def build_development_report(
    inputs: QualityRegressionDevelopmentReportInputs,
) -> quality_regression_development_report_pb2.QualityRegressionDevelopmentReport:
  report_config_sha256 = _sha256(inputs.report_config_textproto)
  config = _parse_textproto(
      quality_regression_development_report_pb2
      .QualityRegressionDevelopmentReportConfig,
      inputs.report_config_textproto,
      "quality-regression development report config",
  )
  _validate_config(config)
  hashes = {
      "frozen_corpus_sha256": _sha256(inputs.frozen_corpus_binary),
      "answer_corpus_sha256": _sha256(inputs.answer_corpus_binary),
  }
  objective_score_hashes = tuple(
      _sha256(data) for data in inputs.objective_scores_binaries
  )
  frozen_corpus = _parse_binary(
      quality_regression_frozen_corpus_pb2.QualityRegressionFrozenCorpus,
      inputs.frozen_corpus_binary,
      "quality-regression frozen corpus",
  )
  answer_corpus = _parse_binary(
      quality_regression_corpus_pb2.QualityRegressionAnswerCorpus,
      inputs.answer_corpus_binary,
      "quality-regression answer corpus",
  )
  objective_scores = tuple(
      _parse_binary(
          quality_regression_objective_evaluation_pb2.DevelopmentObjectiveScores,
          data,
          f"quality-regression development objective scores {index}",
      )
      for index, data in enumerate(inputs.objective_scores_binaries)
  )
  _validate_inputs(
      config,
      frozen_corpus,
      answer_corpus,
      objective_scores,
      objective_score_hashes,
      hashes,
  )

  report = quality_regression_development_report_pb2.QualityRegressionDevelopmentReport()
  report.schema_version = config.report_schema_version
  identity = report.identity
  identity.report_config_sha256 = report_config_sha256
  identity.frozen_corpus_sha256 = config.frozen_corpus_sha256
  identity.answer_corpus_sha256 = config.answer_corpus_sha256
  for model in config.models:
    identity.models.add().CopyFrom(model)
  identity.corpus_identity.CopyFrom(config.expected_corpus_identity)
  identity.metric_definition_version = config.metric_definition_version
  identity.exact_bootstrap_definition_version = (
      config.exact_bootstrap_definition_version
  )
  identity.clean_key_copy_definition_version = (
      config.clean_key_copy_definition_version
  )
  identity.selection_policy_version = config.selection_policy_version

  for model, model_scores in zip(
      config.models, objective_scores, strict=True
  ):
    reference_score_set = model_scores.objectives[0]
    reference_outputs = _case_outputs(
        frozen_corpus,
        answer_corpus,
        reference_score_set,
        config.calibrations[0],
    )
    for objective_index, objective_score_set in enumerate(
        model_scores.objectives
    ):
      for calibration in config.calibrations:
        outputs = _case_outputs(
            frozen_corpus,
            answer_corpus,
            objective_score_set,
            calibration,
        )
        report.objectives.add().CopyFrom(
            _build_objective_metric(
                config,
                model.ordinal,
                objective_score_set.objective,
                model.objectives[objective_index].selection_eligible,
                calibration,
                outputs,
                None
                if (
                    objective_score_set.objective == _STRUCTURED_WITH_TARGET
                    and calibration.ordinal == 0
                )
                else reference_outputs,
            )
          )
  selected = _select_objective(report.objectives, config.models)
  report.selected = selected is not None
  if selected is not None:
    report.selected_model_ordinal = selected.model_ordinal
    report.selected_objective = selected.objective
    model = config.models[selected.model_ordinal]
    index = _OBJECTIVE_ORDER.index(selected.objective)
    report.selected_manifest_sha256 = model.objectives[index].manifest_sha256
    report.selected_calibration.CopyFrom(selected.calibration)
  _validate_report(report, config, report_config_sha256)
  return report


def _validate_report(
    report: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReport,
    config: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReportConfig,
    report_config_sha256: str,
) -> None:
  _require_initialized_without_unknown_fields(
      report, "quality-regression development report"
  )
  if (
      report.schema_version != config.report_schema_version
      or report.identity.report_config_sha256 != report_config_sha256
      or report.identity.frozen_corpus_sha256 != config.frozen_corpus_sha256
      or report.identity.answer_corpus_sha256 != config.answer_corpus_sha256
      or len(report.identity.models) != len(config.models)
      or report.identity.metric_definition_version
      != config.metric_definition_version
      or report.identity.exact_bootstrap_definition_version
      != config.exact_bootstrap_definition_version
      or report.identity.clean_key_copy_definition_version
      != config.clean_key_copy_definition_version
      or report.identity.selection_policy_version != config.selection_policy_version
      or len(report.objectives)
      != len(config.models) * len(_OBJECTIVE_ORDER) * len(config.calibrations)
  ):
    raise ValueError("quality-regression development report identity is invalid")
  for actual, expected in zip(
      report.identity.models, config.models, strict=True
  ):
    _require_same_message(
        actual,
        expected,
        "quality-regression development report model identity mismatch",
    )
  _require_same_message(
      report.identity.corpus_identity,
      config.expected_corpus_identity,
      "quality-regression development report corpus identity mismatch",
  )
  selected_metrics = [metric for metric in report.objectives if metric.development_gate_passed]
  if report.selected != bool(selected_metrics):
    raise ValueError("quality-regression selected flag is invalid")
  expected_metrics = (
      (model.ordinal, objective, calibration)
      for model in config.models
      for objective in _OBJECTIVE_ORDER
      for calibration in config.calibrations
  )
  for metric, (model_ordinal, objective, calibration) in zip(
      report.objectives, expected_metrics, strict=True
  ):
    if (
        metric.model_ordinal != model_ordinal
        or metric.objective != objective
    ):
      raise ValueError("quality-regression report objective order is invalid")
    _require_same_message(
        metric.calibration,
        calibration,
        "quality-regression report calibration order is invalid",
    )
    if (
        metric.case_count != config.expected_case_count
        or metric.baseline_correct_count
        != config.expected_baseline_correct_count
        or metric.baseline_correct_count
        != metric.retained_correct_count + metric.regression_count
        or metric.model_correct_count != metric.retained_correct_count + metric.win_count
        or metric.retained_correct_count
        + metric.win_count
        + metric.regression_count
        + metric.unchanged_incorrect_count
        != metric.case_count
        or metric.paired_bootstrap.sample_size != metric.case_count
        or metric.paired_bootstrap.definition_version
        != config.exact_bootstrap_definition_version
        or metric.paired_bootstrap.win_count != metric.win_count
        or metric.paired_bootstrap.regression_count != metric.regression_count
    ):
      raise ValueError("quality-regression report metric algebra is invalid")
  if report.selected:
    selected = _select_objective(report.objectives, config.models)
    if (
        selected is None
        or not report.HasField("selected_model_ordinal")
        or report.selected_model_ordinal != selected.model_ordinal
        or not report.HasField("selected_objective")
        or report.selected_objective != selected.objective
        or not report.HasField("selected_manifest_sha256")
        or report.selected_manifest_sha256
        != config.models[selected.model_ordinal]
        .objectives[_OBJECTIVE_ORDER.index(selected.objective)]
        .manifest_sha256
        or not report.HasField("selected_calibration")
    ):
      raise ValueError("quality-regression selected objective is invalid")
    _require_same_message(
        report.selected_calibration,
        selected.calibration,
        "quality-regression selected calibration is invalid",
    )
  elif (
      report.HasField("selected_model_ordinal")
      or report.HasField("selected_objective")
      or report.HasField("selected_manifest_sha256")
      or report.HasField("selected_calibration")
  ):
    raise ValueError("quality-regression selected fields are present without selection")


def _json_value(value: Any, field: FieldDescriptor) -> Any:
  if field.type == FieldDescriptor.TYPE_MESSAGE:
    return _message_to_tag_ordered_json(value)
  if field.type == FieldDescriptor.TYPE_ENUM:
    return field.enum_type.values_by_number[int(value)].name
  if field.type == FieldDescriptor.TYPE_BOOL:
    return bool(value)
  if field.type == FieldDescriptor.TYPE_STRING:
    return value
  if field.type == FieldDescriptor.TYPE_BYTES:
    raise ValueError("quality-regression report JSON cannot contain bytes")
  if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
    raise ValueError("quality-regression report JSON cannot contain floating point")
  return str(value)


def _message_to_tag_ordered_json(message: Message) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for field in sorted(message.DESCRIPTOR.fields, key=lambda item: item.number):
    if field.is_repeated:
      values = getattr(message, field.name)
      if values:
        result[field.name] = [_json_value(value, field) for value in values]
    elif field.has_presence and not message.HasField(field.name):
      continue
    else:
      result[field.name] = _json_value(getattr(message, field.name), field)
  return result


def serialize_development_report(
    report: quality_regression_development_report_pb2
    .QualityRegressionDevelopmentReport,
) -> QualityRegressionSerializedDevelopmentReport:
  _require_initialized_without_unknown_fields(
      report, "quality-regression development report"
  )
  binary = report.SerializeToString(deterministic=True)
  canonical_json = (
      json.dumps(
          _message_to_tag_ordered_json(report),
          ensure_ascii=False,
          separators=(",", ":"),
      )
      + "\n"
  ).encode("utf-8")
  return QualityRegressionSerializedDevelopmentReport(
      binary=binary, canonical_json=canonical_json
  )
