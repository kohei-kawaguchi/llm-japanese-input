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

"""Builds deterministic, answer-aware AJIMEE quality reports."""

from __future__ import annotations

from collections import Counter
import dataclasses
import datetime
import hashlib
import json
import re
from typing import Any, Iterable

from google.protobuf import text_format
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from engine.evaluation import ajimee_capacity_audit_pb2
from engine.evaluation import ajimee_corpus_pb2
from engine.evaluation import ajimee_frozen_corpus_pb2
from engine.evaluation import ajimee_quality_report_pb2
from engine.evaluation import ajimee_semantic_results_pb2
from engine.evaluation import evaluation_artifact_pb2
from engine.evaluation import exact_quality_metrics_pb2
from engine.evaluation import frozen_candidate_ranker_pb2


_QUALITY_REPORT_CONFIG_SCHEMA_VERSION = 1
_ANSWER_CORPUS_SCHEMA_VERSION = 1
_FROZEN_CORPUS_SCHEMA_VERSION = 2
_CAPACITY_AUDIT_CONFIG_SCHEMA_VERSION = 1
_CAPACITY_AUDIT_SCHEMA_VERSION = 1
_MODEL_MANIFEST_SCHEMA_VERSION = 4
_SEMANTIC_RESULTS_SCHEMA_VERSION = 1
_QUALITY_REPORT_SCHEMA_VERSION = 1
_METRIC_DEFINITION_VERSION = 1
_EXACT_BOOTSTRAP_DEFINITION_VERSION = 1
_EXPECTED_CASE_COUNT = 200
_EXPECTED_CONTEXT_CASE_COUNT = 100
_EXPECTED_NO_CONTEXT_CASE_COUNT = 100
_CHECKED_ANSWER_CORPUS_SHA256 = (
    "819465115d76cca65eb9d6c8f3f60f8850723376b440eb5c41f0cd8fd7b2c389"
)
_CHECKED_FROZEN_CORPUS_SHA256 = (
    "9b1656a48ed3cfbf1577b7a404688affb49b793f9c556ba2d2fb46b41238c61c"
)
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_UTC = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_KNOWN_STATUS_CODES = frozenset(range(17))
_KNOWN_CAPACITY_LIMITS = frozenset(range(1, 10))
_USAGE_FIELDS = tuple(
    field.name
    for field in ajimee_capacity_audit_pb2.AjimeeCapacityAuditUsage.DESCRIPTOR.fields
)
_SLICE_ORDER = (
    ajimee_quality_report_pb2.AJIMEE_REPORT_OVERALL,
    ajimee_quality_report_pb2.AJIMEE_REPORT_HAS_CONTEXT,
    ajimee_quality_report_pb2.AJIMEE_REPORT_NO_CONTEXT,
)
_PINNED_PRE_MODEL_FACTS = {
    ajimee_quality_report_pb2.AJIMEE_REPORT_OVERALL:
        (103, 180, 251, 4246),
    ajimee_quality_report_pb2.AJIMEE_REPORT_HAS_CONTEXT:
        (46, 88, 158, 2039),
    ajimee_quality_report_pb2.AJIMEE_REPORT_NO_CONTEXT:
        (57, 92, 93, 2207),
}


@dataclasses.dataclass(frozen=True)
class AjimeeQualityReportInputs:
  report_config_textproto: bytes
  capacity_audit_config_textproto: bytes
  answer_corpus_binary: bytes
  frozen_corpus_binary: bytes
  capacity_audit_binary: bytes
  semantic_results_binary: bytes


@dataclasses.dataclass(frozen=True)
class AjimeeSerializedQualityReport:
  binary: bytes
  canonical_json: bytes


@dataclasses.dataclass(frozen=True)
class _RequestFacts:
  segment_count: int
  candidate_count: int
  unprotected_candidate_count: int
  rankable_segment_count: int
  string_bytes: int
  max_unprotected_candidates_in_segment: int
  segments: tuple[tuple[int, int], ...]


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


def _require_lower_hex(value: str, size: int, label: str) -> None:
  pattern = _LOWER_HEX_40 if size == 40 else _LOWER_HEX_64
  if pattern.fullmatch(value) is None:
    raise ValueError(f"{label} is not a lowercase hexadecimal identity")


def _require_utf8(value: str, _label: str) -> bytes:
  return value.encode("utf-8")


def _validate_source_identity(
    source: evaluation_artifact_pb2.EvaluationSourceIdentity,
) -> None:
  _require_initialized_without_unknown_fields(source, "AJIMEE source identity")
  values = (
      source.benchmark_name,
      source.source_revision,
      source.source_relative_path,
      source.creator,
      source.source_url,
      source.license_identifier,
      source.license_url,
      source.upstream_dataset_url,
      source.changes_notice,
  )
  if any(not value for value in values):
    raise ValueError("AJIMEE source identity is invalid")
  for index, value in enumerate(values):
    _require_utf8(value, f"AJIMEE source identity field {index}")
  _require_lower_hex(source.source_revision, 40, "AJIMEE source revision")
  _require_lower_hex(source.source_sha256, 64, "AJIMEE source SHA256")


def _validate_quality_config(
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
) -> None:
  if (
      config.schema_version != _QUALITY_REPORT_CONFIG_SCHEMA_VERSION
      or config.answer_corpus_schema_version != _ANSWER_CORPUS_SCHEMA_VERSION
      or config.frozen_corpus_schema_version != _FROZEN_CORPUS_SCHEMA_VERSION
      or config.capacity_audit_schema_version != _CAPACITY_AUDIT_SCHEMA_VERSION
      or config.semantic_results_schema_version != _SEMANTIC_RESULTS_SCHEMA_VERSION
      or config.quality_report_schema_version != _QUALITY_REPORT_SCHEMA_VERSION
      or config.metric_definition_version != _METRIC_DEFINITION_VERSION
      or config.exact_bootstrap_definition_version
      != _EXACT_BOOTSTRAP_DEFINITION_VERSION
  ):
    raise ValueError("AJIMEE quality report config schema is invalid")
  if (
      config.expected_case_count != _EXPECTED_CASE_COUNT
      or config.expected_context_case_count != _EXPECTED_CONTEXT_CASE_COUNT
      or config.expected_no_context_case_count != _EXPECTED_NO_CONTEXT_CASE_COUNT
      or config.expected_context_case_count
      + config.expected_no_context_case_count
      != config.expected_case_count
  ):
    raise ValueError("AJIMEE quality report config counts are invalid")
  for name in (
      "input_corpus_sha256",
      "answer_corpus_sha256",
      "frozen_corpus_sha256",
      "capacity_audit_config_sha256",
      "capacity_audit_sha256",
      "semantic_results_sha256",
  ):
    _require_lower_hex(getattr(config, name), 64, f"config {name}")
  _validate_source_identity(config.expected_source)


def _validate_capacity_config(
    config: ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
) -> None:
  if (
      config.schema_version != _CAPACITY_AUDIT_CONFIG_SCHEMA_VERSION
      or config.frozen_corpus_schema_version != _FROZEN_CORPUS_SCHEMA_VERSION
      or config.capacity_audit_schema_version != _CAPACITY_AUDIT_SCHEMA_VERSION
      or config.model_manifest_schema_version != _MODEL_MANIFEST_SCHEMA_VERSION
  ):
    raise ValueError("AJIMEE capacity audit config schema is invalid")
  if config.expected_case_count != _EXPECTED_CASE_COUNT:
    raise ValueError("AJIMEE capacity audit config case count is invalid")
  for name in (
      "frozen_corpus_sha256",
      "model_manifest_sha256",
      "gguf_sha256",
      "tokenizer_sha256",
  ):
    _require_lower_hex(getattr(config, name), 64, f"capacity config {name}")
  for name in ("source_revision", "runtime_revision"):
    _require_lower_hex(getattr(config, name), 40, f"capacity config {name}")
  if (
      not config.gguf_file_name
      or config.gguf_file_name in (".", "..")
      or any(char in config.gguf_file_name for char in "/\\:\0")
      or not config.quantization
  ):
    raise ValueError("AJIMEE capacity audit config identity is invalid")


def _validate_frozen_corpus(
    corpus: ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
) -> None:
  if corpus.schema_version != config.frozen_corpus_schema_version:
    raise ValueError("AJIMEE frozen corpus schema is invalid")
  _validate_source_identity(corpus.source)
  if corpus.source != config.expected_source:
    raise ValueError("AJIMEE frozen corpus source identity does not match")
  _require_lower_hex(corpus.input_corpus_sha256, 64, "input corpus SHA256")
  if corpus.input_corpus_sha256 != config.input_corpus_sha256:
    raise ValueError("AJIMEE frozen input corpus SHA256 does not match")
  mozc = corpus.mozc
  _require_lower_hex(mozc.source_revision, 40, "Mozc source revision")
  _require_lower_hex(mozc.data_sha256, 64, "Mozc data SHA256")
  _require_lower_hex(
      mozc.default_desktop_request_sha256, 64, "Mozc request SHA256"
  )
  _require_lower_hex(
      mozc.default_desktop_config_sha256, 64, "Mozc config SHA256"
  )
  if not mozc.data_type or _CANONICAL_UTC.fullmatch(
      mozc.evaluation_clock_utc_rfc3339
  ) is None:
    raise ValueError("AJIMEE Mozc identity is invalid")
  datetime.datetime.strptime(
      mozc.evaluation_clock_utc_rfc3339, "%Y-%m-%dT%H:%M:%SZ"
  )
  if len(corpus.cases) != config.expected_case_count:
    raise ValueError("AJIMEE frozen corpus case count does not match")

  previous_source_index = -1
  expected_candidate_id = 0
  for case_ordinal, frozen_case in enumerate(corpus.cases):
    if frozen_case.source_index <= previous_source_index:
      raise ValueError("AJIMEE frozen source IDs are unordered")
    previous_source_index = frozen_case.source_index
    if not frozen_case.normalized_hiragana_reading:
      raise ValueError("AJIMEE frozen reading is empty")
    _require_utf8(frozen_case.normalized_hiragana_reading, "frozen reading")
    _require_utf8(frozen_case.baseline_output, "frozen baseline")
    request = frozen_case.request
    if (
        request.mode
        != frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest.MODE_CONVERSION
        or request.reading != frozen_case.normalized_hiragana_reading
        or request.following_text
        or request.focused_segment_id != 0
        or request.token.session_generation != 0
        or request.token.state_revision != 0
        or request.token.request_sequence != case_ordinal + 1
        or not request.segments
    ):
      raise ValueError("AJIMEE frozen request identity is invalid")
    for label, value in (
        ("preceding text", request.preceding_text),
        ("following text", request.following_text),
        ("request reading", request.reading),
    ):
      _require_utf8(value, label)
    expected_baseline: list[str] = []
    expected_candidate_id = 0
    for segment_ordinal, segment in enumerate(request.segments):
      if segment.id != segment_ordinal or not segment.candidates:
        raise ValueError("AJIMEE frozen segment structure is invalid")
      _require_utf8(segment.key, "frozen segment key")
      expected_baseline.append(segment.candidates[0].value)
      for candidate in segment.candidates:
        if candidate.id != expected_candidate_id:
          raise ValueError("AJIMEE frozen candidate IDs are unordered")
        expected_candidate_id += 1
        _require_utf8(candidate.key, "frozen candidate key")
        _require_utf8(candidate.value, "frozen candidate value")
    if frozen_case.baseline_output != "".join(expected_baseline):
      raise ValueError("AJIMEE frozen baseline is inconsistent")


def _validate_answers(
    answers: ajimee_corpus_pb2.AjimeeAnswerCorpus,
    corpus: ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
) -> None:
  if answers.schema_version != config.answer_corpus_schema_version:
    raise ValueError("AJIMEE answer corpus schema is invalid")
  _validate_source_identity(answers.source)
  if answers.source != corpus.source or answers.source != config.expected_source:
    raise ValueError("AJIMEE answer source identity does not match")
  if len(answers.cases) != len(corpus.cases):
    raise ValueError("AJIMEE answer case count does not match")
  for answer, frozen_case in zip(answers.cases, corpus.cases, strict=True):
    if answer.source_index != frozen_case.source_index:
      raise ValueError("AJIMEE answer cases are unordered")
    if not answer.accepted_whole_outputs:
      raise ValueError("AJIMEE accepted-output list is empty")
    for output in answer.accepted_whole_outputs:
      if not output:
        raise ValueError("AJIMEE accepted output is empty")
      _require_utf8(output, "AJIMEE accepted output")


def _request_facts(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
) -> _RequestFacts:
  candidate_count = 0
  unprotected_count = 0
  rankable_count = 0
  max_unprotected = 0
  string_bytes = sum(
      len(_require_utf8(value, "AJIMEE request string"))
      for value in (request.preceding_text, request.following_text, request.reading)
  )
  segments: list[tuple[int, int]] = []
  for segment in request.segments:
    candidate_count += len(segment.candidates)
    string_bytes += len(_require_utf8(segment.key, "AJIMEE segment key"))
    segment_unprotected = 0
    for candidate in segment.candidates:
      string_bytes += len(_require_utf8(candidate.key, "AJIMEE candidate key"))
      string_bytes += len(_require_utf8(candidate.value, "AJIMEE candidate value"))
      if not candidate.is_protected:
        segment_unprotected += 1
    unprotected_count += segment_unprotected
    rankable_count += segment_unprotected > 1
    max_unprotected = max(max_unprotected, segment_unprotected)
    segments.append((segment.id, segment_unprotected))
  return _RequestFacts(
      segment_count=len(request.segments),
      candidate_count=candidate_count,
      unprotected_candidate_count=unprotected_count,
      rankable_segment_count=rankable_count,
      string_bytes=string_bytes,
      max_unprotected_candidates_in_segment=max_unprotected,
      segments=tuple(segments),
  )


def _capacity_identity_from_config(
    config: ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
) -> tuple[str, ...]:
  return (
      config.frozen_corpus_sha256,
      config.model_manifest_sha256,
      config.gguf_file_name,
      config.gguf_sha256,
      config.tokenizer_sha256,
      config.quantization,
      config.source_revision,
      config.runtime_revision,
  )


def _capacity_identity_tuple(
    identity: ajimee_capacity_audit_pb2.AjimeeCapacityAuditIdentity,
) -> tuple[str, ...]:
  return (
      identity.frozen_corpus_sha256,
      identity.model_manifest_sha256,
      identity.gguf_file_name,
      identity.gguf_sha256,
      identity.tokenizer_sha256,
      identity.quantization,
      identity.source_revision,
      identity.runtime_revision,
  )


def _selected_maximum_for_limit(
    usage: ajimee_capacity_audit_pb2.AjimeeCapacityAuditUsage, limit: int
) -> int:
  fields = {
      ajimee_capacity_audit_pb2.AJIMEE_SERIALIZED_RECORD_BYTES:
          "max_selected_serialized_record_bytes",
      ajimee_capacity_audit_pb2.AJIMEE_NORMALIZED_RECORD_BYTES:
          "max_selected_normalized_record_bytes",
      ajimee_capacity_audit_pb2.AJIMEE_FULL_RECORD_TOKENS:
          "max_selected_full_record_tokens",
      ajimee_capacity_audit_pb2.AJIMEE_SEGMENTS_PER_DECODE:
          "max_segments_in_decode",
      ajimee_capacity_audit_pb2.AJIMEE_SEQUENCES_PER_DECODE:
          "max_sequences_in_decode",
      ajimee_capacity_audit_pb2.AJIMEE_INPUT_NODES_PER_DECODE:
          "max_input_nodes_in_decode",
      ajimee_capacity_audit_pb2.AJIMEE_OUTPUT_ROWS_PER_DECODE:
          "max_output_rows_in_decode",
      ajimee_capacity_audit_pb2.AJIMEE_RESERVED_OUTPUT_ROWS_PER_DECODE:
          "max_reserved_output_rows_in_decode",
      ajimee_capacity_audit_pb2.AJIMEE_OUTPUT_LOGIT_BYTES_PER_DECODE:
          "max_output_logit_bytes_in_decode",
  }
  if limit not in fields:
    raise ValueError("AJIMEE capacity limiter is invalid")
  return getattr(usage, fields[limit])


def _validate_capacity_pass(
    result: ajimee_capacity_audit_pb2.AjimeeCapacityAuditCaseResult,
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
) -> None:
  if (
      result.outcome != ajimee_capacity_audit_pb2.AJIMEE_CAPACITY_AUDIT_PASS
      or result.canonical_status_code != ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
      or not result.HasField("usage")
      or result.HasField("request_limit_violation")
  ):
    raise ValueError("AJIMEE capacity result is not a passing result")
  facts = _request_facts(request)
  usage = result.usage
  expected_request_values = {
      "request_segment_count": facts.segment_count,
      "request_candidate_count": facts.candidate_count,
      "request_unprotected_candidate_count": facts.unprotected_candidate_count,
      "request_rankable_segment_count": facts.rankable_segment_count,
      "request_string_bytes": facts.string_bytes,
      "max_unprotected_candidates_in_segment":
          facts.max_unprotected_candidates_in_segment,
  }
  if any(
      getattr(usage, name) != value
      for name, value in expected_request_values.items()
  ):
    raise ValueError("AJIMEE capacity request usage is inconsistent")
  if len(result.segments) != facts.rankable_segment_count:
    raise ValueError("AJIMEE capacity segment diagnostics are incomplete")

  totals = {
      "window_candidate_count": 0,
      "candidates_omitted_by_window": 0,
      "selected_segment_count": 0,
      "selected_candidate_count": 0,
      "candidates_omitted_by_capacity": 0,
      "segments_omitted_by_capacity": 0,
  }
  diagnostic_ordinal = 0
  for segment_id, unprotected_count in facts.segments:
    if unprotected_count <= 1:
      totals["window_candidate_count"] += unprotected_count
      continue
    diagnostic = result.segments[diagnostic_ordinal]
    diagnostic_ordinal += 1
    if (
        diagnostic.segment_id != segment_id
        or diagnostic.unprotected_candidate_count != unprotected_count
        or diagnostic.window_candidate_count < 2
        or diagnostic.window_candidate_count > unprotected_count
        or diagnostic.omitted_by_window
        != unprotected_count - diagnostic.window_candidate_count
    ):
      raise ValueError("AJIMEE capacity window diagnostic is inconsistent")
    totals["window_candidate_count"] += diagnostic.window_candidate_count
    totals["candidates_omitted_by_window"] += diagnostic.omitted_by_window
    if (
        diagnostic.disposition
        == ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_SELECTED
    ):
      if (
          diagnostic.selected_candidate_count < 2
          or diagnostic.selected_candidate_count > diagnostic.window_candidate_count
          or diagnostic.omitted_by_capacity
          != diagnostic.window_candidate_count - diagnostic.selected_candidate_count
      ):
        raise ValueError("AJIMEE selected capacity diagnostic is inconsistent")
      totals["selected_segment_count"] += 1
      totals["selected_candidate_count"] += diagnostic.selected_candidate_count
    elif (
        diagnostic.disposition
        == ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_OMITTED_CAPACITY
    ):
      if (
          diagnostic.selected_candidate_count != 0
          or diagnostic.omitted_by_capacity != diagnostic.window_candidate_count
      ):
        raise ValueError("AJIMEE omitted capacity diagnostic is inconsistent")
      totals["segments_omitted_by_capacity"] += 1
    else:
      raise ValueError("AJIMEE capacity disposition is invalid")
    totals["candidates_omitted_by_capacity"] += diagnostic.omitted_by_capacity
    if diagnostic.HasField("first_limiter") != (diagnostic.omitted_by_capacity > 0):
      raise ValueError("AJIMEE capacity limiter presence is inconsistent")
    if diagnostic.HasField("first_limiter"):
      limiter = diagnostic.first_limiter
      if (
          limiter.limit not in _KNOWN_CAPACITY_LIMITS
          or limiter.observed <= limiter.allowed
          or _selected_maximum_for_limit(usage, limiter.limit) > limiter.allowed
      ):
        raise ValueError("AJIMEE capacity limiter is inconsistent")
  if any(getattr(usage, name) != value for name, value in totals.items()):
    raise ValueError("AJIMEE aggregate capacity usage is inconsistent")

  selected = usage.selected_segment_count
  selected_maximum_fields = _USAGE_FIELDS[12:]
  if selected == 0:
    if any(getattr(usage, name) != 0 for name in selected_maximum_fields):
      raise ValueError("AJIMEE empty capacity maxima are inconsistent")
  elif (
      usage.decode_batch_count == 0
      or usage.decode_batch_count > selected
      or usage.max_selected_serialized_record_bytes == 0
      or usage.max_selected_normalized_record_bytes == 0
      or usage.max_selected_full_record_tokens == 0
      or usage.max_segments_in_decode == 0
      or usage.max_segments_in_decode > selected
      or usage.max_sequences_in_decode < 2
      or usage.max_sequences_in_decode > usage.selected_candidate_count
      or usage.max_input_nodes_in_decode == 0
      or usage.max_output_rows_in_decode == 0
      or usage.max_output_rows_in_decode > usage.max_input_nodes_in_decode
      or usage.max_reserved_output_rows_in_decode < usage.max_sequences_in_decode
      or usage.max_output_logit_bytes_in_decode == 0
  ):
    raise ValueError("AJIMEE selected capacity maxima are inconsistent")


def _validate_capacity_audit(
    audit: ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
    capacity_config: ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
    corpus: ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
) -> None:
  if audit.schema_version != capacity_config.capacity_audit_schema_version:
    raise ValueError("AJIMEE capacity audit schema is invalid")
  if _capacity_identity_tuple(audit.identity) != _capacity_identity_from_config(
      capacity_config
  ):
    raise ValueError("AJIMEE capacity audit identity does not match")
  if not audit.all_passed or len(audit.cases) != len(corpus.cases):
    raise ValueError("AJIMEE capacity audit is not complete and all-pass")
  for result, frozen_case in zip(audit.cases, corpus.cases, strict=True):
    if result.source_index != frozen_case.source_index:
      raise ValueError("AJIMEE capacity cases are unordered")
    _validate_capacity_pass(result, frozen_case.request)


def _merge_response_orders(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    response_orders: Iterable[ajimee_semantic_results_pb2.AjimeeSemanticSegmentOrder],
) -> tuple[int, list[list[int]] | None]:
  complete_orders = [
      [candidate.id for candidate in segment.candidates]
      for segment in request.segments
  ]
  segment_by_id = {segment.id: index for index, segment in enumerate(request.segments)}
  seen_segments: set[int] = set()
  for response_order in response_orders:
    if response_order.segment_id in seen_segments:
      return ajimee_capacity_audit_pb2.AJIMEE_STATUS_ALREADY_EXISTS, None
    seen_segments.add(response_order.segment_id)
    if response_order.segment_id not in segment_by_id:
      return ajimee_capacity_audit_pb2.AJIMEE_STATUS_NOT_FOUND, None
    segment_index = segment_by_id[response_order.segment_id]
    segment = request.segments[segment_index]
    candidate_by_id = {candidate.id: candidate for candidate in segment.candidates}
    seen_candidates: set[int] = set()
    ranked: list[int] = []
    for candidate_id in response_order.candidate_ids:
      if candidate_id not in candidate_by_id:
        return ajimee_capacity_audit_pb2.AJIMEE_STATUS_NOT_FOUND, None
      if candidate_id in seen_candidates:
        return ajimee_capacity_audit_pb2.AJIMEE_STATUS_ALREADY_EXISTS, None
      if candidate_by_id[candidate_id].is_protected:
        return ajimee_capacity_audit_pb2.AJIMEE_STATUS_PERMISSION_DENIED, None
      seen_candidates.add(candidate_id)
      ranked.append(candidate_id)
    ranked.extend(
        candidate.id
        for candidate in segment.candidates
        if not candidate.is_protected and candidate.id not in seen_candidates
    )
    ranked_ordinal = 0
    merged: list[int] = []
    for candidate in segment.candidates:
      if candidate.is_protected:
        merged.append(candidate.id)
      else:
        merged.append(ranked[ranked_ordinal])
        ranked_ordinal += 1
    complete_orders[segment_index] = merged
  return ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK, complete_orders


def _merged_top(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    complete_orders: list[list[int]],
) -> str:
  values: list[str] = []
  for segment, order in zip(request.segments, complete_orders, strict=True):
    candidate_by_id = {candidate.id: candidate for candidate in segment.candidates}
    values.append(candidate_by_id[order[0]].value)
  return "".join(values)


def _validate_semantic_response_shape(
    result: ajimee_semantic_results_pb2.AjimeeSemanticCaseResult,
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
) -> None:
  if (
      not result.HasField("response_token_matches_request")
      or not result.HasField("response_within_request_bounds")
      or not result.HasField("response_segment_order_count")
      or not result.HasField("response_candidate_id_count")
  ):
    raise ValueError("AJIMEE semantic response metadata is incomplete")
  expected_within_bounds = (
      result.response_segment_order_count <= len(request.segments)
      and result.response_candidate_id_count
      <= sum(len(segment.candidates) for segment in request.segments)
  )
  if result.response_within_request_bounds != expected_within_bounds:
    raise ValueError("AJIMEE semantic response bounds metadata is inconsistent")
  if not result.response_within_request_bounds:
    if result.response_segment_orders:
      raise ValueError("AJIMEE out-of-bounds response stores raw orders")
    return
  if (
      result.response_segment_order_count != len(result.response_segment_orders)
      or result.response_candidate_id_count
      != sum(
          len(order.candidate_ids) for order in result.response_segment_orders
      )
  ):
    raise ValueError("AJIMEE semantic response counts are inconsistent")


def _validate_semantic_results(
    results: ajimee_semantic_results_pb2.AjimeeSemanticResults,
    corpus: ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
    capacity_config: ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
    capacity_audit: ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
) -> tuple[str | None, ...]:
  if results.schema_version != config.semantic_results_schema_version:
    raise ValueError("AJIMEE semantic results schema is invalid")
  identity = results.identity
  if (
      identity.frozen_corpus_sha256 != config.frozen_corpus_sha256
      or identity.capacity_audit_config_sha256
      != config.capacity_audit_config_sha256
      or identity.capacity_audit_sha256 != config.capacity_audit_sha256
      or identity.capacity_audit_identity != capacity_audit.identity
      or _capacity_identity_tuple(identity.capacity_audit_identity)
      != _capacity_identity_from_config(capacity_config)
      or identity.numeric_profile
      != ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_CONFIGURED
      or len(results.cases) != len(corpus.cases)
  ):
    raise ValueError("AJIMEE semantic results identity is invalid")

  successful_outputs: list[str | None] = []
  for result, frozen_case in zip(results.cases, corpus.cases, strict=True):
    if result.source_index != frozen_case.source_index:
      raise ValueError("AJIMEE semantic cases are unordered")
    request = frozen_case.request
    if result.canonical_status_code not in _KNOWN_STATUS_CODES:
      raise ValueError("AJIMEE semantic status code is invalid")
    if result.outcome == ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_BACKEND_ERROR:
      if (
          result.canonical_status_code
          == ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
          or result.HasField("response_token_matches_request")
          or result.response_segment_orders
          or result.HasField("merged_top_output")
          or result.HasField("response_within_request_bounds")
          or result.HasField("response_segment_order_count")
          or result.HasField("response_candidate_id_count")
      ):
        raise ValueError("AJIMEE semantic backend error is inconsistent")
      successful_outputs.append(None)
      continue
    if result.outcome not in (
        ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_SUCCESS,
        ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_INVALID_RESPONSE,
    ):
      raise ValueError("AJIMEE semantic outcome is invalid")
    _validate_semantic_response_shape(result, request)
    if not result.response_token_matches_request:
      if (
          result.outcome
          != ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_INVALID_RESPONSE
          or result.canonical_status_code
          != ajimee_capacity_audit_pb2.AJIMEE_STATUS_ABORTED
          or result.HasField("merged_top_output")
      ):
        raise ValueError("AJIMEE semantic token mismatch is inconsistent")
      successful_outputs.append(None)
      continue
    if not result.response_within_request_bounds:
      if (
          result.outcome
          != ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_INVALID_RESPONSE
          or result.canonical_status_code
          != ajimee_capacity_audit_pb2.AJIMEE_STATUS_INVALID_ARGUMENT
          or result.HasField("merged_top_output")
      ):
        raise ValueError("AJIMEE out-of-bounds semantic response is inconsistent")
      successful_outputs.append(None)
      continue
    merge_status, complete_orders = _merge_response_orders(
        request, result.response_segment_orders
    )
    if result.outcome == ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_SUCCESS:
      if (
          result.canonical_status_code
          != ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
          or not result.HasField("merged_top_output")
          or merge_status != ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
      ):
        raise ValueError("AJIMEE successful semantic result is inconsistent")
      recomputed = _merged_top(request, complete_orders)
      _require_utf8(result.merged_top_output, "semantic merged top")
      if recomputed != result.merged_top_output:
        raise ValueError("AJIMEE semantic merged top is inconsistent")
      successful_outputs.append(recomputed)
    else:
      if (
          result.HasField("merged_top_output")
          or merge_status == ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
          or result.canonical_status_code != merge_status
      ):
        raise ValueError("AJIMEE invalid semantic response is inconsistent")
      successful_outputs.append(None)
  return tuple(successful_outputs)


def _stable_distinct_outputs(outputs: Iterable[str]) -> tuple[str, ...]:
  seen: set[bytes] = set()
  distinct: list[str] = []
  for output in outputs:
    encoded = _require_utf8(output, "accepted output")
    if encoded not in seen:
      seen.add(encoded)
      distinct.append(output)
  return tuple(distinct)


def _is_oracle_covered(
    request: frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest,
    accepted_outputs: Iterable[str],
) -> bool:
  segment_values = [
      tuple(
          _require_utf8(candidate.value, "candidate value")
          for candidate in segment.candidates
      )
      for segment in request.segments
  ]
  for accepted_output in accepted_outputs:
    target = _require_utf8(accepted_output, "accepted output")
    reachable = {0}
    for values in segment_values:
      reachable = {
          offset + len(value)
          for offset in reachable
          for value in values
          if target.startswith(value, offset)
      }
      if not reachable:
        break
    if len(target) in reachable:
      return True
  return False


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


def _character_error(hypothesis: str, references: tuple[str, ...]) -> tuple[int, int]:
  distances = tuple(_levenshtein(hypothesis, reference) for reference in references)
  minimum = min(distances)
  reference_ordinal = distances.index(minimum)
  return minimum, len(references[reference_ordinal])


def _exact_bootstrap_endpoints(
    win_count: int, regression_count: int, tie_count: int
) -> tuple[int, int]:
  sample_size = win_count + regression_count + tie_count
  if sample_size <= 0:
    raise ValueError("AJIMEE bootstrap sample is empty")
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
    raise ValueError("AJIMEE bootstrap quantiles are incomplete")
  return lower, upper


def _new_metric_accumulator() -> dict[str, int]:
  return {
      "case_count": 0,
      "semantic_success_count": 0,
      "backend_error_count": 0,
      "invalid_response_count": 0,
      "oracle_covered_count": 0,
      "baseline_correct_count": 0,
      "effective_model_correct_count": 0,
      "retained_correct_count": 0,
      "win_count": 0,
      "regression_count": 0,
      "unchanged_incorrect_count": 0,
      "baseline_edit_distance": 0,
      "baseline_reference_scalars": 0,
      "model_edit_distance": 0,
      "model_reference_scalars": 0,
  }


def _new_capacity_accumulator() -> dict[str, Any]:
  return {
      "case_count": 0,
      "pass_count": 0,
      "request_limit_count": 0,
      "audit_error_count": 0,
      "summed_usage": {name: 0 for name in _USAGE_FIELDS},
      "maximum_usage": {name: 0 for name in _USAGE_FIELDS},
      "cases_with_window_omission": 0,
      "cases_with_candidate_capacity_omission": 0,
      "cases_with_segment_capacity_omission": 0,
      "batch_histogram": Counter(),
      "limiter_cases": {},
      "limiter_segments": Counter(),
      "limiter_omitted": Counter(),
  }


def _validate_pinned_pre_model_facts(
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
    accumulators: dict[int, dict[str, int]],
) -> None:
  if (
      config.answer_corpus_sha256 != _CHECKED_ANSWER_CORPUS_SHA256
      or config.frozen_corpus_sha256 != _CHECKED_FROZEN_CORPUS_SHA256
  ):
    return
  for slice_value, expected in _PINNED_PRE_MODEL_FACTS.items():
    accumulator = accumulators[slice_value]
    observed = (
        accumulator["baseline_correct_count"],
        accumulator["oracle_covered_count"],
        accumulator["baseline_edit_distance"],
        accumulator["baseline_reference_scalars"],
    )
    if observed != expected:
      raise ValueError("AJIMEE pinned pre-model facts do not match")


def _set_ratio(
    ratio: exact_quality_metrics_pb2.ExactRatio,
    numerator: int,
    denominator: int,
) -> None:
  if denominator <= 0:
    raise ValueError("AJIMEE exact ratio denominator is not positive")
  ratio.numerator = numerator
  ratio.denominator = denominator


def _build_metric_slice(
    slice_value: int, accumulator: dict[str, int]
) -> ajimee_quality_report_pb2.AjimeeMetricSlice:
  result = ajimee_quality_report_pb2.AjimeeMetricSlice()
  result.slice = slice_value
  for name in (
      "case_count",
      "semantic_success_count",
      "backend_error_count",
      "invalid_response_count",
      "oracle_covered_count",
      "baseline_correct_count",
      "effective_model_correct_count",
      "retained_correct_count",
      "win_count",
      "regression_count",
      "unchanged_incorrect_count",
  ):
    setattr(result, name, accumulator[name])
  result.baseline_character_error.edit_distance_sum = accumulator[
      "baseline_edit_distance"
  ]
  result.baseline_character_error.reference_unicode_scalar_count = accumulator[
      "baseline_reference_scalars"
  ]
  result.effective_model_character_error.edit_distance_sum = accumulator[
      "model_edit_distance"
  ]
  result.effective_model_character_error.reference_unicode_scalar_count = (
      accumulator["model_reference_scalars"]
  )
  tie_count = (
      accumulator["case_count"]
      - accumulator["win_count"]
      - accumulator["regression_count"]
  )
  lower, upper = _exact_bootstrap_endpoints(
      accumulator["win_count"], accumulator["regression_count"], tie_count
  )
  bootstrap = result.paired_bootstrap
  bootstrap.definition_version = _EXACT_BOOTSTRAP_DEFINITION_VERSION
  bootstrap.sample_size = accumulator["case_count"]
  bootstrap.win_count = accumulator["win_count"]
  bootstrap.regression_count = accumulator["regression_count"]
  bootstrap.tie_count = tie_count
  bootstrap.lower_gain_cases = lower
  bootstrap.upper_gain_cases = upper
  bootstrap.gain_denominator_cases = accumulator["case_count"]
  _set_ratio(
      result.baseline_accuracy,
      accumulator["baseline_correct_count"],
      accumulator["case_count"],
  )
  _set_ratio(
      result.effective_model_accuracy,
      accumulator["effective_model_correct_count"],
      accumulator["case_count"],
  )
  _set_ratio(
      result.accuracy_gain,
      accumulator["win_count"] - accumulator["regression_count"],
      accumulator["case_count"],
  )
  _set_ratio(
      result.oracle_coverage,
      accumulator["oracle_covered_count"],
      accumulator["case_count"],
  )
  if accumulator["oracle_covered_count"]:
    _set_ratio(
        result.baseline_conditional_accuracy,
        accumulator["baseline_correct_count"],
        accumulator["oracle_covered_count"],
    )
    _set_ratio(
        result.model_conditional_accuracy,
        accumulator["effective_model_correct_count"],
        accumulator["oracle_covered_count"],
    )
  if accumulator["baseline_correct_count"]:
    _set_ratio(
        result.retention,
        accumulator["retained_correct_count"],
        accumulator["baseline_correct_count"],
    )
  return result


def _update_capacity_accumulator(
    accumulator: dict[str, Any],
    result: ajimee_capacity_audit_pb2.AjimeeCapacityAuditCaseResult,
    case_ordinal: int,
) -> None:
  accumulator["case_count"] += 1
  if result.outcome == ajimee_capacity_audit_pb2.AJIMEE_CAPACITY_AUDIT_PASS:
    accumulator["pass_count"] += 1
  elif (
      result.outcome
      == ajimee_capacity_audit_pb2.AJIMEE_CAPACITY_AUDIT_REQUEST_LIMIT_EXCEEDED
  ):
    accumulator["request_limit_count"] += 1
  else:
    accumulator["audit_error_count"] += 1
  if not result.HasField("usage"):
    return
  usage = result.usage
  for name in _USAGE_FIELDS:
    value = getattr(usage, name)
    accumulator["summed_usage"][name] += value
    accumulator["maximum_usage"][name] = max(
        accumulator["maximum_usage"][name], value
    )
  accumulator["cases_with_window_omission"] += (
      usage.candidates_omitted_by_window > 0
  )
  accumulator["cases_with_candidate_capacity_omission"] += (
      usage.candidates_omitted_by_capacity > 0
  )
  accumulator["cases_with_segment_capacity_omission"] += (
      usage.segments_omitted_by_capacity > 0
  )
  accumulator["batch_histogram"][usage.decode_batch_count] += 1
  for diagnostic in result.segments:
    if diagnostic.HasField("first_limiter"):
      limit = diagnostic.first_limiter.limit
      accumulator["limiter_cases"].setdefault(limit, set()).add(case_ordinal)
      accumulator["limiter_segments"][limit] += 1
      accumulator["limiter_omitted"][limit] += diagnostic.omitted_by_capacity


def _build_capacity_slice(
    slice_value: int, accumulator: dict[str, Any]
) -> ajimee_quality_report_pb2.AjimeeCapacityCoverageSlice:
  result = ajimee_quality_report_pb2.AjimeeCapacityCoverageSlice()
  result.slice = slice_value
  for name in (
      "case_count",
      "pass_count",
      "request_limit_count",
      "audit_error_count",
      "cases_with_window_omission",
      "cases_with_candidate_capacity_omission",
      "cases_with_segment_capacity_omission",
  ):
    setattr(result, name, accumulator[name])
  for name in _USAGE_FIELDS:
    setattr(result.summed_usage, name, accumulator["summed_usage"][name])
    setattr(result.maximum_usage, name, accumulator["maximum_usage"][name])
  for batch_count in sorted(accumulator["batch_histogram"]):
    entry = result.decode_batch_histogram.add()
    entry.batch_count = batch_count
    entry.case_count = accumulator["batch_histogram"][batch_count]
  for limit in sorted(accumulator["limiter_cases"]):
    entry = result.limiter_summaries.add()
    entry.limit = limit
    entry.case_count = len(accumulator["limiter_cases"][limit])
    entry.segment_count = accumulator["limiter_segments"][limit]
    entry.omitted_candidate_count = accumulator["limiter_omitted"][limit]
  return result


def _ratio_tuple(ratio: exact_quality_metrics_pb2.ExactRatio) -> tuple[int, int]:
  return ratio.numerator, ratio.denominator


def _validate_quality_report(
    report: ajimee_quality_report_pb2.AjimeeQualityReport,
    config: ajimee_quality_report_pb2.AjimeeQualityReportConfig,
    report_config_sha256: str,
    capacity_config: ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
) -> None:
  _require_initialized_without_unknown_fields(report, "AJIMEE quality report")
  if report.schema_version != config.quality_report_schema_version:
    raise ValueError("AJIMEE quality report schema is invalid")
  identity = report.identity
  if (
      identity.report_config_sha256 != report_config_sha256
      or identity.input_corpus_sha256 != config.input_corpus_sha256
      or identity.answer_corpus_sha256 != config.answer_corpus_sha256
      or identity.frozen_corpus_sha256 != config.frozen_corpus_sha256
      or identity.capacity_audit_config_sha256
      != config.capacity_audit_config_sha256
      or identity.capacity_audit_sha256 != config.capacity_audit_sha256
      or identity.semantic_results_sha256 != config.semantic_results_sha256
      or identity.source != config.expected_source
      or _capacity_identity_tuple(identity.capacity_audit_identity)
      != _capacity_identity_from_config(capacity_config)
      or identity.numeric_profile
      != ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_CONFIGURED
      or identity.metric_definition_version != config.metric_definition_version
      or identity.bootstrap_definition_version
      != config.exact_bootstrap_definition_version
  ):
    raise ValueError("AJIMEE quality report identity is invalid")
  if (
      len(report.metric_slices) != len(_SLICE_ORDER)
      or len(report.capacity_coverage_slices) != len(_SLICE_ORDER)
  ):
    raise ValueError("AJIMEE quality report slices are incomplete")
  expected_counts = (
      config.expected_case_count,
      config.expected_context_case_count,
      config.expected_no_context_case_count,
  )
  for metric, slice_value, expected_count in zip(
      report.metric_slices, _SLICE_ORDER, expected_counts, strict=True
  ):
    if metric.slice != slice_value or metric.case_count != expected_count:
      raise ValueError("AJIMEE metric slice order or count is invalid")
    if (
        metric.semantic_success_count
        + metric.backend_error_count
        + metric.invalid_response_count
        != metric.case_count
        or metric.retained_correct_count
        + metric.win_count
        + metric.regression_count
        + metric.unchanged_incorrect_count
        != metric.case_count
        or metric.baseline_correct_count
        != metric.retained_correct_count + metric.regression_count
        or metric.effective_model_correct_count
        != metric.retained_correct_count + metric.win_count
        or metric.baseline_correct_count > metric.oracle_covered_count
        or metric.effective_model_correct_count > metric.oracle_covered_count
    ):
      raise ValueError("AJIMEE metric count algebra is invalid")
    expected_ratios = (
        (metric.baseline_accuracy, metric.baseline_correct_count, metric.case_count),
        (
            metric.effective_model_accuracy,
            metric.effective_model_correct_count,
            metric.case_count,
        ),
        (
            metric.accuracy_gain,
            metric.win_count - metric.regression_count,
            metric.case_count,
        ),
        (metric.oracle_coverage, metric.oracle_covered_count, metric.case_count),
    )
    for ratio, numerator, denominator in expected_ratios:
      if _ratio_tuple(ratio) != (numerator, denominator):
        raise ValueError("AJIMEE exact metric ratio is inconsistent")
    if metric.oracle_covered_count:
      if (
          not metric.HasField("baseline_conditional_accuracy")
          or not metric.HasField("model_conditional_accuracy")
          or _ratio_tuple(metric.baseline_conditional_accuracy)
          != (metric.baseline_correct_count, metric.oracle_covered_count)
          or _ratio_tuple(metric.model_conditional_accuracy)
          != (metric.effective_model_correct_count, metric.oracle_covered_count)
      ):
        raise ValueError("AJIMEE conditional accuracy is inconsistent")
    elif metric.HasField("baseline_conditional_accuracy") or metric.HasField(
        "model_conditional_accuracy"
    ):
      raise ValueError("AJIMEE empty conditional accuracy is present")
    if metric.baseline_correct_count:
      if (
          not metric.HasField("retention")
          or _ratio_tuple(metric.retention)
          != (metric.retained_correct_count, metric.baseline_correct_count)
      ):
        raise ValueError("AJIMEE retention is inconsistent")
    elif metric.HasField("retention"):
      raise ValueError("AJIMEE empty retention is present")
    bootstrap = metric.paired_bootstrap
    tie_count = metric.case_count - metric.win_count - metric.regression_count
    lower, upper = _exact_bootstrap_endpoints(
        metric.win_count, metric.regression_count, tie_count
    )
    if (
        bootstrap.definition_version != _EXACT_BOOTSTRAP_DEFINITION_VERSION
        or bootstrap.sample_size != metric.case_count
        or bootstrap.win_count != metric.win_count
        or bootstrap.regression_count != metric.regression_count
        or bootstrap.tie_count != tie_count
        or bootstrap.lower_gain_cases != lower
        or bootstrap.upper_gain_cases != upper
        or bootstrap.gain_denominator_cases != metric.case_count
        or metric.baseline_character_error.reference_unicode_scalar_count == 0
        or metric.effective_model_character_error.reference_unicode_scalar_count
        == 0
    ):
      raise ValueError("AJIMEE exact bootstrap or character error is invalid")
  for coverage, slice_value, expected_count in zip(
      report.capacity_coverage_slices, _SLICE_ORDER, expected_counts, strict=True
  ):
    if (
        coverage.slice != slice_value
        or coverage.case_count != expected_count
        or coverage.pass_count != coverage.case_count
        or coverage.pass_count
        + coverage.request_limit_count
        + coverage.audit_error_count
        != coverage.case_count
        or coverage.cases_with_window_omission > coverage.case_count
        or coverage.cases_with_candidate_capacity_omission
        > coverage.case_count
        or coverage.cases_with_segment_capacity_omission > coverage.case_count
        or sum(entry.case_count for entry in coverage.decode_batch_histogram)
        != coverage.case_count
        or any(
            entry.case_count == 0
            for entry in coverage.decode_batch_histogram
        )
        or [entry.batch_count for entry in coverage.decode_batch_histogram]
        != sorted({entry.batch_count for entry in coverage.decode_batch_histogram})
        or [entry.limit for entry in coverage.limiter_summaries]
        != sorted({entry.limit for entry in coverage.limiter_summaries})
        or any(
            entry.limit not in _KNOWN_CAPACITY_LIMITS
            or entry.case_count == 0
            or entry.case_count > entry.segment_count
            or entry.segment_count > entry.omitted_candidate_count
            for entry in coverage.limiter_summaries
        )
        or sum(
            entry.omitted_candidate_count
            for entry in coverage.limiter_summaries
        )
        != coverage.summed_usage.candidates_omitted_by_capacity
        or any(
            getattr(coverage.maximum_usage, name)
            > getattr(coverage.summed_usage, name)
            for name in _USAGE_FIELDS
        )
    ):
      raise ValueError("AJIMEE capacity coverage slice is invalid")
  all_success = (
      report.metric_slices[0].semantic_success_count
      == report.metric_slices[0].case_count
  )
  if report.semantic_all_success != all_success:
    raise ValueError("AJIMEE semantic all-success flag is inconsistent")


def build_quality_report(
    inputs: AjimeeQualityReportInputs,
) -> ajimee_quality_report_pb2.AjimeeQualityReport:
  report_config_sha256 = _sha256(inputs.report_config_textproto)
  config = _parse_textproto(
      ajimee_quality_report_pb2.AjimeeQualityReportConfig,
      inputs.report_config_textproto,
      "AJIMEE quality report config",
  )
  _validate_quality_config(config)
  actual_hashes = {
      "answer_corpus_sha256": _sha256(inputs.answer_corpus_binary),
      "frozen_corpus_sha256": _sha256(inputs.frozen_corpus_binary),
      "capacity_audit_config_sha256": _sha256(
          inputs.capacity_audit_config_textproto
      ),
      "capacity_audit_sha256": _sha256(inputs.capacity_audit_binary),
      "semantic_results_sha256": _sha256(inputs.semantic_results_binary),
  }

  if (
      actual_hashes["capacity_audit_config_sha256"]
      != config.capacity_audit_config_sha256
  ):
    raise ValueError("AJIMEE capacity audit config SHA256 does not match")
  capacity_config = _parse_textproto(
      ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig,
      inputs.capacity_audit_config_textproto,
      "AJIMEE capacity audit config",
  )
  _validate_capacity_config(capacity_config)
  if (
      capacity_config.frozen_corpus_sha256 != config.frozen_corpus_sha256
      or capacity_config.expected_case_count != config.expected_case_count
  ):
    raise ValueError("AJIMEE capacity config does not match reporter config")

  if actual_hashes["frozen_corpus_sha256"] != config.frozen_corpus_sha256:
    raise ValueError("AJIMEE frozen corpus SHA256 does not match")
  corpus = _parse_binary(
      ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
      inputs.frozen_corpus_binary,
      "AJIMEE frozen corpus",
  )
  _validate_frozen_corpus(corpus, config)

  if actual_hashes["answer_corpus_sha256"] != config.answer_corpus_sha256:
    raise ValueError("AJIMEE answer corpus SHA256 does not match")
  answers = _parse_binary(
      ajimee_corpus_pb2.AjimeeAnswerCorpus,
      inputs.answer_corpus_binary,
      "AJIMEE answer corpus",
  )
  _validate_answers(answers, corpus, config)

  if actual_hashes["capacity_audit_sha256"] != config.capacity_audit_sha256:
    raise ValueError("AJIMEE capacity audit SHA256 does not match")
  capacity_audit = _parse_binary(
      ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
      inputs.capacity_audit_binary,
      "AJIMEE capacity audit",
  )
  _validate_capacity_audit(capacity_audit, capacity_config, corpus)

  if (
      actual_hashes["semantic_results_sha256"]
      != config.semantic_results_sha256
  ):
    raise ValueError("AJIMEE semantic results SHA256 does not match")
  semantic_results = _parse_binary(
      ajimee_semantic_results_pb2.AjimeeSemanticResults,
      inputs.semantic_results_binary,
      "AJIMEE semantic results",
  )
  successful_outputs = _validate_semantic_results(
      semantic_results, corpus, capacity_config, capacity_audit, config
  )

  metric_accumulators = {
      slice_value: _new_metric_accumulator() for slice_value in _SLICE_ORDER
  }
  capacity_accumulators = {
      slice_value: _new_capacity_accumulator() for slice_value in _SLICE_ORDER
  }
  joined_cases = zip(
      corpus.cases,
      answers.cases,
      capacity_audit.cases,
      semantic_results.cases,
      successful_outputs,
      strict=True,
  )
  for case_ordinal, (
      frozen_case,
      answer,
      audit_result,
      semantic_result,
      output,
  ) in enumerate(joined_cases):
    case_slice = (
        ajimee_quality_report_pb2.AJIMEE_REPORT_HAS_CONTEXT
        if frozen_case.request.preceding_text
        else ajimee_quality_report_pb2.AJIMEE_REPORT_NO_CONTEXT
    )
    accepted = _stable_distinct_outputs(answer.accepted_whole_outputs)
    baseline = frozen_case.baseline_output
    effective_model = output if output is not None else baseline
    baseline_correct = baseline in accepted
    model_correct = effective_model in accepted
    oracle_covered = _is_oracle_covered(frozen_case.request, accepted)
    if (baseline_correct or model_correct) and not oracle_covered:
      raise ValueError("AJIMEE correct output is not Mozc-oracle covered")
    baseline_distance, baseline_reference_length = _character_error(
        baseline, accepted
    )
    model_distance, model_reference_length = _character_error(
        effective_model, accepted
    )
    for slice_value in (
        ajimee_quality_report_pb2.AJIMEE_REPORT_OVERALL,
        case_slice,
    ):
      metric = metric_accumulators[slice_value]
      metric["case_count"] += 1
      if semantic_result.outcome == ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_SUCCESS:
        metric["semantic_success_count"] += 1
      elif (
          semantic_result.outcome
          == ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_BACKEND_ERROR
      ):
        metric["backend_error_count"] += 1
      else:
        metric["invalid_response_count"] += 1
      metric["oracle_covered_count"] += oracle_covered
      metric["baseline_correct_count"] += baseline_correct
      metric["effective_model_correct_count"] += model_correct
      if baseline_correct and model_correct:
        metric["retained_correct_count"] += 1
      elif not baseline_correct and model_correct:
        metric["win_count"] += 1
      elif baseline_correct and not model_correct:
        metric["regression_count"] += 1
      else:
        metric["unchanged_incorrect_count"] += 1
      metric["baseline_edit_distance"] += baseline_distance
      metric["baseline_reference_scalars"] += baseline_reference_length
      metric["model_edit_distance"] += model_distance
      metric["model_reference_scalars"] += model_reference_length
      _update_capacity_accumulator(
          capacity_accumulators[slice_value], audit_result, case_ordinal
      )

  if (
      metric_accumulators[ajimee_quality_report_pb2.AJIMEE_REPORT_HAS_CONTEXT][
          "case_count"
      ]
      != config.expected_context_case_count
      or metric_accumulators[
          ajimee_quality_report_pb2.AJIMEE_REPORT_NO_CONTEXT
      ]["case_count"]
      != config.expected_no_context_case_count
  ):
    raise ValueError("AJIMEE context slice counts do not match")
  _validate_pinned_pre_model_facts(config, metric_accumulators)

  report = ajimee_quality_report_pb2.AjimeeQualityReport()
  report.schema_version = config.quality_report_schema_version
  identity = report.identity
  identity.report_config_sha256 = report_config_sha256
  identity.input_corpus_sha256 = config.input_corpus_sha256
  identity.answer_corpus_sha256 = config.answer_corpus_sha256
  identity.frozen_corpus_sha256 = config.frozen_corpus_sha256
  identity.capacity_audit_config_sha256 = config.capacity_audit_config_sha256
  identity.capacity_audit_sha256 = config.capacity_audit_sha256
  identity.semantic_results_sha256 = config.semantic_results_sha256
  identity.source.CopyFrom(config.expected_source)
  identity.capacity_audit_identity.CopyFrom(capacity_audit.identity)
  identity.numeric_profile = semantic_results.identity.numeric_profile
  identity.metric_definition_version = config.metric_definition_version
  identity.bootstrap_definition_version = config.exact_bootstrap_definition_version
  report.semantic_all_success = all(
      result.outcome == ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_SUCCESS
      for result in semantic_results.cases
  )
  for slice_value in _SLICE_ORDER:
    report.metric_slices.add().CopyFrom(
        _build_metric_slice(slice_value, metric_accumulators[slice_value])
    )
    report.capacity_coverage_slices.add().CopyFrom(
        _build_capacity_slice(slice_value, capacity_accumulators[slice_value])
    )
  _validate_quality_report(
      report,
      config,
      report_config_sha256,
      capacity_config,
  )
  return report


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
    raise ValueError("AJIMEE quality report JSON cannot contain bytes")
  if field.type in (
      FieldDescriptor.TYPE_DOUBLE,
      FieldDescriptor.TYPE_FLOAT,
  ):
    raise ValueError("AJIMEE quality report JSON cannot contain floating point")
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


def serialize_quality_report(
    report: ajimee_quality_report_pb2.AjimeeQualityReport,
) -> AjimeeSerializedQualityReport:
  _require_initialized_without_unknown_fields(report, "AJIMEE quality report")
  binary = report.SerializeToString(deterministic=True)
  canonical_json = (
      json.dumps(
          _message_to_tag_ordered_json(report),
          ensure_ascii=False,
          separators=(",", ":"),
      )
      + "\n"
  ).encode("utf-8")
  return AjimeeSerializedQualityReport(binary=binary, canonical_json=canonical_json)
