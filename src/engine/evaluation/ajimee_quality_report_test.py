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

import collections
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from google.protobuf import text_format

from engine.evaluation import ajimee_capacity_audit_pb2
from engine.evaluation import ajimee_corpus_pb2
from engine.evaluation import ajimee_frozen_corpus_pb2
from engine.evaluation import ajimee_quality_report
from engine.evaluation import ajimee_quality_report_pb2
from engine.evaluation import ajimee_semantic_results_pb2
from engine.evaluation import frozen_candidate_ranker_pb2
from engine.evaluation import report_ajimee_quality_main


_HEX_40_A = "a" * 40
_HEX_40_B = "b" * 40
_HEX_40_C = "c" * 40
_HEX_64_1 = "1" * 64
_HEX_64_2 = "2" * 64
_HEX_64_3 = "3" * 64
_HEX_64_4 = "4" * 64
_HEX_64_5 = "5" * 64
_CHECKED_REPORT_CONFIG = Path(__file__).with_name(
    "ajimee_quality_report_config.textproto"
)
_CHECKED_CAPACITY_CONFIG = Path(__file__).with_name(
    "ajimee_capacity_audit_config.textproto"
)
_CHECKED_FREEZER_CONFIG = Path(__file__).with_name(
    "ajimee_freezer_config.textproto"
)


def _binary(message) -> bytes:
  return message.SerializeToString(deterministic=True)


def _textproto(message) -> bytes:
  return text_format.MessageToString(
      message, as_utf8=True, use_index_order=True
  ).encode("utf-8")


def _copy_source(destination) -> None:
  destination.benchmark_name = "AJIMEE synthetic"
  destination.source_revision = _HEX_40_A
  destination.source_relative_path = "synthetic.json"
  destination.source_sha256 = _HEX_64_1
  destination.creator = "Synthetic creator"
  destination.source_url = "https://example.test/source"
  destination.license_identifier = "CC-BY-SA-3.0"
  destination.license_url = "https://example.test/license"
  destination.upstream_dataset_url = "https://example.test/upstream"
  destination.changes_notice = "Synthetic transformation."


def _request_string_bytes(request) -> int:
  values = [request.preceding_text, request.following_text, request.reading]
  for segment in request.segments:
    values.append(segment.key)
    for candidate in segment.candidates:
      values.extend((candidate.key, candidate.value))
  return sum(len(value.encode("utf-8")) for value in values)


def _make_corpora():
  frozen = ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus()
  frozen.schema_version = 2
  _copy_source(frozen.source)
  frozen.input_corpus_sha256 = _HEX_64_2
  frozen.mozc.source_revision = _HEX_40_B
  frozen.mozc.data_type = "oss"
  frozen.mozc.data_sha256 = _HEX_64_3
  frozen.mozc.default_desktop_request_sha256 = _HEX_64_4
  frozen.mozc.default_desktop_config_sha256 = _HEX_64_5
  frozen.mozc.evaluation_clock_utc_rfc3339 = "2000-01-01T00:00:00Z"

  answers = ajimee_corpus_pb2.AjimeeAnswerCorpus()
  answers.schema_version = 1
  _copy_source(answers.source)
  for ordinal in range(200):
    baseline_correct = ordinal in (0, 1, 3)
    baseline_value = "正" if baseline_correct else "誤"
    alternate_value = "誤" if baseline_correct else "正"
    frozen_case = frozen.cases.add()
    frozen_case.source_index = ordinal + 1
    frozen_case.normalized_hiragana_reading = "よみ"
    frozen_case.history_reconstructed = False
    frozen_case.baseline_output = baseline_value
    request = frozen_case.request
    request.token.session_generation = 0
    request.token.state_revision = 0
    request.token.request_sequence = ordinal + 1
    request.mode = (
        frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest.MODE_CONVERSION
    )
    request.preceding_text = "文脈" if ordinal < 100 else ""
    request.following_text = ""
    request.reading = "よみ"
    request.focused_segment_id = 0
    for segment_id, values in enumerate(
        (
            (baseline_value, alternate_value, "他"),
            ("", "", ""),
        )
    ):
      segment = request.segments.add()
      segment.id = segment_id
      segment.key = "よみ"
      for segment_candidate_id, value in enumerate(values):
        candidate_id = 3 * segment_id + segment_candidate_id
        candidate = segment.candidates.add()
        candidate.id = candidate_id
        candidate.key = "よみ"
        candidate.value = value
        candidate.cost = candidate_id
        candidate.attributes = 0
        candidate.consumed_key_size = 0
        candidate.is_protected = False

    answer = answers.cases.add()
    answer.source_index = ordinal + 1
    answer.accepted_whole_outputs.append("正")
    if ordinal == 4:
      answer.accepted_whole_outputs.append("正")
  return frozen, answers


def _make_capacity_config(frozen_binary: bytes):
  config = ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig()
  config.schema_version = 1
  config.frozen_corpus_schema_version = 2
  config.capacity_audit_schema_version = 1
  config.frozen_corpus_sha256 = hashlib.sha256(frozen_binary).hexdigest()
  config.expected_case_count = 200
  config.model_manifest_schema_version = 4
  config.model_manifest_sha256 = _HEX_64_1
  config.gguf_file_name = "synthetic.gguf"
  config.gguf_sha256 = _HEX_64_2
  config.tokenizer_sha256 = _HEX_64_3
  config.quantization = "Q4_K_M"
  config.source_revision = _HEX_40_B
  config.runtime_revision = _HEX_40_C
  return config


def _copy_capacity_identity(config, destination) -> None:
  destination.frozen_corpus_sha256 = config.frozen_corpus_sha256
  destination.model_manifest_sha256 = config.model_manifest_sha256
  destination.gguf_file_name = config.gguf_file_name
  destination.gguf_sha256 = config.gguf_sha256
  destination.tokenizer_sha256 = config.tokenizer_sha256
  destination.quantization = config.quantization
  destination.source_revision = config.source_revision
  destination.runtime_revision = config.runtime_revision


def _make_capacity_audit(frozen, config):
  audit = ajimee_capacity_audit_pb2.AjimeeCapacityAudit()
  audit.schema_version = 1
  _copy_capacity_identity(config, audit.identity)
  audit.all_passed = True
  for ordinal, frozen_case in enumerate(frozen.cases):
    result = audit.cases.add()
    result.source_index = frozen_case.source_index
    result.outcome = ajimee_capacity_audit_pb2.AJIMEE_CAPACITY_AUDIT_PASS
    result.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
    selected_counts = []
    window_candidate_count = 0
    omitted_by_window = 0
    omitted_by_capacity = 0
    omitted_segments = 0
    for segment_id in range(2):
      window_count = 2 if ordinal == 6 and segment_id == 0 else 3
      if ordinal == 5 and segment_id == 0:
        selected_count = 0
        disposition = (
            ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_OMITTED_CAPACITY
        )
        limit = ajimee_capacity_audit_pb2.AJIMEE_SEQUENCES_PER_DECODE
        observed = 4
        allowed = 3
        omitted_segments += 1
      elif ordinal == 4 and segment_id == 0:
        selected_count = 2
        disposition = ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_SELECTED
        limit = ajimee_capacity_audit_pb2.AJIMEE_OUTPUT_ROWS_PER_DECODE
        observed = 4
        allowed = 3
      elif ordinal == 104 and segment_id == 1:
        selected_count = 2
        disposition = ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_SELECTED
        limit = ajimee_capacity_audit_pb2.AJIMEE_FULL_RECORD_TOKENS
        observed = 3
        allowed = 2
      else:
        selected_count = window_count
        disposition = ajimee_capacity_audit_pb2.AJIMEE_SEGMENT_CAPACITY_SELECTED
        limit = None
        observed = 0
        allowed = 0
      diagnostic = result.segments.add()
      diagnostic.segment_id = segment_id
      diagnostic.unprotected_candidate_count = 3
      diagnostic.window_candidate_count = window_count
      diagnostic.selected_candidate_count = selected_count
      diagnostic.omitted_by_window = 3 - window_count
      diagnostic.omitted_by_capacity = window_count - selected_count
      diagnostic.disposition = disposition
      if limit is not None:
        diagnostic.first_limiter.limit = limit
        diagnostic.first_limiter.observed = observed
        diagnostic.first_limiter.allowed = allowed
      window_candidate_count += window_count
      omitted_by_window += 3 - window_count
      omitted_by_capacity += window_count - selected_count
      if selected_count:
        selected_counts.append(selected_count)

    decode_batch_count = 2 if ordinal % 2 == 0 else 1
    max_sequences = (
        max(selected_counts)
        if decode_batch_count == 2
        else sum(selected_counts)
    )
    usage = result.usage
    usage.request_segment_count = 2
    usage.request_candidate_count = 6
    usage.request_unprotected_candidate_count = 6
    usage.request_rankable_segment_count = 2
    usage.request_string_bytes = _request_string_bytes(frozen_case.request)
    usage.max_unprotected_candidates_in_segment = 3
    usage.window_candidate_count = window_candidate_count
    usage.candidates_omitted_by_window = omitted_by_window
    usage.selected_segment_count = len(selected_counts)
    usage.selected_candidate_count = sum(selected_counts)
    usage.candidates_omitted_by_capacity = omitted_by_capacity
    usage.segments_omitted_by_capacity = omitted_segments
    usage.decode_batch_count = decode_batch_count
    usage.max_selected_serialized_record_bytes = 10 + ordinal % 3
    usage.max_selected_normalized_record_bytes = 20 + ordinal % 5
    usage.max_selected_full_record_tokens = 2
    usage.max_segments_in_decode = (
        1 if decode_batch_count == 2 else len(selected_counts)
    )
    usage.max_sequences_in_decode = max_sequences
    usage.max_input_nodes_in_decode = max_sequences + 1
    usage.max_output_rows_in_decode = max_sequences
    usage.max_reserved_output_rows_in_decode = max_sequences + 1
    usage.max_output_logit_bytes_in_decode = max_sequences * 4
  return audit


def _make_semantic_results(
    frozen, capacity_config, capacity_config_bytes, audit, audit_bytes
):
  results = ajimee_semantic_results_pb2.AjimeeSemanticResults()
  results.schema_version = 1
  results.identity.frozen_corpus_sha256 = capacity_config.frozen_corpus_sha256
  results.identity.capacity_audit_config_sha256 = hashlib.sha256(
      capacity_config_bytes
  ).hexdigest()
  results.identity.capacity_audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
  results.identity.capacity_audit_identity.CopyFrom(audit.identity)
  results.identity.numeric_profile = (
      ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_CONFIGURED
  )
  for ordinal, frozen_case in enumerate(frozen.cases):
    result = results.cases.add()
    result.source_index = frozen_case.source_index
    if ordinal == 2:
      result.outcome = ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_BACKEND_ERROR
      result.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_INTERNAL
      continue
    if ordinal == 3:
      result.outcome = ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_INVALID_RESPONSE
      result.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_NOT_FOUND
      result.response_token_matches_request = True
      result.response_within_request_bounds = True
      result.response_segment_order_count = 1
      result.response_candidate_id_count = 1
      order = result.response_segment_orders.add()
      order.segment_id = 0
      order.candidate_ids.append(99)
      continue
    result.outcome = ajimee_semantic_results_pb2.AJIMEE_SEMANTIC_SUCCESS
    result.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK
    result.response_token_matches_request = True
    result.response_within_request_bounds = True
    result.response_segment_order_count = 1
    order = result.response_segment_orders.add()
    order.segment_id = 0
    if ordinal == 0:
      order.candidate_ids.extend((0, 1, 2))
      result.response_candidate_id_count = 3
      result.merged_top_output = "正"
    elif ordinal == 1:
      order.candidate_ids.extend((1, 0, 2))
      result.response_candidate_id_count = 3
      result.merged_top_output = "誤"
    else:
      order.candidate_ids.extend((1, 0))
      result.response_candidate_id_count = 2
      result.merged_top_output = "正"
  return results


def _make_inputs():
  frozen, answers = _make_corpora()
  frozen_bytes = _binary(frozen)
  answers_bytes = _binary(answers)
  capacity_config = _make_capacity_config(frozen_bytes)
  capacity_config_bytes = _textproto(capacity_config)
  audit = _make_capacity_audit(frozen, capacity_config)
  audit_bytes = _binary(audit)
  semantics = _make_semantic_results(
      frozen, capacity_config, capacity_config_bytes, audit, audit_bytes
  )
  semantic_bytes = _binary(semantics)
  report_config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  report_config.schema_version = 1
  report_config.answer_corpus_schema_version = 1
  report_config.frozen_corpus_schema_version = 2
  report_config.capacity_audit_schema_version = 1
  report_config.semantic_results_schema_version = 1
  report_config.quality_report_schema_version = 1
  report_config.input_corpus_sha256 = frozen.input_corpus_sha256
  report_config.answer_corpus_sha256 = hashlib.sha256(answers_bytes).hexdigest()
  report_config.frozen_corpus_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
  report_config.capacity_audit_config_sha256 = hashlib.sha256(
      capacity_config_bytes
  ).hexdigest()
  report_config.capacity_audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
  report_config.semantic_results_sha256 = hashlib.sha256(
      semantic_bytes
  ).hexdigest()
  report_config.expected_case_count = 200
  report_config.expected_context_case_count = 100
  report_config.expected_no_context_case_count = 100
  report_config.expected_source.CopyFrom(frozen.source)
  report_config.metric_definition_version = 1
  report_config.exact_bootstrap_definition_version = 1
  return ajimee_quality_report.AjimeeQualityReportInputs(
      report_config_textproto=_textproto(report_config),
      capacity_audit_config_textproto=capacity_config_bytes,
      answer_corpus_binary=answers_bytes,
      frozen_corpus_binary=frozen_bytes,
      capacity_audit_binary=audit_bytes,
      semantic_results_binary=semantic_bytes,
  )


def _with_semantics(inputs, semantics):
  semantic_bytes = _binary(semantics)
  return _with_semantic_bytes(inputs, semantic_bytes)


def _with_semantic_bytes(inputs, semantic_bytes):
  config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  text_format.Parse(inputs.report_config_textproto.decode("utf-8"), config)
  config.semantic_results_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
  return dataclasses.replace(
      inputs,
      report_config_textproto=_textproto(config),
      semantic_results_binary=semantic_bytes,
  )


def _with_answer_bytes(inputs, answer_bytes):
  config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  text_format.Parse(inputs.report_config_textproto.decode("utf-8"), config)
  config.answer_corpus_sha256 = hashlib.sha256(answer_bytes).hexdigest()
  return dataclasses.replace(
      inputs,
      report_config_textproto=_textproto(config),
      answer_corpus_binary=answer_bytes,
  )


def _with_capacity_audit_bytes(inputs, capacity_audit_bytes):
  config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  text_format.Parse(inputs.report_config_textproto.decode("utf-8"), config)
  config.capacity_audit_sha256 = hashlib.sha256(
      capacity_audit_bytes
  ).hexdigest()
  return dataclasses.replace(
      inputs,
      report_config_textproto=_textproto(config),
      capacity_audit_binary=capacity_audit_bytes,
  )


def _with_capacity_config(inputs, capacity_config):
  capacity_config_bytes = _textproto(capacity_config)
  config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  text_format.Parse(inputs.report_config_textproto.decode("utf-8"), config)
  config.capacity_audit_config_sha256 = hashlib.sha256(
      capacity_config_bytes
  ).hexdigest()
  return dataclasses.replace(
      inputs,
      report_config_textproto=_textproto(config),
      capacity_audit_config_textproto=capacity_config_bytes,
  )


def _parse_binary(message_type, data):
  message = message_type()
  message.ParseFromString(data)
  return message


def _parse_capacity_config(inputs):
  config = ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig()
  text_format.Parse(inputs.capacity_audit_config_textproto.decode("utf-8"), config)
  return config


def _swap_messages(repeated, first_index, second_index) -> None:
  first = repeated[first_index].__class__()
  first.CopyFrom(repeated[first_index])
  repeated[first_index].CopyFrom(repeated[second_index])
  repeated[second_index].CopyFrom(first)


def _rebuild_joined_inputs(
    inputs,
    *,
    capacity_config=None,
    frozen=None,
    answers=None,
    capacity_audit=None,
    semantics=None,
):
  if capacity_config is None:
    capacity_config = _parse_capacity_config(inputs)
  if frozen is None:
    frozen = _parse_binary(
        ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
        inputs.frozen_corpus_binary,
    )
  if answers is None:
    answers = _parse_binary(
        ajimee_corpus_pb2.AjimeeAnswerCorpus,
        inputs.answer_corpus_binary,
    )
  if capacity_audit is None:
    capacity_audit = _parse_binary(
        ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
        inputs.capacity_audit_binary,
    )
  if semantics is None:
    semantics = _parse_binary(
        ajimee_semantic_results_pb2.AjimeeSemanticResults,
        inputs.semantic_results_binary,
    )

  frozen_bytes = _binary(frozen)
  capacity_config.frozen_corpus_sha256 = hashlib.sha256(
      frozen_bytes
  ).hexdigest()
  capacity_config_bytes = _textproto(capacity_config)
  _copy_capacity_identity(capacity_config, capacity_audit.identity)
  capacity_audit_bytes = _binary(capacity_audit)
  semantics.identity.frozen_corpus_sha256 = capacity_config.frozen_corpus_sha256
  semantics.identity.capacity_audit_config_sha256 = hashlib.sha256(
      capacity_config_bytes
  ).hexdigest()
  semantics.identity.capacity_audit_sha256 = hashlib.sha256(
      capacity_audit_bytes
  ).hexdigest()
  semantics.identity.capacity_audit_identity.CopyFrom(capacity_audit.identity)
  semantic_bytes = _binary(semantics)
  answer_bytes = _binary(answers)

  report_config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
  text_format.Parse(
      inputs.report_config_textproto.decode("utf-8"), report_config
  )
  report_config.answer_corpus_sha256 = hashlib.sha256(answer_bytes).hexdigest()
  report_config.frozen_corpus_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
  report_config.capacity_audit_config_sha256 = hashlib.sha256(
      capacity_config_bytes
  ).hexdigest()
  report_config.capacity_audit_sha256 = hashlib.sha256(
      capacity_audit_bytes
  ).hexdigest()
  report_config.semantic_results_sha256 = hashlib.sha256(
      semantic_bytes
  ).hexdigest()
  return ajimee_quality_report.AjimeeQualityReportInputs(
      report_config_textproto=_textproto(report_config),
      capacity_audit_config_textproto=capacity_config_bytes,
      answer_corpus_binary=answer_bytes,
      frozen_corpus_binary=frozen_bytes,
      capacity_audit_binary=capacity_audit_bytes,
      semantic_results_binary=semantic_bytes,
  )


class AjimeeQualityReportTest(unittest.TestCase):

  def test_shared_proto_refactor_preserves_binary_fixtures(self) -> None:
    inputs = _make_inputs()
    report = ajimee_quality_report.serialize_quality_report(
        ajimee_quality_report.build_quality_report(inputs)
    )
    actual = {
        "answer": hashlib.sha256(inputs.answer_corpus_binary).hexdigest(),
        "frozen": hashlib.sha256(inputs.frozen_corpus_binary).hexdigest(),
        "capacity": hashlib.sha256(inputs.capacity_audit_binary).hexdigest(),
        "semantic": hashlib.sha256(inputs.semantic_results_binary).hexdigest(),
        "report": hashlib.sha256(report.binary).hexdigest(),
    }
    self.assertEqual(
        actual,
        {
            "answer": (
                "f7ae9f811a3a653f94e25fc490421c8b7fa2a73abf2e1bba5d979c1b31f5b41a"
            ),
            "frozen": (
                "06fa0043117d9b5081ea0ef9c043d7197d638502e6b3b343d126f9004bd19f41"
            ),
            "capacity": (
                "8bef66a0f808ab76f384fae9762b9c2b50e2426f01be45ad4204167640ff973b"
            ),
            "semantic": (
                "bd235f76a0803747e9645449d49917f4360906e45eb7c0cc2084067c13b71d33"
            ),
            "report": (
                "00caea403bd9cfe398c50b4079b35974ec7037f4aa3f7fe0db7bdba5ef2cf585"
            ),
        },
    )

  def test_report_config_raw_hash_precedes_parse(self) -> None:
    inputs = _make_inputs()
    invalid_config = b"\xff"
    events = []
    original_sha256 = ajimee_quality_report._sha256
    original_parse_textproto = ajimee_quality_report._parse_textproto

    def recording_sha256(data):
      if data == invalid_config:
        events.append("hash")
      return original_sha256(data)

    def recording_parse_textproto(message_type, data, label):
      if data == invalid_config:
        events.append("parse")
      return original_parse_textproto(message_type, data, label)

    with mock.patch.object(
        ajimee_quality_report, "_sha256", side_effect=recording_sha256
    ), mock.patch.object(
        ajimee_quality_report,
        "_parse_textproto",
        side_effect=recording_parse_textproto,
    ):
      with self.assertRaises(UnicodeDecodeError):
        ajimee_quality_report.build_quality_report(
            dataclasses.replace(
                inputs, report_config_textproto=invalid_config
            )
        )
    self.assertEqual(events, ["hash", "parse"])

  def test_checked_config_pins_every_available_identity(self) -> None:
    report_config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
    text_format.Parse(_CHECKED_REPORT_CONFIG.read_text("utf-8"), report_config)
    ajimee_quality_report._require_initialized_without_unknown_fields(
        report_config, "checked AJIMEE quality report config"
    )
    ajimee_quality_report._validate_quality_config(report_config)

    capacity_config = ajimee_capacity_audit_pb2.AjimeeCapacityAuditConfig()
    capacity_config_bytes = _CHECKED_CAPACITY_CONFIG.read_bytes()
    text_format.Parse(capacity_config_bytes.decode("utf-8"), capacity_config)
    freezer_config = ajimee_frozen_corpus_pb2.AjimeeFreezerConfig()
    text_format.Parse(_CHECKED_FREEZER_CONFIG.read_text("utf-8"), freezer_config)
    self.assertEqual(report_config.expected_source, freezer_config.expected_source)
    self.assertEqual(
        (
            report_config.input_corpus_sha256,
            report_config.frozen_corpus_sha256,
            report_config.capacity_audit_config_sha256,
            report_config.capacity_audit_sha256,
            report_config.semantic_results_sha256,
        ),
        (
            freezer_config.input_corpus_sha256,
            capacity_config.frozen_corpus_sha256,
            hashlib.sha256(capacity_config_bytes).hexdigest(),
            "4854ae79bc8a24e292be356180420215dd56e70a04c10b12d2f8b242a75e9a5d",
            "7b1aa411809ca6e74bad9f5f96bcb9706a735833c98c94b4d4a9d39be9ac38c8",
        ),
    )
    self.assertEqual(
        (
            report_config.answer_corpus_sha256,
            report_config.expected_case_count,
            report_config.expected_context_case_count,
            report_config.expected_no_context_case_count,
        ),
        (
            "819465115d76cca65eb9d6c8f3f60f8850723376b440eb5c41f0cd8fd7b2c389",
            200,
            100,
            100,
        ),
    )

  def test_cli_has_only_pinned_flags_and_writes_memory_built_outputs(self) -> None:
    parser = report_ajimee_quality_main._build_argument_parser()
    self.assertEqual(
        tuple(action.option_strings for action in parser._actions),
        tuple(
            [f"--{name}"]
            for name in (
                "config",
                "capacity_audit_config",
                "frozen_corpus",
                "answer_corpus",
                "capacity_audit",
                "semantic_results",
                "output_binary",
                "output_json",
            )
        ),
    )
    self.assertTrue(all(action.required for action in parser._actions))

    inputs = _make_inputs()
    expected = ajimee_quality_report.serialize_quality_report(
        ajimee_quality_report.build_quality_report(inputs)
    )
    with tempfile.TemporaryDirectory() as temp_directory:
      directory = Path(temp_directory)
      paths = {
          "config": directory / "config.textproto",
          "capacity_audit_config": directory / "capacity.textproto",
          "frozen_corpus": directory / "frozen.pb",
          "answer_corpus": directory / "answers.pb",
          "capacity_audit": directory / "capacity.pb",
          "semantic_results": directory / "semantics.pb",
          "output_binary": directory / "report.pb",
          "output_json": directory / "report.json",
      }
      paths["config"].write_bytes(inputs.report_config_textproto)
      paths["capacity_audit_config"].write_bytes(
          inputs.capacity_audit_config_textproto
      )
      paths["frozen_corpus"].write_bytes(inputs.frozen_corpus_binary)
      paths["answer_corpus"].write_bytes(inputs.answer_corpus_binary)
      paths["capacity_audit"].write_bytes(inputs.capacity_audit_binary)
      paths["semantic_results"].write_bytes(inputs.semantic_results_binary)
      arguments = [
          argument
          for name in (
              "config",
              "capacity_audit_config",
              "frozen_corpus",
              "answer_corpus",
              "capacity_audit",
              "semantic_results",
              "output_binary",
              "output_json",
          )
          for argument in (f"--{name}", str(paths[name]))
      ]
      report_ajimee_quality_main.main(arguments)
      self.assertEqual(paths["output_binary"].read_bytes(), expected.binary)
      self.assertEqual(
          paths["output_json"].read_bytes(), expected.canonical_json
      )

  def test_builds_exact_metrics_coverage_and_canonical_artifacts(self) -> None:
    inputs = _make_inputs()
    report = ajimee_quality_report.build_quality_report(inputs)
    overall, context, no_context = report.metric_slices
    self.assertFalse(report.semantic_all_success)
    self.assertEqual(overall.case_count, 200)
    self.assertEqual(overall.semantic_success_count, 198)
    self.assertEqual(overall.backend_error_count, 1)
    self.assertEqual(overall.invalid_response_count, 1)
    self.assertEqual(overall.oracle_covered_count, 200)
    self.assertEqual(overall.baseline_correct_count, 3)
    self.assertEqual(overall.effective_model_correct_count, 198)
    self.assertEqual(overall.retained_correct_count, 2)
    self.assertEqual(overall.win_count, 196)
    self.assertEqual(overall.regression_count, 1)
    self.assertEqual(overall.unchanged_incorrect_count, 1)
    self.assertEqual(
        (
            overall.baseline_character_error.edit_distance_sum,
            overall.baseline_character_error.reference_unicode_scalar_count,
        ),
        (197, 200),
    )
    self.assertEqual(
        (
            overall.effective_model_character_error.edit_distance_sum,
            overall.effective_model_character_error.reference_unicode_scalar_count,
        ),
        (2, 200),
    )
    self.assertEqual((context.case_count, no_context.case_count), (100, 100))
    self.assertEqual((no_context.win_count, no_context.regression_count), (100, 0))
    self.assertEqual(
        (
            context.baseline_accuracy.numerator,
            context.baseline_accuracy.denominator,
            context.effective_model_accuracy.numerator,
            context.effective_model_accuracy.denominator,
        ),
        (3, 100, 98, 100),
    )
    self.assertEqual(
        (
            no_context.baseline_accuracy.numerator,
            no_context.baseline_accuracy.denominator,
            no_context.effective_model_accuracy.numerator,
            no_context.effective_model_accuracy.denominator,
        ),
        (0, 100, 100, 100),
    )
    self.assertEqual(
        (overall.retention.numerator, overall.retention.denominator),
        (2, 3),
    )
    self.assertFalse(no_context.HasField("retention"))

    capacity_audit = _parse_binary(
        ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
        inputs.capacity_audit_binary,
    )
    frozen = _parse_binary(
        ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
        inputs.frozen_corpus_binary,
    )
    coverage_ordinals = (
        tuple(range(200)),
        tuple(
            ordinal
            for ordinal, frozen_case in enumerate(frozen.cases)
            if frozen_case.request.preceding_text
        ),
        tuple(
            ordinal
            for ordinal, frozen_case in enumerate(frozen.cases)
            if not frozen_case.request.preceding_text
        ),
    )
    usage_fields = tuple(
        field.name
        for field in (
            ajimee_capacity_audit_pb2.AjimeeCapacityAuditUsage.DESCRIPTOR.fields
        )
    )
    self.assertEqual(len(usage_fields), 22)
    for coverage, ordinals in zip(
        report.capacity_coverage_slices, coverage_ordinals, strict=True
    ):
      audit_cases = [capacity_audit.cases[ordinal] for ordinal in ordinals]
      self.assertEqual(coverage.case_count, len(audit_cases))
      self.assertEqual(coverage.pass_count, len(audit_cases))
      for field_name in usage_fields:
        values = [getattr(case.usage, field_name) for case in audit_cases]
        self.assertEqual(
            getattr(coverage.summed_usage, field_name), sum(values)
        )
        self.assertEqual(
            getattr(coverage.maximum_usage, field_name), max(values)
        )
      expected_histogram = sorted(
          collections.Counter(
              case.usage.decode_batch_count for case in audit_cases
          ).items()
      )
      self.assertEqual(
          [
              (entry.batch_count, entry.case_count)
              for entry in coverage.decode_batch_histogram
          ],
          expected_histogram,
      )
      expected_limiters = {}
      for ordinal, case in zip(ordinals, audit_cases, strict=True):
        for diagnostic in case.segments:
          if not diagnostic.HasField("first_limiter"):
            continue
          limit = diagnostic.first_limiter.limit
          summary = expected_limiters.setdefault(
              limit, {"cases": set(), "segments": 0, "omitted": 0}
          )
          summary["cases"].add(ordinal)
          summary["segments"] += 1
          summary["omitted"] += diagnostic.omitted_by_capacity
      self.assertEqual(
          [
              (
                  entry.limit,
                  entry.case_count,
                  entry.segment_count,
                  entry.omitted_candidate_count,
              )
              for entry in coverage.limiter_summaries
          ],
          [
              (
                  limit,
                  len(summary["cases"]),
                  summary["segments"],
                  summary["omitted"],
              )
              for limit, summary in sorted(expected_limiters.items())
          ],
      )
      self.assertEqual(
          coverage.cases_with_window_omission,
          sum(case.usage.candidates_omitted_by_window > 0 for case in audit_cases),
      )
      self.assertEqual(
          coverage.cases_with_candidate_capacity_omission,
          sum(
              case.usage.candidates_omitted_by_capacity > 0
              for case in audit_cases
          ),
      )
      self.assertEqual(
          coverage.cases_with_segment_capacity_omission,
          sum(case.usage.segments_omitted_by_capacity > 0 for case in audit_cases),
      )
    overall_coverage = report.capacity_coverage_slices[0]
    self.assertEqual(
        [
            (entry.batch_count, entry.case_count)
            for entry in overall_coverage.decode_batch_histogram
        ],
        [(1, 100), (2, 100)],
    )
    self.assertEqual(
        [entry.limit for entry in overall_coverage.limiter_summaries],
        [
            ajimee_capacity_audit_pb2.AJIMEE_FULL_RECORD_TOKENS,
            ajimee_capacity_audit_pb2.AJIMEE_SEQUENCES_PER_DECODE,
            ajimee_capacity_audit_pb2.AJIMEE_OUTPUT_ROWS_PER_DECODE,
        ],
    )

    first = ajimee_quality_report.serialize_quality_report(report)
    second = ajimee_quality_report.serialize_quality_report(
        ajimee_quality_report.build_quality_report(inputs)
    )
    self.assertEqual(first, second)
    parsed_json = json.loads(first.canonical_json)
    self.assertEqual(list(parsed_json)[:3], [
        "schema_version",
        "identity",
        "semantic_all_success",
    ])
    self.assertEqual(parsed_json["schema_version"], "1")
    self.assertIs(parsed_json["semantic_all_success"], False)

    def assert_no_json_numbers(value) -> None:
      if isinstance(value, dict):
        for nested in value.values():
          assert_no_json_numbers(nested)
      elif isinstance(value, list):
        for nested in value:
          assert_no_json_numbers(nested)
      elif not isinstance(value, bool):
        self.assertNotIsInstance(value, (int, float))

    assert_no_json_numbers(parsed_json)
    for poison in ("正", "誤", "他"):
      self.assertNotIn(poison.encode("utf-8"), first.binary)
      self.assertNotIn(poison.encode("utf-8"), first.canonical_json)

  def test_defensive_merge_replay_matches_shared_cpp_fixture(self) -> None:
    request = frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest()
    request.token.session_generation = 0
    request.token.state_revision = 0
    request.token.request_sequence = 1
    request.mode = (
        frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest.MODE_CONVERSION
    )
    request.preceding_text = ""
    request.following_text = ""
    request.reading = "よみ"
    request.focused_segment_id = 0
    for segment_id, candidates in (
        (
            0,
            (
                (0, "baseline-zero", False),
                (1, "protected-middle", True),
                (2, "winner-zero", False),
                (3, "unused-zero", False),
            ),
        ),
        (
            1,
            (
                (4, "baseline-one", False),
                (5, "winner-one", False),
                (6, "unused-one", False),
            ),
        ),
    ):
      segment = request.segments.add()
      segment.id = segment_id
      segment.key = "key"
      for candidate_id, value, protected in candidates:
        candidate = segment.candidates.add()
        candidate.id = candidate_id
        candidate.key = "key"
        candidate.value = value
        candidate.cost = 0
        candidate.attributes = 0
        candidate.consumed_key_size = 0
        candidate.is_protected = protected
    first = ajimee_semantic_results_pb2.AjimeeSemanticSegmentOrder()
    first.segment_id = 1
    first.candidate_ids.extend((5, 4))
    second = ajimee_semantic_results_pb2.AjimeeSemanticSegmentOrder()
    second.segment_id = 0
    second.candidate_ids.extend((2, 0))
    status, complete_orders = ajimee_quality_report._merge_response_orders(
        request, (first, second)
    )
    self.assertEqual(status, ajimee_capacity_audit_pb2.AJIMEE_STATUS_OK)
    self.assertEqual(complete_orders, [[2, 1, 0, 3], [5, 4, 6]])
    self.assertEqual(
        ajimee_quality_report._merged_top(request, complete_orders),
        "winner-zerowinner-one",
    )
    unknown = ajimee_semantic_results_pb2.AjimeeSemanticSegmentOrder()
    unknown.segment_id = 0
    unknown.candidate_ids.append(99)
    status, complete_orders = ajimee_quality_report._merge_response_orders(
        request, (unknown,)
    )
    self.assertEqual(status, ajimee_capacity_audit_pb2.AJIMEE_STATUS_NOT_FOUND)
    self.assertIsNone(complete_orders)

  def test_unicode_scalar_cer_multi_reference_tie_and_oracle(self) -> None:
    self.assertEqual(ajimee_quality_report._levenshtein("😀", ""), 1)
    self.assertEqual(ajimee_quality_report._levenshtein("é", "e\u0301"), 2)
    self.assertEqual(
        ajimee_quality_report._character_error("a", ("ab", "b")), (1, 2)
    )
    self.assertEqual(
        ajimee_quality_report._stable_distinct_outputs(("正", "正", "誤")),
        ("正", "誤"),
    )
    request = frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest()
    request.token.session_generation = 0
    request.token.state_revision = 0
    request.token.request_sequence = 1
    request.mode = (
        frozen_candidate_ranker_pb2.FrozenCandidateRankerRequest.MODE_CONVERSION
    )
    request.preceding_text = ""
    request.following_text = ""
    request.reading = "ab"
    request.focused_segment_id = 0
    for segment_id, values in enumerate((("a", "x"), ("b", "y"))):
      segment = request.segments.add()
      segment.id = segment_id
      segment.key = str(segment_id)
      for candidate_id, value in enumerate(values, start=2 * segment_id):
        candidate = segment.candidates.add()
        candidate.id = candidate_id
        candidate.key = str(segment_id)
        candidate.value = value
        candidate.cost = 0
        candidate.attributes = 0
        candidate.consumed_key_size = 0
        candidate.is_protected = candidate_id == 0
    self.assertTrue(ajimee_quality_report._is_oracle_covered(request, ("ab",)))
    self.assertFalse(ajimee_quality_report._is_oracle_covered(request, ("az",)))

  def test_exact_bootstrap_matches_brute_force_without_seed(self) -> None:
    for wins, regressions, ties in ((1, 0, 0), (0, 0, 3), (1, 1, 1), (2, 1, 0)):
      population = [-1] * regressions + [0] * ties + [1] * wins
      sums = sorted(
          sum(sample)
          for sample in itertools.product(population, repeat=len(population))
      )
      lower_ordinal = next(
          ordinal
          for ordinal in range(len(sums))
          if 40 * (ordinal + 1) >= len(sums)
      )
      upper_ordinal = next(
          ordinal
          for ordinal in range(len(sums))
          if 40 * (ordinal + 1) >= 39 * len(sums)
      )
      expected = (sums[lower_ordinal], sums[upper_ordinal])
      self.assertEqual(
          ajimee_quality_report._exact_bootstrap_endpoints(
              wins, regressions, ties
          ),
          expected,
      )

  def test_pinned_pre_model_facts_are_exact_for_checked_hashes(self) -> None:
    config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
    config.answer_corpus_sha256 = (
        ajimee_quality_report._CHECKED_ANSWER_CORPUS_SHA256
    )
    config.frozen_corpus_sha256 = (
        ajimee_quality_report._CHECKED_FROZEN_CORPUS_SHA256
    )
    accumulators = {}
    for slice_value, facts in ajimee_quality_report._PINNED_PRE_MODEL_FACTS.items():
      accumulator = ajimee_quality_report._new_metric_accumulator()
      (
          accumulator["baseline_correct_count"],
          accumulator["oracle_covered_count"],
          accumulator["baseline_edit_distance"],
          accumulator["baseline_reference_scalars"],
      ) = facts
      accumulators[slice_value] = accumulator
    ajimee_quality_report._validate_pinned_pre_model_facts(
        config, accumulators
    )
    accumulators[ajimee_quality_report_pb2.AJIMEE_REPORT_OVERALL][
        "baseline_correct_count"
    ] += 1
    with self.assertRaisesRegex(ValueError, "pinned pre-model facts"):
      ajimee_quality_report._validate_pinned_pre_model_facts(
          config, accumulators
      )

  def test_validation_is_stage_ordered_before_later_parsing(self) -> None:
    inputs = _make_inputs()

    capacity_config = _parse_capacity_config(inputs)
    capacity_config.schema_version = 2
    invalid = _with_capacity_config(inputs, capacity_config)
    invalid = _with_semantic_bytes(invalid, b"\x80")
    with self.assertRaisesRegex(ValueError, "capacity audit config schema"):
      ajimee_quality_report.build_quality_report(invalid)

    frozen = _parse_binary(
        ajimee_frozen_corpus_pb2.AjimeeFrozenCorpus,
        inputs.frozen_corpus_binary,
    )
    frozen.schema_version = 3
    invalid = _rebuild_joined_inputs(inputs, frozen=frozen)
    invalid = _with_answer_bytes(invalid, b"\x80")
    with self.assertRaisesRegex(ValueError, "frozen corpus schema"):
      ajimee_quality_report.build_quality_report(invalid)

    answers = _parse_binary(
        ajimee_corpus_pb2.AjimeeAnswerCorpus,
        inputs.answer_corpus_binary,
    )
    answers.source.creator = "Independent source mutation"
    invalid = _rebuild_joined_inputs(inputs, answers=answers)
    invalid = _with_capacity_audit_bytes(invalid, b"\x80")
    with self.assertRaisesRegex(ValueError, "answer source identity"):
      ajimee_quality_report.build_quality_report(invalid)

    capacity_audit = _parse_binary(
        ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
        inputs.capacity_audit_binary,
    )
    _swap_messages(capacity_audit.cases, 0, 1)
    invalid = _rebuild_joined_inputs(
        inputs, capacity_audit=capacity_audit
    )
    invalid = _with_semantic_bytes(invalid, b"\x80")
    with self.assertRaisesRegex(ValueError, "capacity cases are unordered"):
      ajimee_quality_report.build_quality_report(invalid)

    semantics = _parse_binary(
        ajimee_semantic_results_pb2.AjimeeSemanticResults,
        inputs.semantic_results_binary,
    )
    semantics.schema_version = 2
    with self.assertRaisesRegex(ValueError, "semantic results schema"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )

  def test_rejects_independent_source_order_hash_and_capacity_mutations(
      self,
  ) -> None:
    inputs = _make_inputs()

    answers = _parse_binary(
        ajimee_corpus_pb2.AjimeeAnswerCorpus,
        inputs.answer_corpus_binary,
    )
    answers.source.changes_notice = "Independent source mutation"
    with self.assertRaisesRegex(ValueError, "answer source identity"):
      ajimee_quality_report.build_quality_report(
          _rebuild_joined_inputs(inputs, answers=answers)
      )

    answers = _parse_binary(
        ajimee_corpus_pb2.AjimeeAnswerCorpus,
        inputs.answer_corpus_binary,
    )
    _swap_messages(answers.cases, 0, 1)
    with self.assertRaisesRegex(ValueError, "answer cases are unordered"):
      ajimee_quality_report.build_quality_report(
          _rebuild_joined_inputs(inputs, answers=answers)
      )

    semantics = _parse_binary(
        ajimee_semantic_results_pb2.AjimeeSemanticResults,
        inputs.semantic_results_binary,
    )
    _swap_messages(semantics.cases, 0, 1)
    with self.assertRaisesRegex(ValueError, "semantic cases are unordered"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )

    capacity_config = _parse_capacity_config(inputs)
    capacity_config.frozen_corpus_sha256 = "f" * 64
    with self.assertRaisesRegex(
        ValueError, "capacity config does not match reporter config"
    ):
      ajimee_quality_report.build_quality_report(
          _with_capacity_config(inputs, capacity_config)
      )

    semantics = _parse_binary(
        ajimee_semantic_results_pb2.AjimeeSemanticResults,
        inputs.semantic_results_binary,
    )
    semantics.identity.capacity_audit_sha256 = "f" * 64
    with self.assertRaisesRegex(ValueError, "semantic results identity"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )

    capacity_audit = _parse_binary(
        ajimee_capacity_audit_pb2.AjimeeCapacityAudit,
        inputs.capacity_audit_binary,
    )
    capacity_audit.cases[0].usage.selected_candidate_count += 1
    with self.assertRaisesRegex(ValueError, "aggregate capacity usage"):
      ajimee_quality_report.build_quality_report(
          _rebuild_joined_inputs(inputs, capacity_audit=capacity_audit)
      )

  def test_rejects_unknown_fields_hash_changes_and_bad_semantic_top(self) -> None:
    inputs = _make_inputs()
    unknown_semantics = inputs.semantic_results_binary + b"\xa0\x06\x01"
    with self.assertRaisesRegex(ValueError, "unknown fields"):
      ajimee_quality_report.build_quality_report(
          _with_semantic_bytes(inputs, unknown_semantics)
      )

    semantics = ajimee_semantic_results_pb2.AjimeeSemanticResults()
    semantics.ParseFromString(inputs.semantic_results_binary)
    nested_identity = semantics.identity.capacity_audit_identity
    nested_identity.ParseFromString(_binary(nested_identity) + b"\xa0\x06\x01")
    with self.assertRaisesRegex(ValueError, "unknown fields"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )

    config = ajimee_quality_report_pb2.AjimeeQualityReportConfig()
    text_format.Parse(inputs.report_config_textproto.decode("utf-8"), config)
    config.answer_corpus_sha256 = "0" * 64
    with self.assertRaisesRegex(ValueError, "SHA256"):
      ajimee_quality_report.build_quality_report(
          dataclasses.replace(inputs, report_config_textproto=_textproto(config))
      )

    semantics = ajimee_semantic_results_pb2.AjimeeSemanticResults()
    semantics.ParseFromString(inputs.semantic_results_binary)
    semantics.cases[0].merged_top_output = "PRIVATE_BAD_TOP"
    with self.assertRaisesRegex(ValueError, "merged top") as raised:
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )
    self.assertNotIn("PRIVATE_BAD_TOP", str(raised.exception))

  def test_invalid_response_token_precedence_and_outcome_fields(self) -> None:
    inputs = _make_inputs()
    semantics = ajimee_semantic_results_pb2.AjimeeSemanticResults()
    semantics.ParseFromString(inputs.semantic_results_binary)
    invalid = semantics.cases[3]
    invalid.response_token_matches_request = False
    invalid.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_ABORTED
    report = ajimee_quality_report.build_quality_report(
        _with_semantics(inputs, semantics)
    )
    self.assertEqual(report.metric_slices[0].invalid_response_count, 1)

    invalid.ClearField("response_token_matches_request")
    with self.assertRaisesRegex(ValueError, "metadata is incomplete"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )

    semantics.ParseFromString(inputs.semantic_results_binary)
    invalid = semantics.cases[3]
    invalid.ClearField("response_segment_orders")
    invalid.response_within_request_bounds = False
    invalid.response_segment_order_count = 3
    invalid.response_candidate_id_count = 7
    invalid.canonical_status_code = (
        ajimee_capacity_audit_pb2.AJIMEE_STATUS_INVALID_ARGUMENT
    )
    report = ajimee_quality_report.build_quality_report(
        _with_semantics(inputs, semantics)
    )
    self.assertEqual(report.metric_slices[0].invalid_response_count, 1)

    invalid.response_segment_orders.add().segment_id = 0
    with self.assertRaisesRegex(ValueError, "stores raw orders"):
      ajimee_quality_report.build_quality_report(
          _with_semantics(inputs, semantics)
      )
    invalid.ClearField("response_segment_orders")
    invalid.response_token_matches_request = False
    invalid.canonical_status_code = ajimee_capacity_audit_pb2.AJIMEE_STATUS_ABORTED
    report = ajimee_quality_report.build_quality_report(
        _with_semantics(inputs, semantics)
    )
    self.assertEqual(report.metric_slices[0].invalid_response_count, 1)


if __name__ == "__main__":
  unittest.main()
