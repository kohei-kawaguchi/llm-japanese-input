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

"""Imports checked AJIMEE source data into separate input and answer corpora."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from google.protobuf import text_format

from engine.evaluation import ajimee_corpus_pb2
from engine.evaluation import evaluation_artifact_pb2


_SCHEMA_VERSION = 1
_UINT64_MAX = (1 << 64) - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclasses.dataclass(frozen=True)
class AjimeeSource:
  benchmark_name: str
  revision: str
  relative_path: str
  sha256: str
  creator: str
  source_url: str
  license_identifier: str
  license_url: str
  upstream_dataset_url: str
  changes_notice: str


@dataclasses.dataclass(frozen=True)
class AjimeeExpectedCounts:
  total: int
  with_context: int
  without_context: int


@dataclasses.dataclass(frozen=True)
class AjimeeSourceFields:
  source_index: str
  complete_reading: str
  preceding_context: str
  accepted_outputs: str
  original_text_poison: str
  split_data: str


@dataclasses.dataclass(frozen=True)
class AjimeeImportConfig:
  schema_version: int
  source: AjimeeSource
  expected_counts: AjimeeExpectedCounts
  source_fields: AjimeeSourceFields


@dataclasses.dataclass(frozen=True)
class AjimeeCorpora:
  inputs: ajimee_corpus_pb2.AjimeeInputCorpus
  answers: ajimee_corpus_pb2.AjimeeAnswerCorpus


@dataclasses.dataclass(frozen=True)
class AjimeeSerializedArtifacts:
  input_binary: bytes
  input_textproto: bytes
  answer_binary: bytes
  answer_textproto: bytes


_SCHEMA_ONE_SOURCE_FIELDS = AjimeeSourceFields(
    source_index="index",
    complete_reading="input",
    preceding_context="context_text",
    accepted_outputs="expected_output",
    original_text_poison="original_text",
    split_data="splitted_input_for_limited_input_length",
)
_SCHEMA_ONE_ATTRIBUTION = {
    "creator": "azooKey/AJIMEE-Bench contributors",
    "source_url": (
        "https://github.com/azooKey/AJIMEE-Bench/tree/"
        "401666cd56d1a570c2021798b64b6da4396bfd45"
    ),
    "license_identifier": "CC-BY-SA-3.0",
    "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    "upstream_dataset_url": (
        "https://nlp.ist.i.kyoto-u.ac.jp/"
        "?日本語Wikipedia入力誤りデータセット"
    ),
    "changes_notice": (
        "The source JSON was transformed into separated deterministic protobuf "
        "input and answer corpora; original_text and split chunk contents were "
        "omitted."
    ),
}


def _validate_schema_one_config(config: AjimeeImportConfig) -> None:
  if config.schema_version != _SCHEMA_VERSION:
    raise ValueError(
        f"unsupported AJIMEE schema version: {config.schema_version}"
    )
  if config.source_fields != _SCHEMA_ONE_SOURCE_FIELDS:
    raise ValueError("AJIMEE schema-1 source field roles are invalid")
  actual_attribution = {
      name: getattr(config.source, name) for name in _SCHEMA_ONE_ATTRIBUTION
  }
  if actual_attribution != _SCHEMA_ONE_ATTRIBUTION:
    raise ValueError("AJIMEE schema-1 attribution is invalid")


@dataclasses.dataclass(frozen=True)
class _ParsedCase:
  source_index: int
  complete_reading: str
  preceding_context: str
  accepted_outputs: tuple[str, ...]
  has_source_split_data: bool


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON object key: {key}")
    result[key] = value
  return result


def _load_json_bytes(data: bytes, label: str) -> Any:
  decoded = data.decode("utf-8")
  value = json.loads(decoded, object_pairs_hook=_reject_duplicate_object_keys)
  _validate_json_unicode(value, label)
  return value


def _validate_json_unicode(value: Any, path: str) -> None:
  if isinstance(value, str):
    value.encode("utf-8")
    return
  if isinstance(value, list):
    for index, element in enumerate(value):
      _validate_json_unicode(element, f"{path}[{index}]")
    return
  if isinstance(value, dict):
    for key, element in value.items():
      key.encode("utf-8")
      _validate_json_unicode(element, f"{path}.{key}")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
  if type(value) is not dict:
    raise ValueError(f"{path} must be a JSON object")
  return value


def _require_list(value: Any, path: str) -> list[Any]:
  if type(value) is not list:
    raise ValueError(f"{path} must be a JSON array")
  return value


def _require_string(value: Any, path: str) -> str:
  if type(value) is not str:
    raise ValueError(f"{path} must be a string")
  value.encode("utf-8")
  return value


def _require_nonempty_string(value: Any, path: str) -> str:
  result = _require_string(value, path)
  if not result:
    raise ValueError(f"{path} must not be empty")
  return result


def _require_integer(value: Any, path: str) -> int:
  if type(value) is not int:
    raise ValueError(f"{path} must be an integer")
  return value


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], path: str
) -> None:
  actual = set(value)
  if actual != expected:
    raise ValueError(
        f"{path} keys differ: expected {sorted(expected)}, got {sorted(actual)}"
    )


def _read_fields(
    value: dict[str, Any], cls: Callable[..., Any], path: str
) -> Any:
  names = {field.name for field in dataclasses.fields(cls)}
  _require_exact_keys(value, names, path)
  fields = {
      name: _require_nonempty_string(value[name], f"{path}.{name}")
      for name in names
  }
  return cls(**fields)


def load_import_config(config_bytes: bytes) -> AjimeeImportConfig:
  root = _require_mapping(_load_json_bytes(config_bytes, "config"), "config")
  _require_exact_keys(
      root,
      {"schema_version", "source", "expected_counts", "source_fields"},
      "config",
  )
  schema_version = _require_integer(root["schema_version"], "config.schema_version")

  source_value = _require_mapping(root["source"], "config.source")
  _require_exact_keys(
      source_value,
      {
          "benchmark_name",
          "revision",
          "relative_path",
          "sha256",
          "creator",
          "source_url",
          "license_identifier",
          "license_url",
          "upstream_dataset_url",
          "changes_notice",
      },
      "config.source",
  )
  source = AjimeeSource(
      benchmark_name=_require_nonempty_string(
          source_value["benchmark_name"], "config.source.benchmark_name"
      ),
      revision=_require_nonempty_string(
          source_value["revision"], "config.source.revision"
      ),
      relative_path=_require_nonempty_string(
          source_value["relative_path"], "config.source.relative_path"
      ),
      sha256=_require_nonempty_string(
          source_value["sha256"], "config.source.sha256"
      ),
      creator=_require_nonempty_string(
          source_value["creator"], "config.source.creator"
      ),
      source_url=_require_nonempty_string(
          source_value["source_url"], "config.source.source_url"
      ),
      license_identifier=_require_nonempty_string(
          source_value["license_identifier"],
          "config.source.license_identifier",
      ),
      license_url=_require_nonempty_string(
          source_value["license_url"], "config.source.license_url"
      ),
      upstream_dataset_url=_require_nonempty_string(
          source_value["upstream_dataset_url"],
          "config.source.upstream_dataset_url",
      ),
      changes_notice=_require_nonempty_string(
          source_value["changes_notice"], "config.source.changes_notice"
      ),
  )
  if _REVISION_PATTERN.fullmatch(source.revision) is None:
    raise ValueError("config.source.revision must be a lowercase 40-digit hash")
  if _SHA256_PATTERN.fullmatch(source.sha256) is None:
    raise ValueError("config.source.sha256 must be a lowercase SHA256")

  counts_value = _require_mapping(
      root["expected_counts"], "config.expected_counts"
  )
  _require_exact_keys(
      counts_value,
      {"total", "with_context", "without_context"},
      "config.expected_counts",
  )
  counts = AjimeeExpectedCounts(
      total=_require_integer(counts_value["total"], "config.expected_counts.total"),
      with_context=_require_integer(
          counts_value["with_context"], "config.expected_counts.with_context"
      ),
      without_context=_require_integer(
          counts_value["without_context"],
          "config.expected_counts.without_context",
      ),
  )
  if min(dataclasses.astuple(counts)) < 0:
    raise ValueError("AJIMEE expected counts must be nonnegative")
  if counts.total != counts.with_context + counts.without_context:
    raise ValueError("AJIMEE expected context counts must sum to total")

  fields_value = _require_mapping(root["source_fields"], "config.source_fields")
  fields = _read_fields(fields_value, AjimeeSourceFields, "config.source_fields")
  config = AjimeeImportConfig(
      schema_version=schema_version,
      source=source,
      expected_counts=counts,
      source_fields=fields,
  )
  _validate_schema_one_config(config)
  return config


def _require_case_field(
    source_case: dict[str, Any], field_name: str, case_path: str
) -> Any:
  if field_name not in source_case:
    raise ValueError(f"{case_path} is missing required field {field_name}")
  return source_case[field_name]


def _parse_source_index(value: Any, path: str) -> int:
  text = _require_nonempty_string(value, path)
  if not text.isascii() or not text.isdecimal():
    raise ValueError(f"{path} must be an ASCII decimal numeric ID")
  result = int(text)
  if result > _UINT64_MAX:
    raise ValueError(f"{path} exceeds uint64")
  return result


def _parse_source_case(
    source_case_value: Any,
    source_position: int,
    fields: AjimeeSourceFields,
) -> _ParsedCase:
  case_path = f"source[{source_position}]"
  source_case = _require_mapping(source_case_value, case_path)
  source_index = _parse_source_index(
      _require_case_field(source_case, fields.source_index, case_path),
      f"{case_path}.{fields.source_index}",
  )
  complete_reading = _require_nonempty_string(
      _require_case_field(source_case, fields.complete_reading, case_path),
      f"{case_path}.{fields.complete_reading}",
  )
  preceding_context = _require_string(
      _require_case_field(source_case, fields.preceding_context, case_path),
      f"{case_path}.{fields.preceding_context}",
  )

  outputs_value = _require_list(
      _require_case_field(source_case, fields.accepted_outputs, case_path),
      f"{case_path}.{fields.accepted_outputs}",
  )
  if not outputs_value:
    raise ValueError(f"{case_path}.{fields.accepted_outputs} must not be empty")
  accepted_outputs = tuple(
      _require_nonempty_string(
          output, f"{case_path}.{fields.accepted_outputs}[{index}]"
      )
      for index, output in enumerate(outputs_value)
  )

  _require_string(
      _require_case_field(source_case, fields.original_text_poison, case_path),
      f"{case_path}.{fields.original_text_poison}",
  )
  split_value = _require_list(
      _require_case_field(source_case, fields.split_data, case_path),
      f"{case_path}.{fields.split_data}",
  )
  for index, chunk in enumerate(split_value):
    _require_nonempty_string(chunk, f"{case_path}.{fields.split_data}[{index}]")

  return _ParsedCase(
      source_index=source_index,
      complete_reading=complete_reading,
      preceding_context=preceding_context,
      accepted_outputs=accepted_outputs,
      has_source_split_data=bool(split_value),
  )


def _copy_source_identity(
    source: AjimeeSource,
    destination: evaluation_artifact_pb2.EvaluationSourceIdentity,
) -> None:
  destination.benchmark_name = source.benchmark_name
  destination.source_revision = source.revision
  destination.source_relative_path = source.relative_path
  destination.source_sha256 = source.sha256
  destination.creator = source.creator
  destination.source_url = source.source_url
  destination.license_identifier = source.license_identifier
  destination.license_url = source.license_url
  destination.upstream_dataset_url = source.upstream_dataset_url
  destination.changes_notice = source.changes_notice


def build_corpora(source_bytes: bytes, config: AjimeeImportConfig) -> AjimeeCorpora:
  _validate_schema_one_config(config)
  actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
  if actual_sha256 != config.source.sha256:
    raise ValueError(
        "AJIMEE source SHA256 mismatch: "
        f"expected {config.source.sha256}, got {actual_sha256}"
    )

  source_cases = _require_list(
      _load_json_bytes(source_bytes, "source"), "source"
  )
  if len(source_cases) != config.expected_counts.total:
    raise ValueError(
        "AJIMEE source case count mismatch: "
        f"expected {config.expected_counts.total}, got {len(source_cases)}"
    )

  parsed_cases = [
      _parse_source_case(value, position, config.source_fields)
      for position, value in enumerate(source_cases)
  ]
  source_indexes = [case.source_index for case in parsed_cases]
  if len(set(source_indexes)) != len(source_indexes):
    raise ValueError("AJIMEE source numeric IDs must be unique")

  with_context = sum(bool(case.preceding_context) for case in parsed_cases)
  without_context = len(parsed_cases) - with_context
  if with_context != config.expected_counts.with_context:
    raise ValueError(
        "AJIMEE context case count mismatch: "
        f"expected {config.expected_counts.with_context}, got {with_context}"
    )
  if without_context != config.expected_counts.without_context:
    raise ValueError(
        "AJIMEE no-context case count mismatch: "
        f"expected {config.expected_counts.without_context}, got {without_context}"
    )

  inputs = ajimee_corpus_pb2.AjimeeInputCorpus()
  inputs.schema_version = config.schema_version
  _copy_source_identity(config.source, inputs.source)
  answers = ajimee_corpus_pb2.AjimeeAnswerCorpus()
  answers.schema_version = config.schema_version
  _copy_source_identity(config.source, answers.source)

  for parsed_case in sorted(parsed_cases, key=lambda case: case.source_index):
    input_case = inputs.cases.add()
    input_case.source_index = parsed_case.source_index
    input_case.complete_katakana_reading = parsed_case.complete_reading
    input_case.published_preceding_context = parsed_case.preceding_context
    input_case.context_slice = (
        ajimee_corpus_pb2.AjimeeInputCase.CONTEXT_SLICE_HAS_CONTEXT
        if parsed_case.preceding_context
        else ajimee_corpus_pb2.AjimeeInputCase.CONTEXT_SLICE_NO_CONTEXT
    )
    input_case.has_source_split_data = parsed_case.has_source_split_data

    answer_case = answers.cases.add()
    answer_case.source_index = parsed_case.source_index
    answer_case.accepted_whole_outputs.extend(parsed_case.accepted_outputs)

  if not inputs.IsInitialized() or not answers.IsInitialized():
    raise ValueError("AJIMEE importer produced an uninitialized corpus")
  return AjimeeCorpora(inputs=inputs, answers=answers)


def _review_textproto(message: Any) -> bytes:
  text = text_format.MessageToString(
      message,
      as_utf8=True,
      use_index_order=True,
  )
  return text.encode("utf-8")


def serialize_corpora(corpora: AjimeeCorpora) -> AjimeeSerializedArtifacts:
  return AjimeeSerializedArtifacts(
      input_binary=corpora.inputs.SerializeToString(deterministic=True),
      input_textproto=_review_textproto(corpora.inputs),
      answer_binary=corpora.answers.SerializeToString(deterministic=True),
      answer_textproto=_review_textproto(corpora.answers),
  )


def write_corpora(
    output_directory: Path, artifacts: AjimeeSerializedArtifacts
) -> tuple[Path, Path, Path, Path]:
  output_directory.mkdir(parents=True, exist_ok=True)
  input_binary = output_directory / "ajimee_input_corpus.pb"
  input_textproto = output_directory / "ajimee_input_corpus.textproto"
  answer_binary = output_directory / "ajimee_answer_corpus.pb"
  answer_textproto = output_directory / "ajimee_answer_corpus.textproto"
  input_binary.write_bytes(artifacts.input_binary)
  input_textproto.write_bytes(artifacts.input_textproto)
  answer_binary.write_bytes(artifacts.answer_binary)
  answer_textproto.write_bytes(artifacts.answer_textproto)
  return input_binary, input_textproto, answer_binary, answer_textproto
