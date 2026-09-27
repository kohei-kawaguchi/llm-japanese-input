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

"""Model-free invariants for retained rinna GGUF verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT_KINDS = ("f32", "f16", "q4_k_m")
SCHEMA_VERSION = 2
VERIFIER_PROJECT_SOURCE_FILENAMES = (
    "rinna_gguf_verification_lib.py",
    "verify_rinna_gguf.py",
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def verifier_code_identity(source_paths: Sequence[Path]) -> dict[str, Any]:
  paths_by_name = {path.name: path for path in source_paths}
  require(len(paths_by_name) == len(source_paths),
          "verifier project source filenames are not unique")
  require(set(paths_by_name) == set(VERIFIER_PROJECT_SOURCE_FILENAMES),
          "verifier project source closure is incomplete")
  return {
      "project_sources": [
          {
              "filename": filename,
              "sha256": sha256_file(paths_by_name[filename]),
              "size_bytes": paths_by_name[filename].stat().st_size,
          }
          for filename in VERIFIER_PROJECT_SOURCE_FILENAMES
      ],
  }


def expand_tensor_specifications(
    layout: Mapping[str, Any],
) -> list[dict[str, Any]]:
  specifications = [dict(item) for item in layout["global_tensors"]]
  for block in range(layout["block_count"]):
    for template in layout["block_tensors"]:
      item = dict(template)
      item["source"] = item["source"].format(block=block)
      item["gguf"] = item["gguf"].format(block=block)
      specifications.append(item)
  return specifications


def validate_tensor_specifications(layout: Mapping[str, Any]) -> None:
  specifications = expand_tensor_specifications(layout)
  require(len(specifications) == layout["gguf_tensor_count"],
          "tensor specification count differs")
  source_names = [item["source"] for item in specifications]
  require(len(source_names) == len(set(source_names)),
          "tensor specification source names are not unique")
  gguf_names = [item["gguf"] for item in specifications]
  require(len(gguf_names) == len(set(gguf_names)),
          "tensor specification GGUF names are not unique")


def validate_config_schema(config: Mapping[str, Any]) -> None:
  require(config.get("schema_version") == SCHEMA_VERSION,
          "verification config schema is invalid")
  require(set(config.get("artifacts", {})) == set(ARTIFACT_KINDS),
          "verification artifact set is invalid")
  validate_tensor_specifications(config["tensor_layout"])


def select_artifact(
    config: Mapping[str, Any],
    artifact_kind: str,
    quantizer_supplied: bool,
) -> Mapping[str, Any]:
  validate_config_schema(config)
  require(artifact_kind in ARTIFACT_KINDS,
          "verification artifact kind is invalid")
  artifact = config["artifacts"][artifact_kind]
  require(("quantization" in artifact) == quantizer_supplied,
          "quantizer must be specified exactly for a quantized artifact")
  return artifact


def resolve_tensor_type(
    artifact_kind: str,
    artifact: Mapping[str, Any],
    quantization: Mapping[str, Any] | None,
    artifact_types: Mapping[str, str],
    tensor_name: str,
) -> tuple[str, bool]:
  if quantization is None:
    require(not artifact.get("tensor_type_overrides"),
            "tensor type overrides require a quantized artifact")
    require(artifact_kind in artifact_types,
            f"{tensor_name}: artifact tensor type is unspecified")
    return artifact_types[artifact_kind], False

  input_artifact = quantization["input_artifact"]
  require(input_artifact in artifact_types,
          f"{tensor_name}: quantization input tensor type is unspecified")
  input_type = artifact_types[input_artifact]
  type_map = artifact["tensor_type_map"]
  require(input_type in type_map,
          f"{tensor_name}: quantized tensor type is unspecified")
  mapped_type = type_map[input_type]
  overrides = artifact.get("tensor_type_overrides", {})
  if tensor_name not in overrides:
    return mapped_type, False
  override_type = overrides[tensor_name]
  require(override_type != mapped_type,
          f"{tensor_name}: tensor type override has no effect")
  return override_type, True


def require_complete_usage(
    declared: set[str], used: set[str], description: str
) -> None:
  require(used == declared, f"{description} are unused")


def require_exact_usage(
    declared: set[str], usage_counts: Mapping[str, int], description: str
) -> None:
  require(set(usage_counts) == declared, f"{description} are unused")
  require(all(count == 1 for count in usage_counts.values()),
          f"{description} are not used exactly once")


def packed_data_shape(
    logical_data_shape: Sequence[int],
    packing: Mapping[str, int],
    tensor_name: str,
) -> list[int]:
  require(len(logical_data_shape) == 2,
          f"{tensor_name}: quantized tensor is not two-dimensional")
  block_elements = packing["block_elements"]
  block_bytes = packing["block_bytes"]
  require(block_elements > 0 and block_bytes > 0,
          f"{tensor_name}: quantized tensor packing is invalid")
  require(logical_data_shape[-1] % block_elements == 0,
          f"{tensor_name}: quantized row does not fit whole blocks")
  result = list(logical_data_shape)
  result[-1] = result[-1] // block_elements * block_bytes
  return result


def serialize_report(report: Mapping[str, Any]) -> str:
  return json.dumps(
      report, ensure_ascii=False, indent=2, sort_keys=True
  ) + "\n"


def require_report_match(retained: str, generated: str) -> None:
  require(retained == generated, "retained verification report differs")
