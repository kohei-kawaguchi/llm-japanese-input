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
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the distribution.
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

"""Verifies pinned rinna safetensors-to-GGUF conversions offline."""

from __future__ import annotations

import argparse
from collections import Counter
import filecmp
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import rinna_gguf_verification_lib as verification_lib
from rinna_gguf_verification_lib import (
    ARTIFACT_KINDS,
    expand_tensor_specifications,
    packed_data_shape,
    require,
    require_complete_usage,
    require_exact_usage,
    require_report_match,
    resolve_tensor_type,
    select_artifact,
    serialize_report,
    sha256_file,
    verifier_code_identity,
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from gguf import GGUFReader  # pylint: disable=g-import-not-at-top
from safetensors import safe_open  # pylint: disable=g-import-not-at-top
from sentencepiece import (  # pylint: disable=g-import-not-at-top
    SentencePieceProcessor,
    sentencepiece_model_pb2,
)


def read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def verify_file(base: Path, specification: dict[str, Any]) -> dict[str, Any]:
  return verify_path(base / specification["filename"], specification)


def verify_path(path: Path, specification: dict[str, Any]) -> dict[str, Any]:
  require(path.name == specification["filename"],
          f"{specification['filename']}: filename differs")
  size_bytes = path.stat().st_size
  digest = sha256_file(path)
  require(size_bytes == specification["size_bytes"],
          f"{specification['filename']}: size differs")
  require(digest == specification["sha256"],
          f"{specification['filename']}: SHA-256 differs")
  return {
      "filename": specification["filename"],
      "sha256": digest,
      "size_bytes": size_bytes,
  }


def expand_excluded_buffers(
    layout: dict[str, Any]) -> list[dict[str, Any]]:
  buffers = []
  for block in range(layout["block_count"]):
    for template in layout["excluded_buffer_templates"]:
      item = dict(template)
      item["source"] = item["source"].format(block=block)
      buffers.append(item)
  return buffers


def field_scalar(reader: GGUFReader, key: str) -> Any:
  field = reader.fields[key]
  value = field.parts[field.data[0]]
  if field.types[-1].name == "STRING":
    return bytes(value).decode("utf-8")
  scalar = value[0]
  return scalar.item() if hasattr(scalar, "item") else scalar


def field_array(reader: GGUFReader, key: str) -> list[Any]:
  field = reader.fields[key]
  values = [field.parts[index] for index in field.data]
  if field.types[-1].name == "STRING":
    return [bytes(value).decode("utf-8") for value in values]
  return [
      value[0].item() if hasattr(value[0], "item") else value[0]
      for value in values
  ]


def verify_metadata(
    reader: GGUFReader,
    config: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
  metadata_config = config["gguf_metadata"]
  expected_scalars = dict(metadata_config["scalars"])
  expected_scalars.update(artifact["metadata_overrides"])
  for key, expected in expected_scalars.items():
    field = reader.fields[key]
    if field.types[-1].name == "FLOAT32":
      expected = np.float32(expected).item()
    require(field_scalar(reader, key) == expected,
            f"GGUF metadata {key} differs")

  verified_arrays = {}
  for key, expected in metadata_config["arrays"].items():
    actual = field_array(reader, key)
    require(actual == expected,
            f"GGUF metadata array {key} differs")
    verified_arrays[key] = actual

  tokenizer_fields = set(config["tokenizer"]["gguf_array_fields"].values())
  expected_fields = set(expected_scalars)
  expected_fields.update(metadata_config["arrays"])
  expected_fields.update(tokenizer_fields)
  require(set(reader.fields) == expected_fields,
          "GGUF metadata field set differs")
  require(set(metadata_config["field_types"]) == expected_fields,
          "GGUF metadata type specification is incomplete")
  verified_types = {}
  for key, expected_types in metadata_config["field_types"].items():
    actual_types = [field_type.name for field_type in reader.fields[key].types]
    require(actual_types == expected_types,
            f"GGUF metadata type {key} differs")
    verified_types[key] = actual_types
  for key in metadata_config["absent_fields"]:
    require(key not in reader.fields, f"unexpected GGUF metadata field {key}")
  return {
      "absent_fields": metadata_config["absent_fields"],
      "arrays": verified_arrays,
      "scalars": expected_scalars,
      "types": verified_types,
  }


def verify_tensors(
    source_path: Path,
    reader: GGUFReader,
    config: dict[str, Any],
    artifact_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
  layout = config["tensor_layout"]
  artifact = config["artifacts"][artifact_kind]
  quantization = (
      config["quantization"][artifact["quantization"]]
      if "quantization" in artifact else None
  )
  specifications = expand_tensor_specifications(layout)
  excluded_buffers = expand_excluded_buffers(layout)
  require(len(reader.tensors) == layout["gguf_tensor_count"],
          "GGUF tensor count differs")
  tensor_names = [tensor.name for tensor in reader.tensors]
  require(len(tensor_names) == len(set(tensor_names)),
          "GGUF tensor names are not unique")
  gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
  expected_source_names = {item["source"] for item in specifications}
  expected_excluded_names = {item["source"] for item in excluded_buffers}
  expected_gguf_names = {item["gguf"] for item in specifications}
  type_overrides = artifact.get("tensor_type_overrides", {})
  used_type_overrides = Counter()
  used_packing_types = set()

  if quantization is not None:
    source_artifact = quantization["input_artifact"]
    source_storage_types = {
        item["artifact_types"][source_artifact] for item in specifications
    }
    require(set(artifact["tensor_type_map"]) == source_storage_types,
            "quantized tensor type map is incomplete")

  tensor_report = []
  excluded_report = []
  with safe_open(source_path, framework="np") as source:
    require(source.metadata() == layout["source_metadata"],
            "safetensors metadata differs")
    source_names = set(source.keys())
    require(len(source_names) == layout["source_tensor_count"],
            "safetensors tensor count differs")
    require(source_names == expected_source_names | expected_excluded_names,
            "safetensors tensor set differs")
    require(set(gguf_tensors) == expected_gguf_names,
            "GGUF tensor set differs")

    for item in excluded_buffers:
      value = source.get_tensor(item["source"])
      require(str(value.dtype) == item["source_dtype"],
              f"{item['source']}: excluded dtype differs")
      require(list(value.shape) == item["source_shape"],
              f"{item['source']}: excluded shape differs")
      excluded_report.append({
          "gguf_included": False,
          "source_name": item["source"],
          "source_dtype": str(value.dtype),
          "source_shape": list(value.shape),
      })

    for item in specifications:
      source_value = source.get_tensor(item["source"])
      require(str(source_value.dtype) == item["source_dtype"],
              f"{item['source']}: source dtype differs")
      require(list(source_value.shape) == item["source_shape"],
              f"{item['source']}: source shape differs")
      expected_value = source_value.T if item["transpose"] else source_value
      require(list(expected_value.shape) == item["gguf_data_shape"],
              f"{item['gguf']}: mapped source shape differs")
      expected_type, used_type_override = resolve_tensor_type(
          artifact_kind,
          artifact,
          quantization,
          item["artifact_types"],
          item["gguf"],
      )
      if used_type_override:
        used_type_overrides[item["gguf"]] += 1

      if expected_type == "F16":
        expected_value = expected_value.astype(np.float16)
      elif expected_type == "F32":
        expected_value = expected_value.astype(np.float32)

      tensor = gguf_tensors[item["gguf"]]
      require(tensor.tensor_type.name == expected_type,
              f"{item['gguf']}: GGUF tensor type differs")
      require(tensor.shape.tolist() == item["gguf_shape"],
              f"{item['gguf']}: GGUF shape differs")
      tensor_value_report = {
          "gguf_name": item["gguf"],
          "gguf_shape": tensor.shape.tolist(),
          "source_name": item["source"],
          "source_shape": list(source_value.shape),
          "storage_type": tensor.tensor_type.name,
          "transpose": item["transpose"],
      }
      if expected_type in ("F16", "F32"):
        require(list(tensor.data.shape) == item["gguf_data_shape"],
                f"{item['gguf']}: GGUF data shape differs")
        require(expected_value.shape == tensor.data.shape,
                f"{item['gguf']}: mapped shape differs")
        require(np.array_equal(expected_value, tensor.data),
                f"{item['gguf']}: mapped values differ")
        tensor_value_report["value_comparison"] = "exact"
      else:
        require(quantization is not None,
                f"unsupported expected tensor type {expected_type}")
        require(expected_type in quantization["packing"],
                f"unsupported quantized tensor type {expected_type}")
        packing = quantization["packing"][expected_type]
        expected_data_shape = packed_data_shape(
            expected_value.shape, packing, item["gguf"]
        )
        require(str(tensor.data.dtype) == "uint8",
                f"{item['gguf']}: packed storage dtype differs")
        require(list(tensor.data.shape) == expected_data_shape,
                f"{item['gguf']}: packed storage shape differs")
        expected_nbytes = int(np.prod(expected_data_shape))
        require(tensor.data.nbytes == expected_nbytes,
                f"{item['gguf']}: packed storage byte count differs")
        require(tensor.n_bytes == expected_nbytes,
                f"{item['gguf']}: GGUF tensor byte count differs")
        tensor_value_report.update({
            "packed_data_shape": expected_data_shape,
            "packed_storage_sha256": hashlib.sha256(
                tensor.data.tobytes(order="C")
            ).hexdigest(),
            "value_comparison": "exact_quantizer_reproduction",
        })
        used_packing_types.add(expected_type)
      tensor_report.append(tensor_value_report)

  require_exact_usage(
      set(type_overrides), used_type_overrides,
      "quantized tensor type overrides"
  )
  if quantization is not None:
    require_complete_usage(
        set(quantization["packing"]), used_packing_types,
        "quantized tensor packing specifications"
    )

  parameter_count = sum(tensor.n_elements for tensor in gguf_tensors.values())
  require(parameter_count == layout["learned_parameter_count"],
          "learned parameter count differs")
  for name in layout["absent_gguf_tensors"]:
    require(name not in gguf_tensors, f"unexpected GGUF tensor {name}")
  type_counts = Counter(tensor.tensor_type.name
                        for tensor in gguf_tensors.values())
  actual_type_counts = dict(sorted(type_counts.items()))
  require(actual_type_counts ==
          artifact["tensor_type_counts"],
          "GGUF tensor type counts differ")
  return tensor_report, excluded_report, actual_type_counts


def sentencepiece_token_type(
    tokenizer: SentencePieceProcessor,
    token_id: int,
    type_ids: dict[str, int],
) -> int:
  if tokenizer.IsUnknown(token_id):
    return type_ids["unknown"]
  if tokenizer.IsControl(token_id):
    return type_ids["control"]
  if tokenizer.IsUnused(token_id):
    return type_ids["unused"]
  if tokenizer.IsByte(token_id):
    return type_ids["byte"]
  return type_ids["normal"]


def verify_tokenizer(
    model_dir: Path,
    reader: GGUFReader,
    config: dict[str, Any],
) -> dict[str, Any]:
  tokenizer_config = config["tokenizer"]
  source_files = config["source_files"]
  tokenizer = SentencePieceProcessor(
      model_file=str(model_dir / source_files["sentencepiece"]["filename"])
  )
  fields = tokenizer_config["gguf_array_fields"]
  gguf_tokens = field_array(reader, fields["tokens"])
  gguf_scores = field_array(reader, fields["scores"])
  gguf_types = field_array(reader, fields["types"])
  vocabulary_size = tokenizer_config["vocabulary_size"]
  require(tokenizer.vocab_size() == vocabulary_size,
          "SentencePiece vocabulary size differs")
  require(len(gguf_tokens) == vocabulary_size,
          "GGUF token count differs")
  require(len(gguf_scores) == vocabulary_size,
          "GGUF token score count differs")
  require(len(gguf_types) == vocabulary_size,
          "GGUF token type count differs")

  type_counts: Counter[int] = Counter()
  for token_id in range(vocabulary_size):
    expected_type = sentencepiece_token_type(
        tokenizer, token_id, tokenizer_config["token_type_ids"]
    )
    require(gguf_tokens[token_id] == tokenizer.IdToPiece(token_id),
            f"token {token_id}: piece differs")
    require(np.float32(gguf_scores[token_id]) ==
            np.float32(tokenizer.GetScore(token_id)),
            f"token {token_id}: score differs")
    require(gguf_types[token_id] == expected_type,
            f"token {token_id}: type differs")
    type_counts[expected_type] += 1
  actual_type_counts = {
      str(key): value for key, value in sorted(type_counts.items())
  }
  require(actual_type_counts == tokenizer_config["token_type_counts"],
          "token type counts differ")

  model_proto = sentencepiece_model_pb2.ModelProto()
  model_proto.ParseFromString(
      (model_dir / source_files["sentencepiece"]["filename"]).read_bytes()
  )
  require(model_proto.trainer_spec.model_type ==
          tokenizer_config["sentencepiece_model_type"],
          "SentencePiece model type differs")
  charsmap = bytes(field_array(reader, fields["precompiled_charsmap"]))
  require(charsmap == model_proto.normalizer_spec.precompiled_charsmap,
          "SentencePiece normalizer charsmap differs")
  require(len(charsmap) == tokenizer_config["normalizer_charsmap_size_bytes"],
          "normalizer charsmap size differs")
  require(hashlib.sha256(charsmap).hexdigest() ==
          tokenizer_config["normalizer_charsmap_sha256"],
          "normalizer charsmap SHA-256 differs")

  model_json = read_json(model_dir / source_files["config"]["filename"])
  tokenizer_json = read_json(
      model_dir / source_files["tokenizer_config"]["filename"]
  )
  special_json = read_json(
      model_dir / source_files["special_tokens"]["filename"]
  )
  for key, expected in config["model_config_expectations"].items():
    require(model_json[key] == expected, f"config.json {key} differs")
  for key, expected in tokenizer_config["config_expectations"].items():
    require(tokenizer_json[key] == expected,
            f"tokenizer_config.json {key} differs")

  special_report = {}
  for item in tokenizer_config["special_tokens"]:
    require(tokenizer.PieceToId(item["piece"]) == item["id"],
            f"{item['name']}: SentencePiece ID differs")
    require(tokenizer_json[item["json_key"]] == item["piece"],
            f"{item['name']}: tokenizer config differs")
    require(special_json[item["json_key"]] == item["piece"],
            f"{item['name']}: special-token map differs")
    if item["model_config_key"] is not None:
      require(model_json[item["model_config_key"]] == item["id"],
              f"{item['name']}: model config ID differs")
    if item["metadata_key"] is None:
      require(item["piece"] in gguf_tokens,
              f"{item['name']}: vocabulary piece is absent")
    else:
      require(field_scalar(reader, item["metadata_key"]) == item["id"],
              f"{item['name']}: GGUF special-token ID differs")
    special_report[item["name"]] = {
        "id": item["id"],
        "metadata": item["metadata_key"] is not None,
        "piece": item["piece"],
    }

  require(field_scalar(reader, tokenizer_config["add_space_prefix_field"]) ==
          model_proto.normalizer_spec.add_dummy_prefix,
          "add-space-prefix flag differs")
  require(field_scalar(
      reader, tokenizer_config["remove_extra_whitespaces_field"]
  ) == model_proto.normalizer_spec.remove_extra_whitespaces,
          "remove-extra-whitespaces flag differs")
  return {
      "normalizer": {
          "add_space_prefix": model_proto.normalizer_spec.add_dummy_prefix,
          "precompiled_charsmap_sha256": hashlib.sha256(charsmap).hexdigest(),
          "precompiled_charsmap_size_bytes": len(charsmap),
          "remove_extra_whitespaces": (
              model_proto.normalizer_spec.remove_extra_whitespaces
          ),
      },
      "sentencepiece_model_type": model_proto.trainer_spec.model_type,
      "special_tokens": special_report,
      "token_type_counts": actual_type_counts,
      "vocabulary_size": vocabulary_size,
      "vocabulary_value_comparison": "exact",
  }


def verify_quantization(
    config_dir: Path,
    model_dir: Path,
    quantizer_path: Path,
    artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
  quantization = config["quantization"][artifact["quantization"]]
  input_artifact_kind = quantization["input_artifact"]
  executable_report = verify_path(
      quantizer_path, quantization["executable"]
  )
  input_report = verify_file(
      model_dir, config["artifacts"][input_artifact_kind]
  )
  input_report["kind"] = input_artifact_kind
  source_report = {
      role: verify_file(config_dir, specification)
      for role, specification in quantization["sources"].items()
  }
  patch_report = [
      verify_file(config_dir, specification)
      for specification in quantization["patches"]
  ]

  with tempfile.TemporaryDirectory(prefix="rinna-q4-verification-") as temp:
    temp_dir = Path(temp)
    output_path = temp_dir / artifact["filename"]
    subprocess.run(
        [
            str(quantizer_path.resolve()),
            str((model_dir / input_report["filename"]).resolve()),
            str(output_path.resolve()),
        ],
        check=True,
    )
    reproduced_report = verify_file(temp_dir, artifact)
    require(filecmp.cmp(
        model_dir / artifact["filename"], output_path, shallow=False
    ), "reproduced quantized artifact bytes differ")

  reproduced_report["kind"] = artifact["quantization"]
  return {
      "byte_comparison": "exact",
      "input_artifact": input_report,
      "llama_cpp_revision": config["conversion"]["llama_cpp_revision"],
      "mode": quantization["mode"],
      "patches": patch_report,
      "quantizer_executable": executable_report,
      "reproduced_artifact": reproduced_report,
      "sources": source_report,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--artifact", choices=ARTIFACT_KINDS, required=True
  )
  parser.add_argument("--config", required=True, type=Path)
  parser.add_argument("--llama-cpp-dir", required=True, type=Path)
  parser.add_argument("--model-dir", required=True, type=Path)
  parser.add_argument("--quantizer", type=Path)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--check-report", action="store_true")
  args = parser.parse_args()

  config = read_json(args.config)
  artifact = select_artifact(
      config, args.artifact, args.quantizer is not None
  )
  requires_quantizer = "quantization" in artifact
  source_report = {
      role: verify_file(args.model_dir, specification)
      for role, specification in config["source_files"].items()
  }
  artifact_report = verify_file(args.model_dir, artifact)
  artifact_report["kind"] = args.artifact

  conversion = config["conversion"]
  converter_source_report = verify_file(
      args.llama_cpp_dir, conversion["source_archive"]
  )
  patch_path = args.config.parent / conversion["patch_filename"]
  require(sha256_file(patch_path) == conversion["patch_sha256"],
          "converter patch SHA-256 differs")

  reader = GGUFReader(args.model_dir / artifact["filename"])
  verified_metadata = verify_metadata(reader, config, artifact)
  tensor_report, excluded_report, tensor_type_counts = verify_tensors(
      args.model_dir / config["source_files"]["weights"]["filename"],
      reader,
      config,
      args.artifact,
  )
  tokenizer_report = verify_tokenizer(args.model_dir, reader, config)
  quantization_report = (
      verify_quantization(
          args.config.parent,
          args.model_dir,
          args.quantizer,
          artifact,
          config,
      ) if requires_quantizer else None
  )

  report = {
      "artifact": artifact_report,
      "conversion": {
          "llama_cpp_revision": conversion["llama_cpp_revision"],
          "patch_filename": conversion["patch_filename"],
          "patch_sha256": conversion["patch_sha256"],
          "source_archive": converter_source_report,
      },
      "excluded_source_buffers": excluded_report,
      "gguf_metadata": verified_metadata,
      "schema_version": 2,
      "source": {
          "files": source_report,
          "model": config["source_model"],
          "revision": config["source_revision"],
      },
      "status": "passed",
      "tensors": {
          "learned_parameter_count": config["tensor_layout"][
              "learned_parameter_count"
          ],
          "tensor_count": len(tensor_report),
          "tensor_type_counts": tensor_type_counts,
          "values": tensor_report,
      },
      "tokenizer": tokenizer_report,
      "verification_config_sha256": sha256_file(args.config),
      "verifier": verifier_code_identity((
          Path(__file__),
          Path(verification_lib.__file__),
      )),
  }
  if quantization_report is not None:
    report["quantization"] = quantization_report
  serialized_report = serialize_report(report)
  if args.check_report:
    require_report_match(
        args.report.read_text(encoding="utf-8"), serialized_report
    )
  else:
    args.report.write_text(
        serialized_report,
        encoding="utf-8",
        newline="\n",
    )
  print(json.dumps({
      "artifact": args.artifact,
      "artifact_sha256": artifact_report["sha256"],
      "status": "passed",
  }, sort_keys=True))


if __name__ == "__main__":
  main()
