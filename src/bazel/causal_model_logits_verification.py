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

"""Shared reference-logit verification for checked causal model imports."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


_MAGIC = b"LLMJLOG1"


@dataclasses.dataclass(frozen=True)
class VerificationProfile:
  source_weight_filename: str
  tokenizer_filename: str
  gguf_sha256_key: str
  tokenizer_use_fast: bool
  tokenizer_legacy: bool | None
  bos_from_model_config: bool
  normalize_record: Callable[[str], str]
  require_unicode_14: bool


def scalar_lower(text: str) -> str:
  return "".join(character.lower() for character in text)


def identity(text: str) -> str:
  return text


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _read_probe_output(path: Path) -> tuple[list[int], list[int], np.ndarray]:
  data = path.read_bytes()
  if data[: len(_MAGIC)] != _MAGIC:
    raise ValueError("probe output magic is invalid")
  offset = len(_MAGIC)
  token_count, vocabulary_size, row_count = struct.unpack_from(
      "<III", data, offset
  )
  offset += struct.calcsize("<III")
  tokens = list(struct.unpack_from(f"<{token_count}i", data, offset))
  offset += token_count * struct.calcsize("<i")
  positions = list(struct.unpack_from(f"<{row_count}I", data, offset))
  offset += row_count * struct.calcsize("<I")
  value_count = row_count * vocabulary_size
  expected_size = offset + value_count * np.dtype("<f4").itemsize
  if len(data) != expected_size:
    raise ValueError("probe output size is invalid")
  logits = np.frombuffer(data, dtype="<f4", count=value_count, offset=offset)
  return tokens, positions, logits.reshape(row_count, vocabulary_size)


def _build_argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cases", required=True, type=Path)
  parser.add_argument("--model-dir", required=True, type=Path)
  parser.add_argument("--gguf", required=True, type=Path)
  parser.add_argument("--probe", required=True, type=Path)
  parser.add_argument("--report", required=True, type=Path)
  return parser


def run(
    profile: VerificationProfile,
    argv: Sequence[str] | None = None,
) -> None:
  args = _build_argument_parser().parse_args(argv)
  if sys.version_info[:2] != (3, 11):
    raise RuntimeError("the reference requires Python 3.11")
  if profile.require_unicode_14 and unicodedata.unidata_version != "14.0.0":
    raise RuntimeError("the reference requires Unicode 14")

  config = json.loads(args.cases.read_text(encoding="utf-8"))
  if config["schema_version"] != 1:
    raise ValueError("reference case schema is invalid")
  source_weight = args.model_dir / profile.source_weight_filename
  tokenizer_file = args.model_dir / profile.tokenizer_filename
  if _sha256(source_weight) != config["source_weight_sha256"]:
    raise ValueError("source weight hash does not match")
  if _sha256(tokenizer_file) != config["tokenizer_sha256"]:
    raise ValueError("tokenizer hash does not match")
  if _sha256(args.gguf) != config[profile.gguf_sha256_key]:
    raise ValueError("GGUF hash does not match")

  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  torch.use_deterministic_algorithms(True)
  tokenizer_arguments = {
      "use_fast": profile.tokenizer_use_fast,
      "local_files_only": True,
  }
  if profile.tokenizer_legacy is not None:
    tokenizer_arguments["legacy"] = profile.tokenizer_legacy
  tokenizer = AutoTokenizer.from_pretrained(
      args.model_dir,
      **tokenizer_arguments,
  )
  model_config = AutoConfig.from_pretrained(
      args.model_dir,
      local_files_only=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
      args.model_dir,
      dtype=torch.float32,
      local_files_only=True,
  )
  model.eval()
  declared_bos_token_id = (
      model_config.bos_token_id
      if profile.bos_from_model_config
      else tokenizer.bos_token_id
  )
  if declared_bos_token_id != config["bos_token_id"]:
    raise ValueError("BOS token ID does not match")

  results = []
  with tempfile.TemporaryDirectory() as temporary_directory:
    temporary = Path(temporary_directory)
    for index, case in enumerate(config["cases"]):
      record = profile.normalize_record(case["record"])
      record_path = temporary / f"record-{index}.txt"
      output_path = temporary / f"output-{index}.bin"
      record_path.write_text(record, encoding="utf-8", newline="")
      subprocess.run(
          [args.probe, args.gguf, record_path, output_path],
          check=True,
      )
      probe_tokens, positions, probe_logits = _read_probe_output(output_path)

      reference_tokens = [config["bos_token_id"]]
      reference_tokens.extend(
          tokenizer.encode(record, add_special_tokens=False)
      )
      if probe_tokens != reference_tokens:
        raise ValueError(f"{case['name']}: token IDs differ")
      expected_positions = sorted(
          {0, len(reference_tokens) // 2, len(reference_tokens) - 1}
      )
      if positions != expected_positions:
        raise ValueError(f"{case['name']}: output positions differ")
      if probe_logits.shape[1] != config["vocabulary_size"]:
        raise ValueError(f"{case['name']}: vocabulary size differs")

      input_ids = torch.tensor([reference_tokens], dtype=torch.long)
      with torch.inference_mode():
        reference_logits = (
            model(input_ids=input_ids)
            .logits[0, expected_positions, :]
            .cpu()
            .numpy()
        )
      if not np.isfinite(reference_logits).all() or not np.isfinite(
          probe_logits
      ).all():
        raise ValueError(f"{case['name']}: logits are not finite")
      absolute_error = np.abs(reference_logits - probe_logits)
      reference_top = np.argmax(reference_logits, axis=1)
      probe_top = np.argmax(probe_logits, axis=1)
      if not np.allclose(
          reference_logits,
          probe_logits,
          rtol=config["rtol"],
          atol=config["atol"],
      ):
        raise ValueError(
            f"{case['name']}: logits exceed tolerance; "
            f"max_abs={absolute_error.max():.9g}, "
            f"mean_abs={absolute_error.mean():.9g}, "
            f"reference_top={reference_top.tolist()}, "
            f"probe_top={probe_top.tolist()}"
        )
      if not np.array_equal(reference_top, probe_top):
        raise ValueError(f"{case['name']}: top tokens differ")

      results.append(
          {
              "name": case["name"],
              "token_count": len(reference_tokens),
              "positions": positions,
              "maximum_absolute_error": float(absolute_error.max()),
              "mean_absolute_error": float(absolute_error.mean()),
              "top_token_ids": [int(value) for value in reference_top],
          }
      )

  report = {
      "schema_version": 1,
      "source_model": config["source_model"],
      "source_revision": config["source_revision"],
      "source_weight_sha256": config["source_weight_sha256"],
      "tokenizer_sha256": config["tokenizer_sha256"],
      profile.gguf_sha256_key: config[profile.gguf_sha256_key],
      "rtol": config["rtol"],
      "atol": config["atol"],
      "cases": results,
  }
  args.report.write_text(
      json.dumps(report, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8",
      newline="\n",
  )
