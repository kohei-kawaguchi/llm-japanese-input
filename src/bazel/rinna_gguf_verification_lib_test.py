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

"""Tests model-free rinna GGUF verification invariants."""

from pathlib import Path
from collections import Counter
import tempfile
import unittest

import rinna_gguf_verification_lib as verification


def valid_config() -> dict[str, object]:
  return {
      "schema_version": 2,
      "artifacts": {
        "f32": {},
        "f16": {},
        "q4_k_m": {"quantization": "q4_k_m"},
      },
      "tensor_layout": {
          "block_count": 0,
          "block_tensors": [],
          "gguf_tensor_count": 1,
          "global_tensors": [{"gguf": "weight", "source": "weight"}],
      },
  }


class RinnaGgufVerificationLibTest(unittest.TestCase):

  def test_schema_requires_version_and_complete_artifact_set(self) -> None:
    verification.validate_config_schema(valid_config())

    wrong_version = valid_config()
    wrong_version["schema_version"] = 1
    with self.assertRaisesRegex(ValueError, "schema"):
      verification.validate_config_schema(wrong_version)

    missing_artifact = valid_config()
    del missing_artifact["artifacts"]["q4_k_m"]
    with self.assertRaisesRegex(ValueError, "artifact set"):
      verification.validate_config_schema(missing_artifact)

  def test_artifact_branch_requires_quantizer_exactly_for_q4(self) -> None:
    config = valid_config()
    self.assertIs(
        verification.select_artifact(config, "f32", False),
        config["artifacts"]["f32"],
    )
    self.assertIs(
        verification.select_artifact(config, "q4_k_m", True),
        config["artifacts"]["q4_k_m"],
    )
    with self.assertRaisesRegex(ValueError, "quantizer"):
      verification.select_artifact(config, "f32", True)
    with self.assertRaisesRegex(ValueError, "quantizer"):
      verification.select_artifact(config, "q4_k_m", False)

  def test_artifact_selection_rejects_duplicate_tensor_specs(self) -> None:
    config = valid_config()
    config["artifacts"]["q4_k_m"]["tensor_type_overrides"] = {
        "weight": "Q6_K",
    }
    config["tensor_layout"]["global_tensors"].append(
        {"gguf": "weight", "source": "weight"}
    )
    config["tensor_layout"]["gguf_tensor_count"] = 2

    with self.assertRaisesRegex(ValueError, "source names are not unique"):
      verification.select_artifact(config, "q4_k_m", True)

  def test_tensor_type_branch_applies_q4_map_and_override(self) -> None:
    artifact = {
        "tensor_type_map": {"F16": "Q4_K", "F32": "F32"},
        "tensor_type_overrides": {"token_embd.weight": "Q6_K"},
    }
    quantization = {"input_artifact": "f16"}

    self.assertEqual(
        verification.resolve_tensor_type(
            "f32", {}, None, {"f32": "F32"}, "weight"
        ),
        ("F32", False),
    )
    self.assertEqual(
        verification.resolve_tensor_type(
            "q4_k_m", artifact, quantization,
            {"f16": "F16"}, "blk.0.attn_qkv.weight"
        ),
        ("Q4_K", False),
    )
    self.assertEqual(
        verification.resolve_tensor_type(
            "q4_k_m", artifact, quantization,
            {"f16": "F16"}, "token_embd.weight"
        ),
        ("Q6_K", True),
    )

  def test_nonquantized_artifacts_reject_tensor_type_overrides(self) -> None:
    for artifact_kind in ("f32", "f16"):
      artifact = {
          "tensor_type_overrides": {"token_embd.weight": "Q6_K"},
      }
      with self.subTest(artifact_kind=artifact_kind):
        with self.assertRaisesRegex(ValueError, "quantized artifact"):
          verification.resolve_tensor_type(
              artifact_kind,
              artifact,
              None,
              {artifact_kind: "F32"},
              "token_embd.weight",
          )

  def test_quantized_override_must_change_the_mapped_type(self) -> None:
    artifact = {
        "tensor_type_map": {"F16": "Q4_K"},
        "tensor_type_overrides": {"weight": "Q4_K"},
    }
    with self.assertRaisesRegex(ValueError, "has no effect"):
      verification.resolve_tensor_type(
          "q4_k_m",
          artifact,
          {"input_artifact": "f16"},
          {"f16": "F16"},
          "weight",
      )

  def test_declared_overrides_must_all_be_used(self) -> None:
    verification.require_exact_usage(
        {"a", "b"}, Counter({"a": 1, "b": 1}), "overrides"
    )
    with self.assertRaisesRegex(ValueError, "unused"):
      verification.require_exact_usage(
          {"a", "b"}, Counter({"a": 1}), "overrides"
      )
    with self.assertRaisesRegex(ValueError, "exactly once"):
      verification.require_exact_usage(
          {"a", "b"}, Counter({"a": 2, "b": 1}), "overrides"
      )

  def test_packing_types_must_all_be_used(self) -> None:
    verification.require_complete_usage({"Q4_K"}, {"Q4_K"}, "packing")
    with self.assertRaisesRegex(ValueError, "unused"):
      verification.require_complete_usage({"Q4_K"}, set(), "packing")

  def test_packed_shape_uses_declared_block_geometry(self) -> None:
    self.assertEqual(
        verification.packed_data_shape(
            [1536, 512], {"block_elements": 256, "block_bytes": 144},
            "qkv"
        ),
        [1536, 288],
    )
    with self.assertRaisesRegex(ValueError, "two-dimensional"):
      verification.packed_data_shape(
          [512], {"block_elements": 256, "block_bytes": 144}, "bad"
      )
    with self.assertRaisesRegex(ValueError, "whole blocks"):
      verification.packed_data_shape(
          [2, 513], {"block_elements": 256, "block_bytes": 144}, "bad"
      )

  def test_report_serialization_and_comparison_are_exact(self) -> None:
    generated = verification.serialize_report({"z": "飴", "a": 1})
    self.assertEqual(generated, "{\n  \"a\": 1,\n  \"z\": \"飴\"\n}\n")
    verification.require_report_match(generated, generated)
    with self.assertRaisesRegex(ValueError, "report differs"):
      verification.require_report_match(generated, generated + "\n")

  def test_verifier_identity_covers_imported_project_library(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      temp_dir = Path(temp)
      entry_point = temp_dir / "verify_rinna_gguf.py"
      library = temp_dir / "rinna_gguf_verification_lib.py"
      entry_point.write_bytes(b"entry point\n")
      library.write_bytes(b"library version one\n")
      first = verification.verifier_code_identity((entry_point, library))
      reversed_order = verification.verifier_code_identity(
          (library, entry_point)
      )
      with self.assertRaisesRegex(ValueError, "closure is incomplete"):
        verification.verifier_code_identity((entry_point,))

      library.write_bytes(b"library version two\n")
      second = verification.verifier_code_identity((entry_point, library))

    self.assertEqual(first, reversed_order)
    self.assertNotEqual(first, second)
    self.assertEqual(
        [source["filename"] for source in first["project_sources"]],
        list(verification.VERIFIER_PROJECT_SOURCE_FILENAMES),
    )


if __name__ == "__main__":
  unittest.main()
