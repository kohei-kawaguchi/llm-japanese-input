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

"""Tests the checked and privacy-separated AJIMEE source import."""

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from engine.evaluation import ajimee_corpus_pb2
from engine.evaluation import ajimee_importer


_TESTDATA = Path(__file__).with_name("testdata")
_PRODUCTION_CONFIG = Path(__file__).with_name("ajimee_source_config.json")


class AjimeeImporterTest(unittest.TestCase):

  def setUp(self) -> None:
    self.source_bytes = (_TESTDATA / "ajimee_synthetic.json").read_bytes()
    self.config = ajimee_importer.load_import_config(
        (_TESTDATA / "ajimee_synthetic_config.json").read_bytes()
    )

  def _config_for_source(
      self, source_bytes: bytes
  ) -> ajimee_importer.AjimeeImportConfig:
    source = dataclasses.replace(
        self.config.source, sha256=hashlib.sha256(source_bytes).hexdigest()
    )
    return dataclasses.replace(self.config, source=source)

  def _modified_source(self, mutate) -> bytes:
    value = json.loads(self.source_bytes.decode("utf-8"))
    mutate(value)
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

  def test_separates_inputs_answers_and_poison_source_fields(self) -> None:
    corpora = ajimee_importer.build_corpora(self.source_bytes, self.config)
    artifacts = ajimee_importer.serialize_corpora(corpora)
    self.assertEqual(
        (
            hashlib.sha256(artifacts.input_binary).hexdigest(),
            hashlib.sha256(artifacts.answer_binary).hexdigest(),
        ),
        (
            "cfc4cdd8bfb156993a01baae3f5cfb9a8ef85db5d1a22cadb4cad8fb1d99c780",
            "7bf926f1d980aa9c1386c0842ecfefb645e1d45923ede9404fd391ea9b76a1cb",
        ),
    )

    self.assertEqual(
        [case.source_index for case in corpora.inputs.cases], [1, 3, 20, 40]
    )
    self.assertEqual(
        [case.source_index for case in corpora.answers.cases], [1, 3, 20, 40]
    )
    input_case = next(
        case for case in corpora.inputs.cases if case.source_index == 40
    )
    self.assertEqual(input_case.complete_katakana_reading, "サンビャク")
    self.assertTrue(input_case.has_source_split_data)
    self.assertEqual(
        input_case.context_slice,
        ajimee_corpus_pb2.AjimeeInputCase.CONTEXT_SLICE_HAS_CONTEXT,
    )
    no_context_case = next(
        case for case in corpora.inputs.cases if case.source_index == 3
    )
    self.assertEqual(
        no_context_case.context_slice,
        ajimee_corpus_pb2.AjimeeInputCase.CONTEXT_SLICE_NO_CONTEXT,
    )

    input_fields = ajimee_corpus_pb2.AjimeeInputCase.DESCRIPTOR.fields_by_name
    answer_fields = ajimee_corpus_pb2.AjimeeAnswerCase.DESCRIPTOR.fields_by_name
    self.assertNotIn("original_text", input_fields)
    self.assertNotIn("split_data", input_fields)
    self.assertNotIn("complete_katakana_reading", answer_fields)
    self.assertNotIn("published_preceding_context", answer_fields)

    all_artifacts = b"".join(dataclasses.astuple(artifacts))
    for poison in (
        "AJIMEE_POISON_CASE_40",
        "AJIMEE_POISON_CASE_3",
        "AJIMEE_POISON_CASE_20",
        "AJIMEE_POISON_CASE_1",
        "サンゼロゼロ",
        "コーショー",
    ):
      self.assertNotIn(poison.encode("utf-8"), all_artifacts)
    self.assertNotIn("三百".encode("utf-8"), artifacts.input_binary)
    self.assertIn("三百".encode("utf-8"), artifacts.answer_binary)
    self.assertNotIn("数量の説明。".encode("utf-8"), artifacts.answer_binary)

  def test_copies_complete_attribution_to_both_corpora(self) -> None:
    corpora = ajimee_importer.build_corpora(self.source_bytes, self.config)
    self.assertEqual(corpora.inputs.source, corpora.answers.source)
    source = corpora.inputs.source
    self.assertEqual(source.creator, "azooKey/AJIMEE-Bench contributors")
    self.assertEqual(
        source.source_url,
        "https://github.com/azooKey/AJIMEE-Bench/tree/"
        "401666cd56d1a570c2021798b64b6da4396bfd45",
    )
    self.assertEqual(source.license_identifier, "CC-BY-SA-3.0")
    self.assertEqual(
        source.license_url,
        "https://creativecommons.org/licenses/by-sa/3.0/",
    )
    self.assertEqual(
        source.upstream_dataset_url,
        "https://nlp.ist.i.kyoto-u.ac.jp/"
        "?日本語Wikipedia入力誤りデータセット",
    )
    self.assertEqual(
        source.changes_notice,
        "The source JSON was transformed into separated deterministic protobuf "
        "input and answer corpora; original_text and split chunk contents were "
        "omitted.",
    )

  def test_production_config_pins_source_roles_and_attribution(self) -> None:
    config = ajimee_importer.load_import_config(_PRODUCTION_CONFIG.read_bytes())
    self.assertEqual(config.schema_version, 1)
    self.assertEqual(config.source.benchmark_name, "AJIMEE-Bench")
    self.assertEqual(
        config.source.revision,
        "401666cd56d1a570c2021798b64b6da4396bfd45",
    )
    self.assertEqual(
        config.source.relative_path, "JWTD_v2/v1/evaluation_items.json"
    )
    self.assertEqual(
        config.source.sha256,
        "e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca",
    )
    self.assertEqual(
        config.expected_counts,
        ajimee_importer.AjimeeExpectedCounts(
            total=200, with_context=100, without_context=100
        ),
    )
    self.assertEqual(
        config.source_fields,
        ajimee_importer.AjimeeSourceFields(
            source_index="index",
            complete_reading="input",
            preceding_context="context_text",
            accepted_outputs="expected_output",
            original_text_poison="original_text",
            split_data="splitted_input_for_limited_input_length",
        ),
    )

  def test_rejects_poison_field_remapping(self) -> None:
    config_value = json.loads(
        (_TESTDATA / "ajimee_synthetic_config.json").read_text(encoding="utf-8")
    )
    fields = config_value["source_fields"]
    fields["complete_reading"], fields["original_text_poison"] = (
        fields["original_text_poison"],
        fields["complete_reading"],
    )
    with self.assertRaisesRegex(ValueError, "source field roles are invalid"):
      ajimee_importer.load_import_config(
          json.dumps(config_value, ensure_ascii=False).encode("utf-8")
      )

  def test_rejects_incorrect_attribution(self) -> None:
    config_value = json.loads(
        (_TESTDATA / "ajimee_synthetic_config.json").read_text(encoding="utf-8")
    )
    config_value["source"]["license_identifier"] = "CC0-1.0"
    with self.assertRaisesRegex(ValueError, "schema-1 attribution is invalid"):
      ajimee_importer.load_import_config(
          json.dumps(config_value, ensure_ascii=False).encode("utf-8")
      )

  def test_build_rejects_remapped_roles_in_loaded_config(self) -> None:
    fields = dataclasses.replace(
        self.config.source_fields,
        complete_reading="original_text",
        original_text_poison="input",
    )
    remapped = dataclasses.replace(self.config, source_fields=fields)
    with self.assertRaisesRegex(ValueError, "source field roles are invalid"):
      ajimee_importer.build_corpora(self.source_bytes, remapped)

  def test_build_rejects_modified_attribution_in_loaded_config(self) -> None:
    source = dataclasses.replace(
        self.config.source, license_identifier="CC0-1.0"
    )
    modified = dataclasses.replace(self.config, source=source)
    with self.assertRaisesRegex(ValueError, "schema-1 attribution is invalid"):
      ajimee_importer.build_corpora(self.source_bytes, modified)

  def test_serialization_and_written_files_are_deterministic(self) -> None:
    first = ajimee_importer.serialize_corpora(
        ajimee_importer.build_corpora(self.source_bytes, self.config)
    )
    second = ajimee_importer.serialize_corpora(
        ajimee_importer.build_corpora(self.source_bytes, self.config)
    )
    self.assertEqual(first, second)

    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
      first_paths = ajimee_importer.write_corpora(Path(first_dir), first)
      second_paths = ajimee_importer.write_corpora(Path(second_dir), second)
      self.assertEqual(
          [path.name for path in first_paths], [path.name for path in second_paths]
      )
      self.assertEqual(
          [path.read_bytes() for path in first_paths],
          [path.read_bytes() for path in second_paths],
      )

  def test_rejects_source_hash_mismatch(self) -> None:
    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
      ajimee_importer.build_corpora(self.source_bytes + b" ", self.config)

  def test_rejects_total_and_context_count_mismatches(self) -> None:
    wrong_total = dataclasses.replace(
        self.config,
        expected_counts=ajimee_importer.AjimeeExpectedCounts(
            total=5, with_context=2, without_context=3
        ),
    )
    with self.assertRaisesRegex(ValueError, "case count mismatch"):
      ajimee_importer.build_corpora(self.source_bytes, wrong_total)

    wrong_slices = dataclasses.replace(
        self.config,
        expected_counts=ajimee_importer.AjimeeExpectedCounts(
            total=4, with_context=3, without_context=1
        ),
    )
    with self.assertRaisesRegex(ValueError, "context case count mismatch"):
      ajimee_importer.build_corpora(self.source_bytes, wrong_slices)

  def test_requires_unique_decimal_source_ids(self) -> None:
    duplicate = self._modified_source(
        lambda cases: cases[1].__setitem__("index", cases[0]["index"])
    )
    with self.assertRaisesRegex(ValueError, "numeric IDs must be unique"):
      ajimee_importer.build_corpora(
          duplicate, self._config_for_source(duplicate)
      )

    integer_id = self._modified_source(
        lambda cases: cases[0].__setitem__("index", 40)
    )
    with self.assertRaisesRegex(ValueError, "must be a string"):
      ajimee_importer.build_corpora(
          integer_id, self._config_for_source(integer_id)
      )

  def test_requires_poison_field_but_never_copies_it(self) -> None:
    missing_poison = self._modified_source(
        lambda cases: cases[0].pop("original_text")
    )
    with self.assertRaisesRegex(ValueError, "missing required field original_text"):
      ajimee_importer.build_corpora(
          missing_poison, self._config_for_source(missing_poison)
      )

  def test_rejects_invalid_utf8_even_when_hash_matches(self) -> None:
    invalid_utf8 = b"\xff"
    with self.assertRaises(UnicodeDecodeError):
      ajimee_importer.build_corpora(
          invalid_utf8, self._config_for_source(invalid_utf8)
      )


if __name__ == "__main__":
  unittest.main()
