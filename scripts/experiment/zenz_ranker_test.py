#!/usr/bin/env python3

import os
import struct
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

import cursor_grok_ranker
import cursor_grok_ranker_corpus
import zenz_ranker
import zenz_ranker_config
import zenz_ranker_model


CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "zenz_ranker.json")


def _candidate(candidate_id, value, protected=False):
  return cursor_grok_ranker.RankerCandidate(
      id=candidate_id,
      key="",
      value=value,
      cost=1000,
      attributes=0,
      consumed_key_size=0,
      is_protected=protected,
  )


def _request():
  return cursor_grok_ranker.RankerRequest(
      token=cursor_grok_ranker.RankerToken(1, 2, 3),
      mode=3,
      preceding_text="傘を持って",
      following_text="から",
      reading="あめがふる",
      focused_segment_id=0,
      segments=(
          cursor_grok_ranker.RankerSegment(
              id=0,
              key="あめが",
              candidates=(
                  _candidate(0, "飴が"),
                  _candidate(1, "アメガ", protected=True),
                  _candidate(2, "雨が"),
              ),
          ),
          cursor_grok_ranker.RankerSegment(
              id=1,
              key="ふる",
              candidates=(_candidate(3, "降る"), _candidate(4, "振る")),
          ),
      ),
  )


def _string(value):
  encoded = value.encode("utf-8")
  return struct.pack("<Q", len(encoded)) + encoded


def _gguf(pre_tokenizer):
  header = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
  kv = (
      _string("general.alignment") + struct.pack("<II", 4, 32)
      + _string("tokenizer.ggml.pre") + struct.pack("<I", 8) + _string(pre_tokenizer)
  )
  tensor = _string("w") + struct.pack("<I", 1) + struct.pack("<Q", 4)
  tensor += struct.pack("<I", 0) + struct.pack("<Q", 0)
  body = header + kv + tensor
  return body + b"\0" * ((-len(body)) % 32) + b"DATA"


class ZenzRankerTest(unittest.TestCase):
  def setUp(self):
    self.config = zenz_ranker_config.load_config(CONFIG_PATH)

  def test_config_matches_checked_frozen_corpus(self):
    cases, _, _, _ = cursor_grok_ranker_corpus.load_development_inputs(
        self.config, Path(SCRIPT_DIR).parents[1]
    )
    self.assertEqual(len(cases), self.config["expected_case_count"])

  def test_to_katakana_converts_hiragana_only(self):
    self.assertEqual(zenz_ranker.to_katakana("あめがふる。ーA1ゝ"), "アメガフル。ーA1ヽ")

  def test_prompt_orders_context_before_input(self):
    self.assertEqual(
        zenz_ranker.build_prompt(self.config, _request()),
        "傘を持ってからアメガフル",
    )

  def test_prompt_replaces_spaces_and_newlines(self):
    request = _request()
    request = cursor_grok_ranker.RankerRequest(
        token=request.token,
        mode=request.mode,
        preceding_text="a b\n",
        following_text="",
        reading=request.reading,
        focused_segment_id=request.focused_segment_id,
        segments=request.segments,
    )
    self.assertEqual(
        zenz_ranker.build_prompt(self.config, request),
        "a　bアメガフル",
    )

  def test_full_window_uses_baseline_neighbors_and_skips_protected(self):
    self.assertEqual(
        zenz_ranker.candidate_outputs(_request(), None, "full"),
        (
            (0, 0, "", "飴が降る", True),
            (0, 2, "", "雨が降る", True),
            (1, 3, "飴が", "降る", True),
            (1, 4, "飴が", "振る", True),
        ),
    )

  def test_next_window_scores_next_segment_or_end(self):
    self.assertEqual(
        zenz_ranker.candidate_outputs(_request(), None, "next"),
        (
            (0, 0, "", "飴が降る", False),
            (0, 2, "", "雨が降る", False),
            (1, 3, "飴が", "降る", True),
            (1, 4, "飴が", "振る", True),
        ),
    )

  def test_none_window_scores_candidate_only(self):
    self.assertEqual(
        zenz_ranker.candidate_outputs(_request(), None, "none"),
        (
            (0, 0, "", "飴が", False),
            (0, 2, "", "雨が", False),
            (1, 3, "飴が", "降る", False),
            (1, 4, "飴が", "振る", False),
        ),
    )

  def test_candidate_outputs_respect_top_k(self):
    self.assertEqual(
        zenz_ranker.candidate_outputs(_request(), 1, "full"),
        ((0, 0, "", "飴が降る", True), (1, 3, "飴が", "降る", True)),
    )

  def test_calibrated_response_orders_by_score_and_merges_protected(self):
    raw = {(0, 0): -5.0, (0, 2): -1.0, (1, 3): -5.0, (1, 4): -9.0}
    response = zenz_ranker.calibrated_response(
        _request(), raw, zenz_ranker.CalibrationCell(0, 0, None)
    )
    merged = cursor_grok_ranker.merge_response(_request(), response)
    self.assertEqual(merged[0].candidate_ids, (2, 1, 0))
    self.assertEqual(merged[1].candidate_ids, (3, 4))

  def test_calibrated_response_keeps_mozc_order_on_ties(self):
    raw = {(0, 0): 0.0, (0, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0}
    response = zenz_ranker.calibrated_response(
        _request(), raw, zenz_ranker.CalibrationCell(0, 0, None)
    )
    self.assertEqual(response.segment_orders[0].candidate_ids, (0, 2))

  def test_order_prior_subtracts_per_mozc_index(self):
    raw = {(0, 0): -2.0, (0, 2): -1.0, (1, 3): 0.0, (1, 4): 0.0}
    response = zenz_ranker.calibrated_response(
        _request(), raw, zenz_ranker.CalibrationCell(0, 1.5, None)
    )
    self.assertEqual(response.segment_orders[0].candidate_ids, (0, 2))

  def test_copy_penalty_applies_to_hiragana_and_katakana_copies(self):
    request = _request()
    segment = cursor_grok_ranker.RankerSegment(
        id=0,
        key="あめが",
        candidates=(_candidate(0, "雨が"), _candidate(1, "あめが"), _candidate(2, "アメガ")),
    )
    request = cursor_grok_ranker.RankerRequest(
        token=request.token,
        mode=request.mode,
        preceding_text="",
        following_text="",
        reading="あめが",
        focused_segment_id=0,
        segments=(segment,),
    )
    raw = {(0, 0): -3.0, (0, 1): -1.0, (0, 2): -2.0}
    response = zenz_ranker.calibrated_response(
        request, raw, zenz_ranker.CalibrationCell(5, 0, None)
    )
    self.assertEqual(response.segment_orders[0].candidate_ids, (0, 1, 2))

  def test_copy_penalty_is_skipped_when_mozc_top_is_a_copy(self):
    request = _request()
    segment = cursor_grok_ranker.RankerSegment(
        id=0,
        key="あめが",
        candidates=(_candidate(0, "あめが"), _candidate(1, "雨が")),
    )
    request = cursor_grok_ranker.RankerRequest(
        token=request.token,
        mode=request.mode,
        preceding_text="",
        following_text="",
        reading="あめが",
        focused_segment_id=0,
        segments=(segment,),
    )
    raw = {(0, 0): -1.0, (0, 1): -3.0}
    response = zenz_ranker.calibrated_response(
        request, raw, zenz_ranker.CalibrationCell(100, 0, None)
    )
    self.assertEqual(response.segment_orders[0].candidate_ids, (0, 1))

  def test_top_k_limits_scored_candidates_and_latency(self):
    raw = {(0, 0): -5.0, (0, 2): -1.0, (1, 3): -5.0, (1, 4): -1.0}
    cell = zenz_ranker.CalibrationCell(0, 0, 1)
    response = zenz_ranker.calibrated_response(_request(), raw, cell)
    self.assertEqual(response.segment_orders[0].candidate_ids, (0,))
    merged = cursor_grok_ranker.merge_response(_request(), response)
    self.assertEqual(merged[0].candidate_ids, (0, 1, 2))
    self.assertEqual(merged[1].candidate_ids, (3, 4))
    scores = ((0, 0, -5.0, 2.0), (0, 2, -1.0, 3.0), (1, 3, -5.0, 4.0), (1, 4, -1.0, 5.0))
    self.assertEqual(
        zenz_ranker.request_milliseconds(_request(), 10.0, scores, cell), 16.0
    )

  def test_score_request_pairs_scores_with_candidates(self):
    prompt_milliseconds, scores = zenz_ranker.score_request(
        self.config,
        lambda prompt, items: (7.0, [(float(len(body)), 1.0) for _, body, _ in items]),
        _request(),
        None,
        "full",
    )
    self.assertEqual(prompt_milliseconds, 7.0)
    self.assertEqual(scores[1], (0, 2, 4.0, 1.0))

  def test_score_request_rejects_wrong_score_count(self):
    with self.assertRaises(cursor_grok_ranker.RankerError):
      zenz_ranker.score_request(
          self.config,
          lambda prompt, items: (0.0, [(0.0, 0.0)]),
          _request(),
          None,
          "full",
      )

  def test_replace_gguf_string_keeps_aligned_tensor_data(self):
    rewritten = zenz_ranker_model.replace_gguf_string(
        _gguf("gpt2-small-japanese-char"),
        "tokenizer.ggml.pre",
        "gpt2-small-japanese-char",
        "gpt-2",
    )
    self.assertEqual(rewritten, _gguf("gpt-2"))

  def test_replace_gguf_string_rejects_unexpected_value(self):
    with self.assertRaises(cursor_grok_ranker.RankerError):
      zenz_ranker_model.replace_gguf_string(
          _gguf("llama-bpe"), "tokenizer.ggml.pre", "gpt2-small-japanese-char", "gpt-2"
      )


if __name__ == "__main__":
  unittest.main()
