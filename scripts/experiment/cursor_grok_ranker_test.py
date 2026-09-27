#!/usr/bin/env python3

import json
import os
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

import cursor_grok_ranker
import cursor_grok_ranker_client
import cursor_grok_ranker_config
import cursor_grok_ranker_corpus
import cursor_grok_ranker_metrics
import run_cursor_grok_ranker


CONFIG_PATH = os.path.join(
    SCRIPT_DIR, "config", "cursor_grok_ranker.json"
)


def _candidate(candidate_id, key, value, protected=False):
  return cursor_grok_ranker.RankerCandidate(
      id=candidate_id,
      key=key,
      value=value,
      cost=1000,
      attributes=0,
      consumed_key_size=len(key),
      is_protected=protected,
  )


def _request(candidates, protected=None):
  items = []
  for index, (key, value) in enumerate(candidates):
    items.append(_candidate(index, key, value, protected == index))
  return cursor_grok_ranker.RankerRequest(
      token=cursor_grok_ranker.RankerToken(1, 2, 3),
      mode=3,
      preceding_text="前",
      following_text="",
      reading="あめ",
      focused_segment_id=0,
      segments=(
          cursor_grok_ranker.RankerSegment(
              id=0,
              key="あめ",
              candidates=tuple(items),
          ),
      ),
  )


class CursorGrokRankerTest(unittest.TestCase):
  def setUp(self):
    self.config = cursor_grok_ranker_config.load_config(CONFIG_PATH)

  def test_config_matches_checked_frozen_corpus(self):
    root = Path(SCRIPT_DIR).parents[1]
    cases, frozen_sha256 = cursor_grok_ranker_corpus.load_frozen_corpus(
        root / self.config["frozen_corpus_path"],
        self.config["frozen_corpus_sha256"],
        self.config["expected_case_count"],
    )
    answers, answer_sha256 = cursor_grok_ranker_corpus.load_answers(
        root / self.config["answer_source_path"],
        self.config["answer_source_sha256"],
        self.config["conversion_expected_command"],
        cases,
    )
    self.assertEqual(frozen_sha256, self.config["frozen_corpus_sha256"])
    self.assertEqual(answer_sha256, self.config["answer_source_sha256"])
    self.assertEqual(len(cases), self.config["expected_case_count"])
    self.assertEqual(len(answers), self.config["expected_case_count"])
    baseline_correct = sum(
        case.mozc_baseline_output in accepted
        for case, accepted in zip(cases, answers)
    )
    self.assertEqual(
        baseline_correct, self.config["expected_baseline_correct_count"]
    )
    self.assertGreater(len(cases[0].request.segments[0].candidates), 1)

  def test_parse_accepts_complete_permutation(self):
    request = _request((("あめ", "雨"), ("あめ", "飴"), ("あめ", "あめ")))
    response = cursor_grok_ranker.parse_response(
        request,
        json.dumps(
            {
                "token": {
                    "session_generation": 1,
                    "state_revision": 2,
                    "request_sequence": 3,
                },
                "segment_orders": [
                    {"segment_id": 0, "candidate_ids": [2, 0, 1]}
                ],
            },
            ensure_ascii=False,
        ),
    )
    merged = cursor_grok_ranker.merge_response(request, response)
    output, top = cursor_grok_ranker.merged_top_output(request, merged)
    self.assertEqual(output, "あめ")
    self.assertEqual(top[0], 2)

  def test_parse_rejects_unknown_candidate(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")))
    with self.assertRaisesRegex(
        cursor_grok_ranker.RankerError, "unknown candidate"
    ):
      cursor_grok_ranker.parse_response(
          request,
          json.dumps(
              {
                  "token": {
                      "session_generation": 1,
                      "state_revision": 2,
                      "request_sequence": 3,
                  },
                  "segment_orders": [
                      {"segment_id": 0, "candidate_ids": [0, 9]}
                  ],
              }
          ),
      )

  def test_parse_rejects_duplicate_candidate(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")))
    with self.assertRaisesRegex(
        cursor_grok_ranker.RankerError, "more than once"
    ):
      cursor_grok_ranker.parse_response(
          request,
          json.dumps(
              {
                  "token": {
                      "session_generation": 1,
                      "state_revision": 2,
                      "request_sequence": 3,
                  },
                  "segment_orders": [
                      {"segment_id": 0, "candidate_ids": [0, 0]}
                  ],
              }
          ),
      )

  def test_parse_rejects_omitted_candidate(self):
    request = _request((("あめ", "雨"), ("あめ", "飴"), ("あめ", "あめ")))
    with self.assertRaisesRegex(cursor_grok_ranker.RankerError, "omits"):
      cursor_grok_ranker.parse_response(
          request,
          json.dumps(
              {
                  "token": {
                      "session_generation": 1,
                      "state_revision": 2,
                      "request_sequence": 3,
                  },
                  "segment_orders": [
                      {"segment_id": 0, "candidate_ids": [1, 0]}
                  ],
              }
          ),
      )

  def test_parse_restores_protected_candidate(self):
    request = _request(
        (("あめ", "雨"), ("あめ", "飴"), ("あめ", "あめ")),
        protected=0,
    )
    prompt = cursor_grok_ranker.build_prompt(self.config, request)
    payload = json.loads(prompt.rsplit("\n", 1)[1])
    self.assertEqual(
        [candidate["id"] for candidate in payload["segments"][0]["candidates"]],
        [1, 2],
    )
    response = cursor_grok_ranker.parse_response(
        request,
        json.dumps(
            {
                "token": {
                    "session_generation": 1,
                    "state_revision": 2,
                    "request_sequence": 3,
                },
                "segment_orders": [
                    {"segment_id": 0, "candidate_ids": [2, 1]}
                ],
            }
        ),
    )
    merged = cursor_grok_ranker.merge_response(request, response)
    self.assertEqual(merged[0].candidate_ids, (0, 2, 1))

  def test_parse_rejects_protected_candidate_in_reply(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")), protected=0)
    with self.assertRaisesRegex(
        cursor_grok_ranker.RankerError, "protected candidate"
    ):
      cursor_grok_ranker.parse_response(
          request,
          json.dumps(
              {
                  "token": {
                      "session_generation": 1,
                      "state_revision": 2,
                      "request_sequence": 3,
                  },
                  "segment_orders": [
                      {"segment_id": 0, "candidate_ids": [0, 1]}
                  ],
              }
          ),
      )

  def test_parse_rejects_token_mismatch(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")))
    with self.assertRaisesRegex(
        cursor_grok_ranker.RankerError, "token does not match"
    ):
      cursor_grok_ranker.parse_response(
          request,
          json.dumps(
              {
                  "token": {
                      "session_generation": 9,
                      "state_revision": 2,
                      "request_sequence": 3,
                  },
                  "segment_orders": [
                      {"segment_id": 0, "candidate_ids": [1, 0]}
                  ],
              }
          ),
      )

  def test_select_model_requires_unique_high_variant(self):
    class Value:
      def __init__(self, value):
        self.value = value

    class Parameter:
      def __init__(self, parameter_id, values):
        self.id = parameter_id
        self.values = [Value(item) for item in values]

    class Model:
      def __init__(self, model_id, parameters):
        self.id = model_id
        self.parameters = parameters

    selected = cursor_grok_ranker_client.select_model(
        self.config,
        [
            Model("composer-2.5", []),
            Model("grok-4.6-fast", [
                Parameter("effort", ["high"]),
                Parameter("fast", ["false", "true"]),
            ]),
            Model("grok-4.6", [
                Parameter("effort", ["low", "high"]),
                Parameter("fast", ["false", "true"]),
            ]),
        ],
    )
    self.assertEqual(selected["id"], "grok-4.6")
    self.assertEqual(
        selected["params"],
        [
            {"id": "effort", "value": "high"},
            {"id": "fast", "value": "false"},
        ],
    )

  def test_evaluate_cases_applies_development_gates(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")))
    case = cursor_grok_ranker.FrozenCase(
        source_line=1,
        conversion_reading="あめ",
        mozc_baseline_output="飴",
        request=request,
    )
    config = dict(self.config)
    config["expected_case_count"] = 1
    config["expected_baseline_correct_count"] = 0
    config["minimum_net_gain_cases"] = 1
    config["maximum_baseline_correct_regressions"] = 0
    metrics = cursor_grok_ranker_metrics.evaluate_cases(
        config,
        (case,),
        (("雨",),),
        ("雨",),
        ({0: 0},),
    )
    self.assertEqual(metrics["win_count"], 1)
    self.assertTrue(metrics["development_gate_passed"])

  def test_restore_completed_keeps_corpus_prefix(self):
    request = _request((("あめ", "雨"), ("あめ", "飴")))
    case = cursor_grok_ranker.FrozenCase(
        source_line=54,
        conversion_reading="あめ",
        mozc_baseline_output="雨",
        request=request,
    )
    later = cursor_grok_ranker.FrozenCase(
        source_line=55,
        conversion_reading="あめ",
        mozc_baseline_output="雨",
        request=request,
    )
    records, outputs, tops = run_cursor_grok_ranker.restore_completed(
        (case, later),
        {
            "run_name": "run2",
            "completed_case_count": 1,
            "cases": [
                {
                    "source_line": 54,
                    "token": {
                        "session_generation": 1,
                        "state_revision": 2,
                        "request_sequence": 3,
                    },
                    "merged_segment_orders": [
                        {"segment_id": 0, "candidate_ids": [1, 0]}
                    ],
                    "merged_top_output": "飴",
                }
            ],
        },
        "run2",
    )
    self.assertEqual(len(records), 1)
    self.assertEqual(outputs, ["飴"])
    self.assertEqual(tops, [{0: 1}])


if __name__ == "__main__":
  unittest.main()
