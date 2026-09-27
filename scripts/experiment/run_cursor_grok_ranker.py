#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

import cursor_grok_ranker
import cursor_grok_ranker_client
import cursor_grok_ranker_config
import cursor_grok_ranker_corpus
import cursor_grok_ranker_metrics


def _repo_relative(root, value):
  return Path(root) / value


def _write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8",
  )


def _load_runtime(config, root):
  cursor_grok_ranker_config.load_env_file(_repo_relative(root, config["env_file"]))
  api_key = cursor_grok_ranker_client.require_api_key(config)
  models = cursor_grok_ranker_client.list_models(config, api_key)
  return cursor_grok_ranker_client.select_model(config, models), api_key


def _confirm_model(config):
  return _load_runtime(config, Path.cwd())


def _serialize_orders(orders):
  return [
      {
          "segment_id": order.segment_id,
          "candidate_ids": list(order.candidate_ids),
      }
      for order in orders
  ]


def _deserialize_orders(raw_orders):
  return tuple(
      cursor_grok_ranker.RankerSegmentOrder(
          segment_id=order["segment_id"],
          candidate_ids=tuple(order["candidate_ids"]),
      )
      for order in raw_orders
  )


def _case_token(case):
  return {
      "session_generation": case.request.token.session_generation,
      "state_revision": case.request.token.state_revision,
      "request_sequence": case.request.token.request_sequence,
  }


def restore_completed(cases, progress, run_name):
  if progress["run_name"] != run_name:
    raise cursor_grok_ranker.RankerError("progress run name does not match")
  records = progress["cases"]
  if progress["completed_case_count"] != len(records):
    raise cursor_grok_ranker.RankerError("progress case count does not match")
  if len(records) > len(cases):
    raise cursor_grok_ranker.RankerError("progress has more cases than the corpus")
  model_outputs = []
  top_by_case = []
  for index, record in enumerate(records):
    case = cases[index]
    if record["source_line"] != case.source_line:
      raise cursor_grok_ranker.RankerError("progress source line does not match")
    if record["token"] != _case_token(case):
      raise cursor_grok_ranker.RankerError("progress token does not match")
    output, top_by_segment = cursor_grok_ranker.merged_top_output(
        case.request, _deserialize_orders(record["merged_segment_orders"])
    )
    if output != record["merged_top_output"]:
      raise cursor_grok_ranker.RankerError("progress top output does not match")
    model_outputs.append(output)
    top_by_case.append(top_by_segment)
  return list(records), model_outputs, top_by_case


def load_progress(path, cases, run_name):
  if not path.exists():
    return [], [], []
  progress = json.loads(path.read_text(encoding="utf-8"))
  return restore_completed(cases, progress, run_name)


def run_rank(config_path, run_name):
  config = cursor_grok_ranker_config.load_config(config_path)
  if run_name not in config["run_names"]:
    raise cursor_grok_ranker.RankerError("run name is not in the config")
  root = Path.cwd()
  cases, answers, frozen_sha256, answer_sha256 = (
      cursor_grok_ranker_corpus.load_development_inputs(config, root)
  )
  model, api_key = _load_runtime(config, root)
  output_directory = _repo_relative(root, config["output_directory"])
  progress_path = output_directory / f"{run_name}.progress.json"
  case_records, model_outputs, top_by_case = load_progress(
      progress_path, cases, run_name
  )
  if case_records:
    print(f"resumed {len(case_records)}/{len(cases)}", flush=True)
  for case, answer in zip(cases[len(case_records):], answers[len(case_records):]):
    started = time.perf_counter()
    identity, response, merged = cursor_grok_ranker_client.rank_request(
        config, case.request, model, api_key
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    output, top_by_segment = cursor_grok_ranker.merged_top_output(
        case.request, merged
    )
    model_outputs.append(output)
    top_by_case.append(top_by_segment)
    case_records.append(
        {
            "source_line": case.source_line,
            "token": {
                "session_generation": case.request.token.session_generation,
                "state_revision": case.request.token.state_revision,
                "request_sequence": case.request.token.request_sequence,
            },
            "parsed_segment_orders": _serialize_orders(response.segment_orders),
            "merged_segment_orders": _serialize_orders(merged),
            "merged_top_output": output,
            "elapsed_milliseconds": elapsed_ms,
            "agent_id": identity["agent_id"],
            "run_id": identity["run_id"],
        }
    )
    print(
        f"{len(case_records)}/{len(cases)} {case.source_line} {elapsed_ms} "
        f"{identity['agent_id']} {identity['run_id']}",
        flush=True,
    )
    _write_json(
        output_directory / f"{run_name}.progress.json",
        {
            "run_name": run_name,
            "completed_case_count": len(case_records),
            "cases": case_records,
        },
    )
  metrics = cursor_grok_ranker_metrics.evaluate_cases(
      config, cases, answers, model_outputs, top_by_case
  )
  report = {
      "run_name": run_name,
      "prompt_version": config["prompt_version"],
      "sdk_package": config["sdk_package"],
      "runtime": config["runtime"],
      "model": model,
      "frozen_corpus_sha256": frozen_sha256,
      "answer_source_sha256": answer_sha256,
      "metric_definition_version": config["metric_definition_version"],
      "exact_bootstrap_definition_version": (
          config["exact_bootstrap_definition_version"]
      ),
      "clean_key_copy_definition_version": (
          config["clean_key_copy_definition_version"]
      ),
      "metrics": metrics,
      "cases": case_records,
  }
  _write_json(output_directory / f"{run_name}.json", report)
  return report


def compare_runs(config_path):
  config = cursor_grok_ranker_config.load_config(config_path)
  root = Path.cwd()
  output_directory = _repo_relative(root, config["output_directory"])
  reports = []
  for run_name in config["run_names"]:
    path = output_directory / f"{run_name}.json"
    reports.append(json.loads(path.read_text(encoding="utf-8")))
  disagreements = []
  first = reports[0]
  for case_index, first_case in enumerate(first["cases"]):
    for report in reports[1:]:
      other = report["cases"][case_index]
      if (
          first_case["parsed_segment_orders"] != other["parsed_segment_orders"]
          or first_case["merged_top_output"] != other["merged_top_output"]
      ):
        disagreements.append(
            {
                "source_line": first_case["source_line"],
                "run_names": [first["run_name"], report["run_name"]],
            }
        )
  comparison = {
      "run_names": config["run_names"],
      "disagreement_count": len(disagreements),
      "disagreements": disagreements,
      "metrics": [
          {
              "run_name": report["run_name"],
              "metrics": report["metrics"],
          }
          for report in reports
      ],
      "any_development_gate_passed": any(
          report["metrics"]["development_gate_passed"] for report in reports
      ),
  }
  _write_json(output_directory / "comparison.json", comparison)
  return comparison


def print_usage():
  print(
      "Usage: run_cursor_grok_ranker.py <config> "
      "{validate|confirm-model|rank|compare} [run-name]",
      file=sys.stderr,
  )


def main():
  if len(sys.argv) < 3:
    print_usage()
    return 2
  config_path = sys.argv[1]
  command = sys.argv[2]
  if command == "validate" and len(sys.argv) == 3:
    cursor_grok_ranker_config.load_config(config_path)
    return 0
  if command == "confirm-model" and len(sys.argv) == 3:
    config = cursor_grok_ranker_config.load_config(config_path)
    model, _ = _confirm_model(config)
    print(model["id"], json.dumps(model["params"], separators=(",", ":")))
    return 0
  if command == "rank" and len(sys.argv) == 4:
    report = run_rank(config_path, sys.argv[3])
    print(report["run_name"], report["metrics"]["development_gate_passed"])
    return 0
  if command == "compare" and len(sys.argv) == 3:
    comparison = compare_runs(config_path)
    print(
        comparison["disagreement_count"],
        comparison["any_development_gate_passed"],
    )
    return 0
  print_usage()
  return 2


if __name__ == "__main__":
  sys.stdout.reconfigure(newline="\n")
  sys.exit(main())
