#!/usr/bin/env python3

import itertools
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

import cursor_grok_ranker
import cursor_grok_ranker_corpus
import cursor_grok_ranker_metrics
import zenz_ranker
import zenz_ranker_config
import zenz_ranker_model


def _write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8",
  )


def _output_path(root, config, name):
  return root / config["output_directory"] / f"{name}.json"


def _identity(config, frozen_sha256, answer_sha256):
  model = config["models"][config["reference_model"]]
  return {
      "prompt_version": config["prompt_version"],
      "model_name": config["reference_model"],
      "right_window": config["reference_right_window"],
      "model_url": model["url"],
      "model_sha256": model["sha256"],
      "runtime_model_sha256": model["runtime_sha256"],
      "thread_count": config["thread_count"],
      "frozen_corpus_sha256": frozen_sha256,
      "answer_source_sha256": answer_sha256,
  }


def run_score(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  cases, _, frozen_sha256, answer_sha256 = (
      cursor_grok_ranker_corpus.load_development_inputs(config, root)
  )
  score_outputs = _reference_scorer(config, root)
  records = []
  for case in cases:
    prompt_milliseconds, candidate_scores = zenz_ranker.score_request(
        config,
        score_outputs,
        case.request,
        None,
        config["reference_right_window"],
    )
    records.append(
        {
            "source_line": case.source_line,
            "prompt_milliseconds": prompt_milliseconds,
            "candidates": [
                {
                    "segment_id": segment_id,
                    "candidate_id": candidate_id,
                    "score": score,
                    "milliseconds": milliseconds,
                }
                for segment_id, candidate_id, score, milliseconds in candidate_scores
            ],
        }
    )
    print(f"{len(records)}/{len(cases)} {case.source_line}", flush=True)
  _write_json(
      _output_path(root, config, config["scores_name"]),
      {**_identity(config, frozen_sha256, answer_sha256), "cases": records},
  )


def _cells(config):
  return [
      zenz_ranker.CalibrationCell(
          copy_penalty=copy_penalty, order_prior=order_prior, top_k=top_k
      )
      for copy_penalty, order_prior, top_k in itertools.product(
          config["copy_penalties"], config["order_priors"], config["top_ks"]
      )
  ]


def _selection_key(result):
  cell = result["cell"]
  return (
      -result["metrics"]["net_gain_cases"],
      float("inf") if cell["top_k"] is None else cell["top_k"],
      cell["order_prior"],
      cell["copy_penalty"],
  )


def run_grid(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  cases, answers, frozen_sha256, answer_sha256 = (
      cursor_grok_ranker_corpus.load_development_inputs(config, root)
  )
  scores = json.loads(
      _output_path(root, config, config["scores_name"]).read_text(encoding="utf-8")
  )
  if scores["frozen_corpus_sha256"] != frozen_sha256 or scores[
      "runtime_model_sha256"
  ] != config["models"][config["reference_model"]]["runtime_sha256"]:
    raise cursor_grok_ranker.RankerError("cached scores do not match the config")
  if [record["source_line"] for record in scores["cases"]] != [
      case.source_line for case in cases
  ]:
    raise cursor_grok_ranker.RankerError("cached scores do not match the corpus")
  results = []
  for cell in _cells(config):
    model_outputs = []
    top_by_case = []
    latencies = []
    for case, record in zip(cases, scores["cases"]):
      raw_scores = {
          (item["segment_id"], item["candidate_id"]): item["score"]
          for item in record["candidates"]
      }
      response = zenz_ranker.calibrated_response(case.request, raw_scores, cell)
      merged = cursor_grok_ranker.merge_response(case.request, response)
      output, top_by_segment = cursor_grok_ranker.merged_top_output(
          case.request, merged
      )
      model_outputs.append(output)
      top_by_case.append(top_by_segment)
      latencies.append(
          zenz_ranker.request_milliseconds(
              case.request,
              record["prompt_milliseconds"],
              tuple(
                  (
                      item["segment_id"],
                      item["candidate_id"],
                      item["score"],
                      item["milliseconds"],
                  )
                  for item in record["candidates"]
              ),
              cell,
          )
      )
    latencies.sort()
    results.append(
        {
            "cell": {
                "copy_penalty": cell.copy_penalty,
                "order_prior": cell.order_prior,
                "top_k": cell.top_k,
            },
            "metrics": cursor_grok_ranker_metrics.evaluate_cases(
                config, cases, answers, model_outputs, top_by_case
            ),
            "latency_milliseconds": {
                "median": latencies[len(latencies) // 2],
                "p90": latencies[int(len(latencies) * 0.9)],
                "maximum": latencies[-1],
            },
        }
    )
  passing = sorted(
      (result for result in results if result["metrics"]["development_gate_passed"]),
      key=_selection_key,
  )
  report = {
      **_identity(config, frozen_sha256, answer_sha256),
      "metric_definition_version": config["metric_definition_version"],
      "exact_bootstrap_definition_version": (
          config["exact_bootstrap_definition_version"]
      ),
      "clean_key_copy_definition_version": (
          config["clean_key_copy_definition_version"]
      ),
      "cell_count": len(results),
      "passing_cell_count": len(passing),
      "selected": passing[0] if passing else None,
      "best_by_net_gain": sorted(results, key=_selection_key)[0],
      "cells": results,
  }
  _write_json(_output_path(root, config, config["grid_name"]), report)
  return report


def _selected_cell(config):
  return zenz_ranker.CalibrationCell(
      copy_penalty=config["selected_copy_penalty"],
      order_prior=config["selected_order_prior"],
      top_k=config["selected_top_k"],
  )


def _reference_scorer(config, root):
  model, end_token = zenz_ranker_model.load_model(
      config, root, config["reference_model"]
  )
  return zenz_ranker_model.make_scorer(model, end_token)


def _evaluate_variant(
    config,
    score_outputs,
    cases,
    answers,
    expected_case_count,
    baseline_correct,
    cell,
    right_window,
):
  model_outputs = []
  top_by_case = []
  latencies = []
  records = []
  for case in cases:
    started = time.perf_counter()
    _, candidate_scores = zenz_ranker.score_request(
        config, score_outputs, case.request, cell.top_k, right_window
    )
    raw_scores = {
        (segment_id, candidate_id): score
        for segment_id, candidate_id, score, _ in candidate_scores
    }
    response = zenz_ranker.calibrated_response(case.request, raw_scores, cell)
    latencies.append((time.perf_counter() - started) * 1000)
    merged = cursor_grok_ranker.merge_response(case.request, response)
    output, top_by_segment = cursor_grok_ranker.merged_top_output(
        case.request, merged
    )
    model_outputs.append(output)
    top_by_case.append(top_by_segment)
    records.append(
        {
            "source_line": case.source_line,
            "has_context": bool(case.request.preceding_text),
            "merged_top_output": output,
            "request_milliseconds": latencies[-1],
        }
    )
  metric_config = dict(config)
  metric_config["expected_case_count"] = expected_case_count
  metric_config["expected_baseline_correct_count"] = baseline_correct
  metrics = cursor_grok_ranker_metrics.evaluate_cases(
      metric_config, cases, answers, model_outputs, top_by_case
  )
  del metrics["development_gate_passed"]
  ordered = sorted(latencies)
  return {
      "cell": {
          "copy_penalty": cell.copy_penalty,
          "order_prior": cell.order_prior,
          "top_k": cell.top_k,
      },
      "right_window": right_window,
      "metrics": metrics,
      "latency_milliseconds": {
          "median": ordered[len(ordered) // 2],
          "p90": ordered[int(len(ordered) * 0.9)],
          "maximum": ordered[-1],
      },
      "cases": records,
  }, model_outputs


def _evaluate_selected(config, root, cases, answers, expected_case_count, baseline_correct):
  return _evaluate_variant(
      config,
      _reference_scorer(config, root),
      cases,
      answers,
      expected_case_count,
      baseline_correct,
      _selected_cell(config),
      config["reference_right_window"],
  )


def run_holdout(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  cases, frozen_sha256 = cursor_grok_ranker_corpus.load_frozen_corpus(
      root / config["holdout_frozen_corpus_path"],
      config["holdout_frozen_corpus_sha256"],
      config["holdout_expected_case_count"],
  )
  answers, answer_sha256 = cursor_grok_ranker_corpus.load_answers(
      root / config["holdout_answer_source_path"],
      config["holdout_answer_source_sha256"],
      config["conversion_expected_command"],
      cases,
  )
  result, _ = _evaluate_selected(
      config,
      root,
      cases,
      answers,
      config["holdout_expected_case_count"],
      config["holdout_expected_baseline_correct_count"],
  )
  metrics = result["metrics"]
  gate = {
      "net_gain": metrics["net_gain_cases"] >= config["holdout_minimum_net_gain_cases"],
      "bootstrap_lower_gain": metrics["paired_bootstrap"]["lower_gain_cases"] > 0,
      "full_retention": metrics["regression_count"] == 0,
      "no_exact_key_regression": metrics["mozc_correct_exact_key_regression_count"] == 0,
  }
  report = {
      **_identity(config, frozen_sha256, answer_sha256),
      **result,
      "gate": gate,
      "gate_passed": all(gate.values()),
  }
  _write_json(_output_path(root, config, config["holdout_name"]), report)
  return report


def _load_ajimee(config, root):
  cases, frozen_sha256 = cursor_grok_ranker_corpus.load_ajimee_frozen_corpus(
      root / config["ajimee_frozen_corpus_path"],
      config["ajimee_frozen_corpus_sha256"],
      config["ajimee_expected_case_count"],
  )
  answers, answer_sha256 = cursor_grok_ranker_corpus.load_ajimee_answers(
      root / config["ajimee_answer_corpus_path"],
      config["ajimee_answer_corpus_sha256"],
      cases,
  )
  return cases, answers, frozen_sha256, answer_sha256


def _slice_points(cases, answers, model_outputs, has_context):
  selected = [
      (case, accepted, output)
      for case, accepted, output in zip(cases, answers, model_outputs)
      if bool(case.request.preceding_text) == has_context
  ]
  baseline = sum(case.mozc_baseline_output in accepted for case, accepted, _ in selected)
  model = sum(output in accepted for _, accepted, output in selected)
  return {
      "case_count": len(selected),
      "baseline_correct_count": baseline,
      "model_correct_count": model,
      "gain_points": 100 * (model - baseline) / len(selected),
  }


def run_ajimee(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  cases, answers, frozen_sha256, answer_sha256 = _load_ajimee(config, root)
  result, model_outputs = _evaluate_selected(
      config,
      root,
      cases,
      answers,
      config["ajimee_expected_case_count"],
      config["ajimee_expected_baseline_correct_count"],
  )
  metrics = result["metrics"]
  slices = {
      "context": _slice_points(cases, answers, model_outputs, True),
      "no_context": _slice_points(cases, answers, model_outputs, False),
  }
  gain_points = 100 * metrics["net_gain_cases"] / metrics["case_count"]
  retention_percent = (
      100 * metrics["retained_correct_count"] / metrics["baseline_correct_count"]
  )
  gate = {
      "gain_points": gain_points >= config["ajimee_minimum_gain_points"],
      "bootstrap_lower_gain": metrics["paired_bootstrap"]["lower_gain_cases"] > 0,
      "retention": retention_percent >= config["ajimee_minimum_retention_percent"],
      "slice_loss": all(
          item["gain_points"] >= -config["ajimee_maximum_slice_loss_points"]
          for item in slices.values()
      ),
  }
  report = {
      **_identity(config, frozen_sha256, answer_sha256),
      **result,
      "gain_points": gain_points,
      "retention_percent": retention_percent,
      "slices": slices,
      "gate": gate,
      "gate_passed": all(gate.values()),
  }
  _write_json(_output_path(root, config, config["ajimee_name"]), report)
  return report


def _speed_key(config, variant):
  return (
      -variant["development"]["metrics"]["net_gain_cases"],
      config["speed_models"].index(variant["model_name"]) * -1,
      variant["top_k"],
      config["right_windows"].index(variant["right_window"]) * -1,
  )


def run_speed(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  development_cases, development_answers, development_sha256, _ = (
      cursor_grok_ranker_corpus.load_development_inputs(config, root)
  )
  ajimee_cases, ajimee_answers, ajimee_sha256, _ = _load_ajimee(config, root)
  variants = []
  for model_name in config["speed_models"]:
    model, end_token = zenz_ranker_model.load_model(config, root, model_name)
    score_outputs = zenz_ranker_model.make_scorer(model, end_token)
    for right_window in config["right_windows"]:
      for top_k in config["speed_top_ks"]:
        cell = zenz_ranker.CalibrationCell(
            copy_penalty=config["selected_copy_penalty"],
            order_prior=config["selected_order_prior"],
            top_k=top_k,
        )
        development, _ = _evaluate_variant(
            config,
            score_outputs,
            development_cases,
            development_answers,
            config["expected_case_count"],
            config["expected_baseline_correct_count"],
            cell,
            right_window,
        )
        ajimee, _ = _evaluate_variant(
            config,
            score_outputs,
            ajimee_cases,
            ajimee_answers,
            config["ajimee_expected_case_count"],
            config["ajimee_expected_baseline_correct_count"],
            cell,
            right_window,
        )
        target = config["latency_target_p90_milliseconds"]
        variant = {
            "model_name": model_name,
            "right_window": right_window,
            "top_k": top_k,
            "meets_latency_target": (
                development["latency_milliseconds"]["p90"] <= target
                and ajimee["latency_milliseconds"]["p90"] <= target
            ),
            "development": {
                key: value for key, value in development.items() if key != "cases"
            },
            "ajimee": {key: value for key, value in ajimee.items() if key != "cases"},
        }
        variants.append(variant)
        print(
            model_name,
            right_window,
            top_k,
            variant["development"]["metrics"]["net_gain_cases"],
            round(variant["development"]["latency_milliseconds"]["p90"]),
            variant["ajimee"]["metrics"]["model_correct_count"],
            round(variant["ajimee"]["latency_milliseconds"]["p90"]),
            flush=True,
        )
  eligible = sorted(
      (variant for variant in variants if variant["meets_latency_target"]),
      key=lambda variant: _speed_key(config, variant),
  )
  report = {
      "prompt_version": config["prompt_version"],
      "models": {
          name: {"url": spec["url"], "runtime_sha256": spec["runtime_sha256"]}
          for name, spec in config["models"].items()
          if name in config["speed_models"]
      },
      "thread_count": config["thread_count"],
      "development_frozen_corpus_sha256": development_sha256,
      "ajimee_frozen_corpus_sha256": ajimee_sha256,
      "latency_target_p90_milliseconds": config["latency_target_p90_milliseconds"],
      "selected": eligible[0] if eligible else None,
      "variants": variants,
  }
  _write_json(_output_path(root, config, config["speed_name"]), report)
  return report


def _prediction_splits(config, root):
  development, development_answers, _, _ = (
      cursor_grok_ranker_corpus.load_development_inputs(config, root)
  )
  holdout, _ = cursor_grok_ranker_corpus.load_frozen_corpus(
      root / config["holdout_frozen_corpus_path"],
      config["holdout_frozen_corpus_sha256"],
      config["holdout_expected_case_count"],
  )
  holdout_answers, _ = cursor_grok_ranker_corpus.load_answers(
      root / config["holdout_answer_source_path"],
      config["holdout_answer_source_sha256"],
      config["conversion_expected_command"],
      holdout,
  )
  return {
      "development": (development, development_answers),
      "holdout": (holdout, holdout_answers),
  }


def _prediction_path(root, config, split, suffix):
  return root / config["prediction_directory"] / f"{split}_{suffix}"


def run_prediction_inputs(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  for split, (cases, answers) in _prediction_splits(config, root).items():
    lines = []
    targets = {}
    for case, accepted in zip(cases, answers):
      reading = case.conversion_reading
      if len(reading) < config["prediction_minimum_reading_characters"]:
        continue
      lines.append(
          f"{case.source_line}\t{reading[:-config['prediction_trim_characters']]}"
      )
      targets[str(case.source_line)] = list(accepted)
    tsv = _prediction_path(root, config, split, "inputs.tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _write_json(_prediction_path(root, config, split, "targets.json"), targets)
    print(split, len(lines), tsv)


def _top_values(request, merged_orders, top_n):
  segment = request.segments[0]
  order = merged_orders[0].candidate_ids
  values = {candidate.id: candidate.value for candidate in segment.candidates}
  return [values[candidate_id] for candidate_id in order[:top_n]]


def _prediction_metrics(cases, targets, mozc_tops, ranked_tops):
  wins = 0
  regressions = 0
  counts = {"mozc_top1": 0, "zenz_top1": 0, "mozc_topn": 0, "zenz_topn": 0}
  for case, mozc_top, ranked_top in zip(cases, mozc_tops, ranked_tops):
    accepted = targets[str(case.source_line)]
    mozc_hit = mozc_top[0] in accepted
    zenz_hit = ranked_top[0] in accepted
    counts["mozc_top1"] += mozc_hit
    counts["zenz_top1"] += zenz_hit
    counts["mozc_topn"] += any(value in accepted for value in mozc_top)
    counts["zenz_topn"] += any(value in accepted for value in ranked_top)
    wins += zenz_hit and not mozc_hit
    regressions += mozc_hit and not zenz_hit
  lower, upper = cursor_grok_ranker_metrics.exact_bootstrap_endpoints(
      wins, regressions, len(cases) - wins - regressions
  )
  return {
      "case_count": len(cases),
      **counts,
      "win_count": wins,
      "regression_count": regressions,
      "net_gain_cases": wins - regressions,
      "bootstrap_lower_gain_cases": lower,
      "bootstrap_upper_gain_cases": upper,
  }


def run_prediction(config_path):
  config = zenz_ranker_config.load_config(config_path)
  root = Path.cwd()
  cell = zenz_ranker.CalibrationCell(
      copy_penalty=config["selected_copy_penalty"],
      order_prior=config["selected_order_prior"],
      top_k=config["shipped_top_k"],
  )
  score_outputs = _reference_scorer(config, root)
  report = {
      "prompt_version": config["prompt_version"],
      "model_name": config["reference_model"],
      "right_window": config["shipped_right_window"],
      "cell": {
          "copy_penalty": cell.copy_penalty,
          "order_prior": cell.order_prior,
          "top_k": cell.top_k,
      },
      "top_n": config["prediction_top_n"],
      "splits": {},
  }
  for split in ("development", "holdout"):
    frozen_path = _prediction_path(root, config, split, "frozen.pb")
    targets = json.loads(
        _prediction_path(root, config, split, "targets.json").read_text(
            encoding="utf-8"
        )
    )
    cases, frozen_sha256 = cursor_grok_ranker_corpus.load_frozen_corpus(
        frozen_path, cursor_grok_ranker_corpus._sha256(frozen_path.read_bytes()),
        len(targets),
    )
    mozc_tops = []
    ranked_tops = []
    records = []
    for case in cases:
      if len(case.request.segments) != 1:
        raise cursor_grok_ranker.RankerError("prediction request has several segments")
      _, candidate_scores = zenz_ranker.score_request(
          config,
          score_outputs,
          case.request,
          cell.top_k,
          config["shipped_right_window"],
      )
      raw_scores = {
          (segment_id, candidate_id): score
          for segment_id, candidate_id, score, _ in candidate_scores
      }
      response = zenz_ranker.calibrated_response(case.request, raw_scores, cell)
      merged = cursor_grok_ranker.merge_response(case.request, response)
      mozc_order = cursor_grok_ranker.merge_response(
          case.request,
          cursor_grok_ranker.RankerResponse(
              token=case.request.token, segment_orders=()
          ),
      )
      mozc_tops.append(
          _top_values(case.request, mozc_order, config["prediction_top_n"])
      )
      ranked_tops.append(
          _top_values(case.request, merged, config["prediction_top_n"])
      )
      records.append(
          {
              "source_line": case.source_line,
              "preedit": case.conversion_reading,
              "mozc_top": mozc_tops[-1],
              "zenz_top": ranked_tops[-1],
          }
      )
    report["splits"][split] = {
        "frozen_corpus_sha256": frozen_sha256,
        "metrics": _prediction_metrics(cases, targets, mozc_tops, ranked_tops),
        "cases": records,
    }
    print(split, json.dumps(report["splits"][split]["metrics"]), flush=True)
  _write_json(_output_path(root, config, config["prediction_name"]), report)
  return report


def print_usage():
  print(
      "Usage: run_zenz_ranker.py <config> "
      "{validate|prepare|score|grid|holdout|ajimee|speed|"
      "prediction-inputs|prediction}",
      file=sys.stderr,
  )


def main():
  if len(sys.argv) != 3:
    print_usage()
    return 2
  config_path, command = sys.argv[1], sys.argv[2]
  if command == "validate":
    zenz_ranker_config.load_config(config_path)
    print("config ok")
    return 0
  if command == "prepare":
    config = zenz_ranker_config.load_config(config_path)
    for name in config["models"]:
      print(zenz_ranker_model.prepare_model(config, Path.cwd(), name))
    return 0
  if command == "score":
    run_score(config_path)
    return 0
  if command == "grid":
    report = run_grid(config_path)
    print(json.dumps(
        {
            "passing_cell_count": report["passing_cell_count"],
            "selected": report["selected"],
            "best_by_net_gain": report["best_by_net_gain"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0
  if command == "prediction-inputs":
    run_prediction_inputs(config_path)
    return 0
  if command == "prediction":
    run_prediction(config_path)
    return 0
  if command == "speed":
    report = run_speed(config_path)
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))
    return 0
  if command in ("holdout", "ajimee"):
    report = (run_holdout if command == "holdout" else run_ajimee)(config_path)
    print(json.dumps(
        {key: value for key, value in report.items() if key != "cases"},
        ensure_ascii=False,
        indent=2,
    ))
    return 0
  print_usage()
  return 2


if __name__ == "__main__":
  sys.exit(main())
