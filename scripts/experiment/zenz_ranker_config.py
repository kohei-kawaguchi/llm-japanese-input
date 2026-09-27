#!/usr/bin/env python3

import json

from cursor_grok_ranker_config import _require_keys
from cursor_grok_ranker_config import _require_positive_int
from cursor_grok_ranker_config import _require_relative_path
from cursor_grok_ranker_config import _require_sha256
from cursor_grok_ranker_config import _require_text


RIGHT_WINDOWS = ("full", "next", "none")
MODEL_FIELDS = ("url", "path", "size", "sha256", "runtime_path", "runtime_sha256")
TEXT_FIELDS = (
    "pre_tokenizer_key",
    "source_pre_tokenizer",
    "runtime_pre_tokenizer",
    "end_token_piece",
    "left_context_tag",
    "right_context_tag",
    "input_tag",
    "output_tag",
    "prompt_version",
    "conversion_expected_command",
    "scores_name",
    "grid_name",
    "holdout_name",
    "ajimee_name",
    "speed_name",
    "reference_model",
    "reference_right_window",
    "shipped_right_window",
    "prediction_name",
)
GRID_FIELDS = ("copy_penalties", "order_priors", "top_ks")
SELECTED_NUMBER_FIELDS = ("selected_copy_penalty", "selected_order_prior")
SPEED_FIELDS = ("speed_models", "right_windows", "speed_top_ks")
PATH_FIELDS = (
    "frozen_corpus_path",
    "answer_source_path",
    "output_directory",
    "holdout_frozen_corpus_path",
    "holdout_answer_source_path",
    "ajimee_frozen_corpus_path",
    "ajimee_answer_corpus_path",
    "prediction_directory",
)
SHA256_FIELDS = (
    "frozen_corpus_sha256",
    "answer_source_sha256",
    "holdout_frozen_corpus_sha256",
    "holdout_answer_source_sha256",
    "ajimee_frozen_corpus_sha256",
    "ajimee_answer_corpus_sha256",
)
POSITIVE_INT_FIELDS = (
    "context_size",
    "thread_count",
    "expected_case_count",
    "expected_baseline_correct_count",
    "minimum_net_gain_cases",
    "maximum_baseline_correct_regressions",
    "metric_definition_version",
    "exact_bootstrap_definition_version",
    "clean_key_copy_definition_version",
    "selected_top_k",
    "holdout_expected_case_count",
    "holdout_expected_baseline_correct_count",
    "holdout_minimum_net_gain_cases",
    "ajimee_expected_case_count",
    "ajimee_expected_baseline_correct_count",
    "ajimee_minimum_gain_points",
    "ajimee_minimum_retention_percent",
    "ajimee_maximum_slice_loss_points",
    "latency_target_p90_milliseconds",
    "shipped_top_k",
    "prediction_trim_characters",
    "prediction_minimum_reading_characters",
    "prediction_top_n",
)


def _require_non_negative_number(value, label):
  if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
    raise ValueError(f"{label} must be a non-negative number")


def _require_list(value, label):
  if not isinstance(value, list) or not value or len(value) != len(
      {json.dumps(item) for item in value}
  ):
    raise ValueError(f"{label} must be a non-empty list of unique values")
  return value


def _validate_model(model, label):
  _require_keys(model, MODEL_FIELDS, label)
  _require_text(model["url"], f"{label}.url")
  _require_relative_path(model["path"], f"{label}.path")
  _require_relative_path(model["runtime_path"], f"{label}.runtime_path")
  _require_positive_int(model["size"], f"{label}.size")
  _require_sha256(model["sha256"], f"{label}.sha256")
  _require_sha256(model["runtime_sha256"], f"{label}.runtime_sha256")


def validate_config(config):
  _require_keys(
      config,
      TEXT_FIELDS + PATH_FIELDS + SHA256_FIELDS + POSITIVE_INT_FIELDS
      + GRID_FIELDS + SELECTED_NUMBER_FIELDS + SPEED_FIELDS + ("models",),
      "zenz ranker config",
  )
  for field in TEXT_FIELDS:
    _require_text(config[field], field)
  for field in PATH_FIELDS:
    _require_relative_path(config[field], field)
  for field in SHA256_FIELDS:
    _require_sha256(config[field], field)
  for field in POSITIVE_INT_FIELDS:
    _require_positive_int(config[field], field)
  if not isinstance(config["models"], dict) or not config["models"]:
    raise ValueError("models must be a non-empty mapping")
  for name, model in config["models"].items():
    _validate_model(model, f"models.{name}")
  for field in ("copy_penalties", "order_priors"):
    for value in _require_list(config[field], field):
      _require_non_negative_number(value, f"{field} item")
  for field in SELECTED_NUMBER_FIELDS:
    _require_non_negative_number(config[field], field)
  for value in _require_list(config["top_ks"], "top_ks"):
    if value is not None:
      _require_positive_int(value, "top_ks item")
  for value in _require_list(config["speed_top_ks"], "speed_top_ks"):
    _require_positive_int(value, "speed_top_ks item")
  for name in _require_list(config["speed_models"], "speed_models") + [
      config["reference_model"]
  ]:
    if name not in config["models"]:
      raise ValueError(f"model {name} is not defined")
  for window in _require_list(config["right_windows"], "right_windows") + [
      config["reference_right_window"],
      config["shipped_right_window"],
  ]:
    if window not in RIGHT_WINDOWS:
      raise ValueError(f"right window {window} is unknown")
  tags = [
      config["left_context_tag"],
      config["right_context_tag"],
      config["input_tag"],
      config["output_tag"],
  ]
  if len(set(tags)) != len(tags):
    raise ValueError("prompt tags must be distinct")
  return config


def load_config(path):
  with open(path, encoding="utf-8") as file:
    return validate_config(json.load(file))
