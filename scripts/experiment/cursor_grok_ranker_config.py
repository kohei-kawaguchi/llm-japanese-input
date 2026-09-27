#!/usr/bin/env python3

import json
import os
import re
from pathlib import PurePosixPath


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_keys(value, expected, label):
  if not isinstance(value, dict) or set(value) != set(expected):
    raise ValueError(f"{label} keys do not match the required contract")


def _require_text(value, label):
  if not isinstance(value, str) or not value or any(
      character in value for character in "\t\r\n"
  ):
    raise ValueError(f"{label} must be non-empty single-line text")


def _require_relative_path(value, label):
  _require_text(value, label)
  path = PurePosixPath(value)
  if path.is_absolute() or ".." in path.parts or "\\" in value:
    raise ValueError(f"{label} must be a repository-relative POSIX path")


def _require_sha256(value, label):
  if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
    raise ValueError(f"{label} must be a lowercase SHA256")


def _require_positive_int(value, label):
  if not isinstance(value, int) or isinstance(value, bool) or value < 1:
    raise ValueError(f"{label} must be a positive integer")


def _require_non_negative_int(value, label):
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{label} must be a non-negative integer")


def _require_string_list(value, label):
  if not isinstance(value, list) or not value:
    raise ValueError(f"{label} must be a non-empty list")
  items = []
  for index, item in enumerate(value):
    _require_text(item, f"{label}[{index}]")
    items.append(item)
  if len(set(items)) != len(items):
    raise ValueError(f"{label} values must be unique")
  return items


def validate_config(config):
  _require_keys(
      config,
      (
          "api_key_environment",
          "env_file",
          "models_list_url",
          "agents_url",
          "sdk_package",
          "prompt_version",
          "runtime",
          "cloud_repos",
          "agent_name",
          "local_cwd",
          "retry_limit",
          "poll_interval_seconds",
          "request_timeout_seconds",
          "run_timeout_seconds",
          "model_selection",
          "modes",
          "frozen_corpus_path",
          "frozen_corpus_sha256",
          "answer_source_path",
          "answer_source_sha256",
          "conversion_expected_command",
          "output_directory",
          "run_names",
          "expected_case_count",
          "expected_baseline_correct_count",
          "minimum_net_gain_cases",
          "maximum_baseline_correct_regressions",
          "metric_definition_version",
          "exact_bootstrap_definition_version",
          "clean_key_copy_definition_version",
      ),
      "cursor grok ranker config",
  )
  for field in (
      "api_key_environment",
      "models_list_url",
      "agents_url",
      "sdk_package",
      "prompt_version",
      "agent_name",
      "conversion_expected_command",
  ):
    _require_text(config[field], field)
  if config["runtime"] != "cloud":
    raise ValueError("runtime must be cloud")
  if not isinstance(config["cloud_repos"], list) or not config["cloud_repos"]:
    raise ValueError("cloud_repos must be a non-empty list")
  for index, repo in enumerate(config["cloud_repos"]):
    _require_keys(repo, ("url", "starting_ref"), f"cloud_repos[{index}]")
    _require_text(repo["url"], f"cloud_repos[{index}].url")
    _require_text(repo["starting_ref"], f"cloud_repos[{index}].starting_ref")
  _require_relative_path(config["env_file"], "env_file")
  _require_relative_path(config["local_cwd"], "local_cwd")
  _require_positive_int(config["retry_limit"], "retry_limit")
  _require_positive_int(config["poll_interval_seconds"], "poll_interval_seconds")
  _require_positive_int(
      config["request_timeout_seconds"], "request_timeout_seconds"
  )
  _require_positive_int(config["run_timeout_seconds"], "run_timeout_seconds")
  selection = config["model_selection"]
  _require_keys(
      selection,
      (
          "id_required_substrings",
          "id_forbidden_substrings",
          "required_params",
      ),
      "model_selection",
  )
  _require_string_list(
      selection["id_required_substrings"],
      "model_selection.id_required_substrings",
  )
  _require_string_list(
      selection["id_forbidden_substrings"],
      "model_selection.id_forbidden_substrings",
  )
  if not isinstance(selection["required_params"], list) or not selection["required_params"]:
    raise ValueError("model_selection.required_params must be a non-empty list")
  for index, param in enumerate(selection["required_params"]):
    _require_keys(param, ("id", "value"), f"model_selection.required_params[{index}]")
    _require_text(param["id"], f"model_selection.required_params[{index}].id")
    _require_text(param["value"], f"model_selection.required_params[{index}].value")
  modes = config["modes"]
  if not isinstance(modes, dict) or set(modes) != {"1", "2", "3"}:
    raise ValueError("modes must define suggestion, prediction, and conversion")
  for key in ("1", "2", "3"):
    _require_text(modes[key], f"modes.{key}")
  for field in (
      "frozen_corpus_path",
      "answer_source_path",
      "output_directory",
  ):
    _require_relative_path(config[field], field)
  _require_sha256(config["frozen_corpus_sha256"], "frozen_corpus_sha256")
  _require_sha256(config["answer_source_sha256"], "answer_source_sha256")
  _require_string_list(config["run_names"], "run_names")
  _require_positive_int(config["expected_case_count"], "expected_case_count")
  _require_positive_int(
      config["expected_baseline_correct_count"],
      "expected_baseline_correct_count",
  )
  _require_non_negative_int(
      config["minimum_net_gain_cases"],
      "minimum_net_gain_cases",
  )
  _require_non_negative_int(
      config["maximum_baseline_correct_regressions"],
      "maximum_baseline_correct_regressions",
  )
  for field in (
      "metric_definition_version",
      "exact_bootstrap_definition_version",
      "clean_key_copy_definition_version",
  ):
    _require_positive_int(config[field], field)


def load_config(path):
  with open(path, encoding="utf-8") as file:
    config = json.load(file)
  validate_config(config)
  return config


def load_env_file(path):
  with open(path, encoding="utf-8") as file:
    for raw in file:
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      if "=" not in line:
        raise ValueError("env file line is invalid")
      name, value = line.split("=", 1)
      if not name:
        raise ValueError("env file name is empty")
      os.environ[name] = value
