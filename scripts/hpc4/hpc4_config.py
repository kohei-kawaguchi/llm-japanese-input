#!/usr/bin/env python3

import json
import os
import re
import sys
from pathlib import PurePosixPath

_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
if _HELPER_DIR not in sys.path:
  sys.path.insert(0, _HELPER_DIR)

import qwen_prepare_config


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


def _validate_named_entries(entries, expected_names, fields, label):
  if not isinstance(entries, list):
    raise ValueError(f"{label} must be a list")
  names = []
  for index, entry in enumerate(entries):
    _require_keys(entry, fields, f"{label}[{index}]")
    _require_text(entry["name"], f"{label}[{index}].name")
    names.append(entry["name"])
  if set(names) != set(expected_names) or len(names) != len(expected_names):
    raise ValueError(f"{label} names do not match the required contract")


def validate_config(config):
  _require_keys(
      config,
      (
          "login_node",
          "repo_dir",
          "account",
          "environment",
          "resources",
          "qwen",
      ),
      "HPC4 config",
  )
  _require_text(config["login_node"], "login_node")
  _require_text(config["repo_dir"], "repo_dir")
  if not config["repo_dir"].startswith("/"):
    raise ValueError("repo_dir must be an absolute remote path")
  _require_text(config["account"], "account")
  _require_keys(
      config["environment"], ("compiler_module",), "environment"
  )
  _require_text(
      config["environment"]["compiler_module"],
      "environment.compiler_module",
  )

  resources = config["resources"]
  _require_keys(resources, ("qwen_setup", "qwen_score"), "resources")
  resource_fields = ("partition", "nodes", "tasks", "cpus", "time", "memory")
  for name, resource in resources.items():
    _require_keys(resource, resource_fields, f"resources.{name}")
    for field in ("partition", "time", "memory"):
      _require_text(resource[field], f"resources.{name}.{field}")
    for field in ("nodes", "tasks", "cpus"):
      if not isinstance(resource[field], int) or resource[field] < 1:
        raise ValueError(f"resources.{name}.{field} must be positive")

  qwen = config["qwen"]
  _require_keys(
      qwen,
      (
          "prepare_config",
          "bazelisk",
          "upload_input_names",
          "build",
          "source_configs",
          "inputs",
          "runs",
          "checksum_output",
          "logs",
      ),
      "qwen",
  )
  _require_relative_path(qwen["prepare_config"], "qwen.prepare_config")
  bazelisk = qwen["bazelisk"]
  _require_keys(bazelisk, ("url", "sha256", "path"), "qwen.bazelisk")
  _require_text(bazelisk["url"], "qwen.bazelisk.url")
  _require_sha256(bazelisk["sha256"], "qwen.bazelisk.sha256")
  _require_relative_path(bazelisk["path"], "qwen.bazelisk.path")
  if not isinstance(qwen["upload_input_names"], list):
    raise ValueError("qwen.upload_input_names must be a list")
  for index, name in enumerate(qwen["upload_input_names"]):
    _require_text(name, f"qwen.upload_input_names[{index}]")
  if len(set(qwen["upload_input_names"])) != len(qwen["upload_input_names"]):
    raise ValueError("qwen.upload_input_names values must be unique")

  build = qwen["build"]
  _require_keys(
      build,
      ("working_directory", "target", "compilation_mode", "executable"),
      "qwen.build",
  )
  _require_relative_path(
      build["working_directory"], "qwen.build.working_directory"
  )
  _require_text(build["target"], "qwen.build.target")
  _require_text(build["compilation_mode"], "qwen.build.compilation_mode")
  _require_relative_path(build["executable"], "qwen.build.executable")

  source_configs = qwen["source_configs"]
  _validate_named_entries(
      source_configs,
      ("execution_config", "score_config"),
      ("name", "path", "sha256"),
      "qwen.source_configs",
  )
  for entry in source_configs:
    _require_relative_path(entry["path"], f"{entry['name']}.path")
    _require_sha256(entry["sha256"], f"{entry['name']}.sha256")

  inputs = qwen["inputs"]
  _validate_named_entries(
      inputs,
      (
          "frozen_corpus",
          "native_coverage",
          "manifest_structured_with_target",
          "manifest_structured_without_target",
          "manifest_natural_text_only",
          "model",
      ),
      ("name", "local_path", "remote_path", "sha256"),
      "qwen.inputs",
  )
  for entry in inputs:
    _require_relative_path(entry["local_path"], f"{entry['name']}.local_path")
    _require_relative_path(entry["remote_path"], f"{entry['name']}.remote_path")
    _require_sha256(entry["sha256"], f"{entry['name']}.sha256")
  if len({entry["local_path"] for entry in inputs}) != len(inputs):
    raise ValueError("qwen input local paths must be unique")
  if len({entry["remote_path"] for entry in inputs}) != len(inputs):
    raise ValueError("qwen input remote paths must be unique")
  input_names = {entry["name"] for entry in inputs}
  for name in qwen["upload_input_names"]:
    if name not in input_names:
      raise ValueError(f"qwen.upload_input_names names unknown input {name}")

  prepare = qwen_prepare_config.load_config(qwen["prepare_config"])
  artifacts = qwen_prepare_config.load_manifest_artifacts(prepare)
  expected_model_path = qwen_prepare_config.cache_gguf_path(prepare)
  model = named_entry(inputs, "model", "qwen.inputs")
  if model["local_path"] != expected_model_path:
    raise ValueError("qwen model local_path must match the prepare GGUF path")
  if model["remote_path"] != expected_model_path:
    raise ValueError("qwen model remote_path must match the prepare GGUF path")
  if model["sha256"] != artifacts["gguf_sha256"]:
    raise ValueError("qwen model sha256 must match the checked manifest GGUF hash")
  expected_manifest_directory = prepare["manifest_directory"]
  for name in (
      "manifest_structured_with_target",
      "manifest_structured_without_target",
      "manifest_natural_text_only",
  ):
    entry = named_entry(inputs, name, "qwen.inputs")
    directory = str(PurePosixPath(entry["local_path"]).parent)
    if directory != expected_manifest_directory:
      raise ValueError(f"{name} must live in the checked manifest directory")
    if entry["local_path"] != entry["remote_path"]:
      raise ValueError(f"{name} local_path and remote_path must be identical")
  for name in ("frozen_corpus", "native_coverage"):
    entry = named_entry(inputs, name, "qwen.inputs")
    if entry["local_path"] != entry["remote_path"]:
      raise ValueError(f"{name} local_path and remote_path must be identical")
    if str(PurePosixPath(entry["local_path"]).parent) != expected_manifest_directory:
      raise ValueError(f"{name} must live in the checked fixture directory")

  runs = qwen["runs"]
  _validate_named_entries(
      runs,
      ("qwen-base-run1", "qwen-base-run2"),
      ("name", "output_binary", "output_textproto"),
      "qwen.runs",
  )
  output_paths = []
  for run in runs:
    _require_relative_path(
        run["output_binary"], f"{run['name']}.output_binary"
    )
    _require_relative_path(
        run["output_textproto"], f"{run['name']}.output_textproto"
    )
    output_paths.extend((run["output_binary"], run["output_textproto"]))
  _require_relative_path(qwen["checksum_output"], "qwen.checksum_output")
  output_paths.append(qwen["checksum_output"])
  if len(set(output_paths)) != len(output_paths):
    raise ValueError("qwen output paths must be unique")

  logs = qwen["logs"]
  _require_keys(
      logs,
      ("directory", "setup_prefix", "score_prefix", "suffix"),
      "qwen.logs",
  )
  _require_relative_path(logs["directory"], "qwen.logs.directory")
  for field in ("setup_prefix", "score_prefix", "suffix"):
    _require_text(logs[field], f"qwen.logs.{field}")


def load_config(path):
  with open(path, encoding="utf-8") as file:
    config = json.load(file)
  validate_config(config)
  return config


def named_entry(entries, name, label):
  matches = [entry for entry in entries if entry["name"] == name]
  if len(matches) != 1:
    raise ValueError(f"{label} has no unique entry named {name}")
  return matches[0]


def print_usage():
  print(
      "Usage: hpc4_config.py <config> "
      "{validate|get|resource|source-configs|inputs|upload-inputs|runs|"
      "outputs|input|source-config} [arguments]",
      file=sys.stderr,
  )


def main():
  if len(sys.argv) < 3:
    print_usage()
    return 2
  config = load_config(sys.argv[1])
  command = sys.argv[2]

  if command == "validate" and len(sys.argv) == 3:
    return 0
  if command == "get" and len(sys.argv) == 4:
    value = config
    for key in sys.argv[3].split("."):
      value = value[key]
    if isinstance(value, (dict, list)):
      raise ValueError("get requires a scalar config value")
    print(value)
    return 0
  if command == "resource" and len(sys.argv) == 4:
    resource = config["resources"][sys.argv[3]]
    print(
        resource["partition"],
        resource["nodes"],
        resource["tasks"],
        resource["cpus"],
        resource["time"],
        resource["memory"],
        sep="\t",
    )
    return 0
  if command == "source-configs" and len(sys.argv) == 3:
    for entry in config["qwen"]["source_configs"]:
      print(entry["name"], entry["path"], entry["sha256"], sep="\t")
    return 0
  if command == "inputs" and len(sys.argv) == 3:
    for entry in config["qwen"]["inputs"]:
      print(
          entry["name"],
          entry["local_path"],
          entry["remote_path"],
          entry["sha256"],
          sep="\t",
      )
    return 0
  if command == "upload-inputs" and len(sys.argv) == 3:
    if not config["qwen"]["upload_input_names"]:
      raise ValueError(
          "Qwen upload of frozen, coverage, and manifests is disabled; "
          "run scripts/run.sh qwen-prepare"
      )
    for name in config["qwen"]["upload_input_names"]:
      entry = named_entry(config["qwen"]["inputs"], name, "qwen.inputs")
      print(
          entry["name"],
          entry["local_path"],
          entry["remote_path"],
          entry["sha256"],
          sep="\t",
      )
    return 0
  if command == "runs" and len(sys.argv) == 3:
    for run in config["qwen"]["runs"]:
      print(
          run["name"],
          run["output_binary"],
          run["output_textproto"],
          sep="\t",
      )
    return 0
  if command == "outputs" and len(sys.argv) == 3:
    for run in config["qwen"]["runs"]:
      print(f"{run['name']}-binary", run["output_binary"], sep="\t")
      print(f"{run['name']}-textproto", run["output_textproto"], sep="\t")
    print("sha256", config["qwen"]["checksum_output"], sep="\t")
    return 0
  if command == "input" and len(sys.argv) == 5:
    entry = named_entry(config["qwen"]["inputs"], sys.argv[3], "qwen.inputs")
    print(entry[sys.argv[4]])
    return 0
  if command == "source-config" and len(sys.argv) == 5:
    entry = named_entry(
        config["qwen"]["source_configs"],
        sys.argv[3],
        "qwen.source_configs",
    )
    print(entry[sys.argv[4]])
    return 0

  print_usage()
  return 2


if __name__ == "__main__":
  sys.stdout.reconfigure(newline="\n")
  sys.exit(main())
