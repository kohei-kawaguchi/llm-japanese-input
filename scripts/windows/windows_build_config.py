#!/usr/bin/env python3

import json
import os
import re
import sys
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


def is_windows_absolute(value):
  return (
      len(value) >= 3
      and value[0].isalpha()
      and value[1] == ":"
      and value[2] == "/"
  )


def _require_windows_absolute_path(value, label):
  _require_text(value, label)
  if "\\" in value or not is_windows_absolute(value):
    raise ValueError(f"{label} must be an absolute Windows path")
  parts = value[3:].split("/")
  if any(part in ("", ".", "..") for part in parts):
    raise ValueError(f"{label} must be an absolute Windows path")


def _require_tool_path(value, label):
  _require_text(value, label)
  if "\\" in value:
    raise ValueError(f"{label} must use forward slashes")
  if is_windows_absolute(value):
    _require_windows_absolute_path(value, label)
    return
  path = PurePosixPath(value)
  if path.is_absolute() or any(part in (".", "..") for part in path.parts):
    raise ValueError(
        f"{label} must be an absolute Windows path or a repository-relative path"
    )


def _require_sha256(value, label):
  if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
    raise ValueError(f"{label} must be a lowercase SHA256")


def resolve_tool_path(root, value):
  if os.path.isabs(value) or is_windows_absolute(value):
    return value
  return os.path.join(root, *PurePosixPath(value).parts)


def validate_config(config):
  _require_keys(
      config,
      ("python", "bazelisk", "qt_path_directories"),
      "Windows build config",
  )
  _require_windows_absolute_path(config["python"], "python")
  _require_keys(
      config["bazelisk"],
      ("url", "sha256", "path"),
      "bazelisk",
  )
  _require_text(config["bazelisk"]["url"], "bazelisk.url")
  _require_sha256(config["bazelisk"]["sha256"], "bazelisk.sha256")
  _require_tool_path(config["bazelisk"]["path"], "bazelisk.path")
  directories = config["qt_path_directories"]
  if not isinstance(directories, list) or not directories:
    raise ValueError("qt_path_directories must be a non-empty list")
  for index, directory in enumerate(directories):
    _require_windows_absolute_path(
        directory, f"qt_path_directories[{index}]"
    )


def load_config(path):
  with open(path, encoding="utf-8") as file:
    config = json.load(file)
  validate_config(config)
  return config


def print_usage():
  print(
      "Usage: windows_build_config.py <config> {validate|get|qt-path} [arguments]",
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
  if command == "qt-path" and len(sys.argv) == 3:
    for directory in config["qt_path_directories"]:
      print(directory)
    return 0

  print_usage()
  return 2


if __name__ == "__main__":
  sys.stdout.reconfigure(newline="\n")
  sys.exit(main())
