#!/usr/bin/env python3

import hashlib
import os
import subprocess
import sys

_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
if _HELPER_DIR not in sys.path:
  sys.path.insert(0, _HELPER_DIR)

import windows_build_config


ROOT = os.path.abspath(os.path.join(_HELPER_DIR, "..", ".."))
DEFAULT_CONFIG = os.path.join("scripts", "config", "windows_build.json")


def _sha256(path):
  digest = hashlib.sha256()
  with open(path, "rb") as file:
    chunk = file.read(1024 * 1024)
    while chunk:
      digest.update(chunk)
      chunk = file.read(1024 * 1024)
  return digest.hexdigest()


def _require_sha256(path, expected):
  if _sha256(path) != expected:
    raise ValueError(f"SHA256 mismatch for {path}")


def _require_file(path, label):
  if not os.path.isfile(path):
    raise ValueError(f"{label} is missing: {path}")


def _require_directory(path, label):
  if not os.path.isdir(path):
    raise ValueError(f"{label} is missing: {path}")


def _ensure_bazelisk(url, expected_sha256, path):
  if os.path.isfile(path):
    _require_sha256(path, expected_sha256)
    return
  os.makedirs(os.path.dirname(path), exist_ok=True)
  partial = path + ".partial"
  if os.path.exists(partial):
    os.remove(partial)
  subprocess.run(
      ["curl", "--fail", "--location", "--output", partial, "--", url],
      check=True,
  )
  _require_sha256(partial, expected_sha256)
  os.replace(partial, path)
  _require_sha256(path, expected_sha256)


def _qt_path(config):
  return os.pathsep.join(
      [
          *config["qt_path_directories"],
          os.path.dirname(config["python"]),
      ]
  )


def _commands(python, bazelisk):
  return [
      [python, "build_tools/update_deps.py"],
      [
          python,
          "build_tools/build_qt.py",
          "--release",
          "--confirm_license",
      ],
      [bazelisk, "build", "package", "--config", "release_build"],
  ]


def _print_plan(python, bazelisk, qt_path, commands):
  print(f"python: {python}")
  print(f"bazelisk: {bazelisk}")
  print(f"cwd: {os.path.join(ROOT, 'src')}")
  print(f"qt_path: {qt_path}")
  for command in commands:
    print(" ".join(command))


def _parse_args(argv):
  config_path = DEFAULT_CONFIG
  dry_run = False
  index = 0
  while index < len(argv):
    arg = argv[index]
    if arg == "--config":
      if index + 1 >= len(argv):
        raise ValueError("--config requires a value")
      config_path = argv[index + 1]
      index += 2
    elif arg == "--dry-run":
      dry_run = True
      index += 1
    elif arg in ("--help", "-h"):
      return None
    else:
      raise ValueError(f"Unknown argument: {arg}")
  return config_path, dry_run


def print_usage():
  print(
      "Usage: build_ime.py [--config <path>] [--dry-run]",
      file=sys.stderr,
  )


def main():
  parsed = _parse_args(sys.argv[1:])
  if parsed is None:
    print_usage()
    return 0
  config_path, dry_run = parsed
  config_path = windows_build_config.resolve_tool_path(ROOT, config_path)
  config = windows_build_config.load_config(config_path)
  python = config["python"]
  bazelisk = windows_build_config.resolve_tool_path(
      ROOT, config["bazelisk"]["path"]
  )
  _require_file(python, "python")
  for directory in config["qt_path_directories"]:
    _require_directory(directory, "qt_path_directories")
  commands = _commands(python, bazelisk)
  qt_path = _qt_path(config)
  if dry_run:
    _print_plan(python, bazelisk, qt_path, commands)
    return 0
  _ensure_bazelisk(
      config["bazelisk"]["url"],
      config["bazelisk"]["sha256"],
      bazelisk,
  )
  src = os.path.join(ROOT, "src")
  subprocess.run(commands[0], cwd=src, check=True)
  qt_env = os.environ.copy()
  qt_env["PATH"] = qt_path
  subprocess.run(commands[1], cwd=src, env=qt_env, check=True)
  subprocess.run(commands[2], cwd=src, check=True)
  return 0


if __name__ == "__main__":
  sys.stdout.reconfigure(newline="\n")
  sys.exit(main())
