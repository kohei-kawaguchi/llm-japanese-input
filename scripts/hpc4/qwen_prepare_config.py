#!/usr/bin/env python3

import hashlib
import json
import re
import sys
from pathlib import PurePosixPath


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TEXTPROTO_STRING_PATTERN = re.compile(r'^([A-Za-z0-9_]+):\s+"(.*)"\s*$')

MANIFEST_ARTIFACT_FIELDS = (
    "source_model",
    "source_revision",
    "source_weight_sha256",
    "tokenizer_sha256",
    "gguf_file_name",
    "gguf_sha256",
    "quantization",
    "runtime_name",
    "runtime_revision",
    "runtime_source_archive_sha256",
)

MANIFEST_HASH_FIELDS = (
    "source_weight_sha256",
    "tokenizer_sha256",
    "gguf_sha256",
    "runtime_source_archive_sha256",
)


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


def _require_string_list(value, label, allow_empty=False):
  if not isinstance(value, list) or (not value and not allow_empty):
    raise ValueError(f"{label} must be a non-empty list")
  items = []
  for index, item in enumerate(value):
    _require_text(item, f"{label}[{index}]")
    items.append(item)
  if len(set(items)) != len(items):
    raise ValueError(f"{label} values must be unique")
  return items


def _join_relative(directory, filename, label):
  _require_relative_path(directory, f"{label} directory")
  _require_text(filename, f"{label} filename")
  if "/" in filename or "\\" in filename:
    raise ValueError(f"{label} filename must be a simple name")
  return f"{directory}/{filename}"


def validate_config(config):
  _require_keys(
      config,
      (
          "model_id",
          "revision",
          "license_identifier",
          "forbidden_model_id_substrings",
          "allow_patterns",
          "source_weight_filename",
          "tokenizer_filename",
          "gguf_filename",
          "quantization",
          "outtype",
          "runtime_name",
          "converter_script",
          "converter_archive_url_template",
          "converter_strip_prefix_template",
          "huggingface_file_url_template",
          "cache_model_directory",
          "prepare_work_directory",
          "venv_directory",
          "manifest_directory",
          "manifest_filenames",
          "requirements_file",
      ),
      "Qwen prepare config",
  )
  for field in (
      "model_id",
      "revision",
      "license_identifier",
      "source_weight_filename",
      "tokenizer_filename",
      "gguf_filename",
      "quantization",
      "outtype",
      "runtime_name",
      "converter_script",
      "converter_archive_url_template",
      "converter_strip_prefix_template",
      "huggingface_file_url_template",
  ):
    _require_text(config[field], field)
  for field in (
      "cache_model_directory",
      "prepare_work_directory",
      "venv_directory",
      "manifest_directory",
      "requirements_file",
  ):
    _require_relative_path(config[field], field)
  forbidden = _require_string_list(
      config["forbidden_model_id_substrings"],
      "forbidden_model_id_substrings",
  )
  allow_patterns = _require_string_list(config["allow_patterns"], "allow_patterns")
  for pattern in allow_patterns:
    if "/" in pattern or "\\" in pattern:
      raise ValueError("allow_patterns entries must be simple names")
  if config["source_weight_filename"] not in allow_patterns:
    raise ValueError("source_weight_filename must be listed in allow_patterns")
  if config["tokenizer_filename"] not in allow_patterns:
    raise ValueError("tokenizer_filename must be listed in allow_patterns")
  _require_string_list(config["manifest_filenames"], "manifest_filenames")
  for filename in (
      config["source_weight_filename"],
      config["tokenizer_filename"],
      config["gguf_filename"],
      config["converter_script"],
  ):
    if "/" in filename or "\\" in filename:
      raise ValueError(f"{filename} must be a simple name")
  if "{revision}" not in config["converter_archive_url_template"]:
    raise ValueError("converter_archive_url_template must contain {revision}")
  if "{revision}" not in config["converter_strip_prefix_template"]:
    raise ValueError("converter_strip_prefix_template must contain {revision}")
  for placeholder in ("{model_id}", "{revision}", "{filename}"):
    if placeholder not in config["huggingface_file_url_template"]:
      raise ValueError(
          "huggingface_file_url_template must contain {model_id}, "
          "{revision}, and {filename}"
      )
  for substring in forbidden:
    if substring in config["model_id"]:
      raise ValueError(f"model_id must not contain {substring}")
  if config["quantization"] != "F16" or config["outtype"] != "f16":
    raise ValueError("prepare outtype must be F16")


def load_config(path):
  with open(path, encoding="utf-8") as file:
    config = json.load(file)
  validate_config(config)
  return config


def read_manifest_artifacts(path):
  artifacts = {}
  with open(path, encoding="utf-8", newline="\n") as file:
    for raw_line in file:
      line = raw_line.strip()
      match = TEXTPROTO_STRING_PATTERN.fullmatch(line)
      if match is None:
        continue
      key, value = match.group(1), match.group(2)
      if key not in MANIFEST_ARTIFACT_FIELDS:
        continue
      if key in artifacts:
        raise ValueError(f"{path} repeats artifact field {key}")
      artifacts[key] = value
  missing = [field for field in MANIFEST_ARTIFACT_FIELDS if field not in artifacts]
  if missing:
    raise ValueError(f"{path} is missing artifact fields: {', '.join(missing)}")
  for field in MANIFEST_HASH_FIELDS:
    _require_sha256(artifacts[field], f"{path} {field}")
  for field in MANIFEST_ARTIFACT_FIELDS:
    _require_text(artifacts[field], f"{path} {field}")
  return artifacts


def load_manifest_artifacts(config):
  artifacts = None
  for filename in config["manifest_filenames"]:
    path = _join_relative(config["manifest_directory"], filename, "manifest")
    current = read_manifest_artifacts(path)
    if artifacts is None:
      artifacts = current
      continue
    if current != artifacts:
      raise ValueError(f"{path} artifact identity does not match the other manifests")
  if artifacts["source_model"] != config["model_id"]:
    raise ValueError("manifest source_model does not match prepare model_id")
  if artifacts["source_revision"] != config["revision"]:
    raise ValueError("manifest source_revision does not match prepare revision")
  if artifacts["gguf_file_name"] != config["gguf_filename"]:
    raise ValueError("manifest gguf_file_name does not match prepare gguf_filename")
  if artifacts["quantization"] != config["quantization"]:
    raise ValueError("manifest quantization does not match prepare quantization")
  if artifacts["runtime_name"] != config["runtime_name"]:
    raise ValueError("manifest runtime_name does not match prepare runtime_name")
  return artifacts


def cache_gguf_path(config):
  return _join_relative(
      config["cache_model_directory"],
      config["gguf_filename"],
      "GGUF",
  )


def source_file_path(config, filename):
  return _join_relative(config["cache_model_directory"], filename, "source file")


def converter_archive_url(config, artifacts):
  return config["converter_archive_url_template"].format(
      revision=artifacts["runtime_revision"]
  )


def converter_strip_prefix(config, artifacts):
  return config["converter_strip_prefix_template"].format(
      revision=artifacts["runtime_revision"]
  )


def huggingface_file_url(config, filename):
  return config["huggingface_file_url_template"].format(
      model_id=config["model_id"],
      revision=config["revision"],
      filename=filename,
  )


def file_sha256(path):
  digest = hashlib.sha256()
  with open(path, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def require_file_sha256(path, expected, label):
  _require_sha256(expected, f"{label} expected SHA256")
  actual = file_sha256(path)
  if actual != expected:
    raise ValueError(f"SHA256 mismatch for {path}")
  return actual


def print_usage():
  print(
      "Usage: qwen_prepare_config.py <config> "
      "{validate|get|manifest-field|gguf-path|converter-url|"
      "converter-prefix|source-files|source-url} [arguments]",
      file=sys.stderr,
  )


def main():
  if len(sys.argv) < 3:
    print_usage()
    return 2
  config = load_config(sys.argv[1])
  artifacts = load_manifest_artifacts(config)
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
  if command == "manifest-field" and len(sys.argv) == 4:
    print(artifacts[sys.argv[3]])
    return 0
  if command == "gguf-path" and len(sys.argv) == 3:
    print(cache_gguf_path(config))
    return 0
  if command == "converter-url" and len(sys.argv) == 3:
    print(converter_archive_url(config, artifacts))
    return 0
  if command == "converter-prefix" and len(sys.argv) == 3:
    print(converter_strip_prefix(config, artifacts))
    return 0
  if command == "source-files" and len(sys.argv) == 3:
    for filename in config["allow_patterns"]:
      print(filename, source_file_path(config, filename), sep="\t")
    return 0
  if command == "source-url" and len(sys.argv) == 4:
    print(huggingface_file_url(config, sys.argv[3]))
    return 0

  print_usage()
  return 2


if __name__ == "__main__":
  sys.stdout.reconfigure(newline="\n")
  sys.exit(main())
