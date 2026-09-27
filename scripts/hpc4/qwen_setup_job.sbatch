#!/usr/bin/env bash
set -euo pipefail

: "${HPC4_CONFIG_FILE:?HPC4_CONFIG_FILE is required}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null; then
    echo "ERROR: Required command is missing: $name" >&2
    exit 1
  fi
}

verify_sha256() {
  local path="$1"
  local expected_sha256="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file is missing: $path" >&2
    exit 1
  fi
  local actual_output
  local actual_sha256
  actual_output="$(sha256sum "$path")"
  actual_sha256="${actual_output%% *}"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "ERROR: SHA256 mismatch for $path" >&2
    exit 1
  fi
}

export PATH="$HOME/.local/bin:$PATH"
require_command python
require_command curl
require_command sha256sum
python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" validate
compiler_module="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get environment.compiler_module)"

if ! type module >/dev/null; then
  echo "ERROR: Required command is missing: module" >&2
  exit 1
fi
module purge
module load "$compiler_module"

require_command python
require_command gcc
require_command g++

repo_dir="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"
working_directory="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.build.working_directory)"
target="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.build.target)"
compilation_mode="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.build.compilation_mode)"
executable="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.build.executable)"
setup_cpus="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get resources.qwen_setup.cpus)"
bazelisk_url="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.bazelisk.url)"
bazelisk_sha256="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.bazelisk.sha256)"
bazelisk_path="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.bazelisk.path)"

cd "$repo_dir"
if [[ -f "$bazelisk_path" ]]; then
  verify_sha256 "$bazelisk_path" "$bazelisk_sha256"
else
  mkdir -p "$(dirname "$bazelisk_path")"
  partial_path="$bazelisk_path.partial"
  rm -f -- "$partial_path"
  curl --fail --location --output "$partial_path" -- "$bazelisk_url"
  verify_sha256 "$partial_path" "$bazelisk_sha256"
  chmod +x -- "$partial_path"
  mv -- "$partial_path" "$bazelisk_path"
  verify_sha256 "$bazelisk_path" "$bazelisk_sha256"
fi
test -x "$bazelisk_path"

bash scripts/run.sh qwen-prepare
cd "$repo_dir/$working_directory"
"$repo_dir/$bazelisk_path" build "$target" --jobs="$setup_cpus" --compilation_mode="$compilation_mode"
test -x "$repo_dir/$executable"
