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

export PATH="$HOME/.local/bin:$PATH"
require_command python
python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" validate
compiler_module="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get environment.compiler_module)"

if ! type module >/dev/null; then
  echo "ERROR: Required command is missing: module" >&2
  exit 1
fi
module purge
module load "$compiler_module"

require_command python
require_command sha256sum
require_command cmp

repo_dir="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"
executable="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.build.executable)"
execution_config="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" source-config execution_config path)"
score_config="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" source-config score_config path)"
frozen_corpus="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input frozen_corpus remote_path)"
native_coverage="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input native_coverage remote_path)"
first_manifest="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input manifest_structured_with_target remote_path)"
second_manifest="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input manifest_structured_without_target remote_path)"
third_manifest="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input manifest_natural_text_only remote_path)"
model="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" input model remote_path)"
checksum_output="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.checksum_output)"

cd "$repo_dir"

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

while IFS=$'\t' read -r name path expected_sha256; do
  verify_sha256 "$path" "$expected_sha256"
done < <(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" source-configs)

while IFS=$'\t' read -r name local_path remote_path expected_sha256; do
  verify_sha256 "$remote_path" "$expected_sha256"
done < <(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" inputs)

manifest_directory="$(dirname "$first_manifest")"
model_directory="$(dirname "$model")"
test "$(dirname "$second_manifest")" = "$manifest_directory"
test "$(dirname "$third_manifest")" = "$manifest_directory"
test "$model_directory" != "$manifest_directory"
test -x "$executable"

declare -a run_names=()
declare -a output_binaries=()
declare -a output_textprotos=()
while IFS=$'\t' read -r name output_binary output_textproto; do
  run_names+=("$name")
  output_binaries+=("$output_binary")
  output_textprotos+=("$output_textproto")
done < <(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" runs)

for index in "${!run_names[@]}"; do
  mkdir -p \
    "$(dirname "${output_binaries[$index]}")" \
    "$(dirname "${output_textprotos[$index]}")"
  if [[ -e "${output_binaries[$index]}" || -e "${output_textprotos[$index]}" ]]; then
    echo "ERROR: Qwen output already exists for ${run_names[$index]}." >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$checksum_output")"
if [[ -e "$checksum_output" ]]; then
  echo "ERROR: Qwen checksum output already exists: $checksum_output" >&2
  exit 1
fi

for index in "${!run_names[@]}"; do
  "$executable" \
    "--score_config=$score_config" \
    "--execution_config=$execution_config" \
    "--frozen_corpus=$frozen_corpus" \
    "--native_coverage=$native_coverage" \
    "--manifest_directory=$manifest_directory" \
    "--model_directory=$model_directory" \
    "--output_binary=${output_binaries[$index]}" \
    "--output_textproto=${output_textprotos[$index]}"
done

cmp -- "${output_binaries[0]}" "${output_binaries[1]}"
cmp -- "${output_textprotos[0]}" "${output_textprotos[1]}"

sha256sum --binary \
  "${output_binaries[0]}" \
  "${output_binaries[1]}" \
  "${output_textprotos[0]}" \
  "${output_textprotos[1]}" \
  > "$checksum_output"
sha256sum --check --strict "$checksum_output"
