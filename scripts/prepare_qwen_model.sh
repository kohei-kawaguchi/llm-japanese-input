#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

QWEN_PREPARE_CONFIG="${QWEN_PREPARE_CONFIG:-scripts/config/qwen_prepare.json}"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_qwen_model.sh \
    [--prepare-config <path>]

Reads pinned Qwen identities from the prepare config and checked manifests.
If the cache GGUF already matches the manifest SHA256, the command succeeds.
Otherwise it downloads the pinned llama.cpp converter and Hugging Face
revision, converts F16, verifies every configured SHA256, and writes the GGUF
only to the cache model directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-config)
      QWEN_PREPARE_CONFIG="${2:-}"; shift 2 ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null; then
    echo "ERROR: Required command is missing: $name" >&2
    exit 1
  fi
}

require_command python
require_command curl
require_command tar
require_command sha256sum

python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" validate

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

download_file() {
  local url="$1"
  local path="$2"
  mkdir -p "$(dirname "$path")"
  curl --fail --location --output "$path" -- "$url"
}

resolve_venv_python() {
  local venv="$1"
  if [[ -x "$venv/bin/python" ]]; then
    printf '%s\n' "$venv/bin/python"
    return
  fi
  if [[ -x "$venv/Scripts/python.exe" ]]; then
    printf '%s\n' "$venv/Scripts/python.exe"
    return
  fi
  echo "ERROR: venv python is missing: $venv" >&2
  exit 1
}

gguf_path="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" gguf-path)"
expected_gguf_sha256="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" manifest-field gguf_sha256)"
if [[ -f "$gguf_path" ]]; then
  verify_sha256 "$gguf_path" "$expected_gguf_sha256"
  echo "Qwen F16 GGUF already matches the pinned SHA256."
  exit 0
fi

cache_model_directory="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get cache_model_directory)"
prepare_work_directory="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get prepare_work_directory)"
venv_directory="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get venv_directory)"
requirements_file="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get requirements_file)"
converter_script_name="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get converter_script)"
outtype="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get outtype)"
source_weight_filename="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get source_weight_filename)"
tokenizer_filename="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" get tokenizer_filename)"
expected_source_sha256="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" manifest-field source_weight_sha256)"
expected_tokenizer_sha256="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" manifest-field tokenizer_sha256)"
expected_archive_sha256="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" manifest-field runtime_source_archive_sha256)"
runtime_revision="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" manifest-field runtime_revision)"
converter_url="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" converter-url)"
converter_prefix="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" converter-prefix)"

mkdir -p "$cache_model_directory" "$prepare_work_directory"

declare -a source_names=()
declare -a source_paths=()
while IFS=$'\t' read -r filename path; do
  source_names+=("$filename")
  source_paths+=("$path")
done < <(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" source-files)

for index in "${!source_names[@]}"; do
  if [[ ! -f "${source_paths[$index]}" ]]; then
    source_url="$(python scripts/hpc4/qwen_prepare_config.py "$QWEN_PREPARE_CONFIG" source-url "${source_names[$index]}")"
    download_file "$source_url" "${source_paths[$index]}"
  fi
done

verify_sha256 "$cache_model_directory/$source_weight_filename" "$expected_source_sha256"
verify_sha256 "$cache_model_directory/$tokenizer_filename" "$expected_tokenizer_sha256"

archive_path="$prepare_work_directory/llama.cpp-$runtime_revision.tar.gz"
if [[ ! -f "$archive_path" ]]; then
  download_file "$converter_url" "$archive_path"
fi
verify_sha256 "$archive_path" "$expected_archive_sha256"

extracted_root="$prepare_work_directory/$converter_prefix"
if [[ ! -d "$extracted_root" ]]; then
  tar -xzf "$archive_path" -C "$prepare_work_directory"
fi
converter_script="$extracted_root/$converter_script_name"
if [[ ! -f "$converter_script" ]]; then
  echo "ERROR: Converter script is missing: $converter_script" >&2
  exit 1
fi

if [[ ! -d "$venv_directory" ]]; then
  python -m venv "$venv_directory"
fi
venv_python="$(resolve_venv_python "$venv_directory")"
"$venv_python" -m pip install --requirement "$requirements_file"

partial_gguf_path="$gguf_path.partial"
rm -f -- "$partial_gguf_path"
"$venv_python" "$converter_script" \
  "$cache_model_directory" \
  --outtype "$outtype" \
  --outfile "$partial_gguf_path"
verify_sha256 "$partial_gguf_path" "$expected_gguf_sha256"
mv -- "$partial_gguf_path" "$gguf_path"
verify_sha256 "$gguf_path" "$expected_gguf_sha256"

echo "Prepared Qwen F16 GGUF at $gguf_path."
