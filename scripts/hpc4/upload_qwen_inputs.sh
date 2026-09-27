#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HPC4_CONFIG_FILE="${HPC4_CONFIG_FILE:-scripts/config/hpc4.json}"
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc4/upload_qwen_inputs.sh \
    [--hpc4-config-file <path>] \
    [--dry-run]

Uploads only the optional prepared GGUF. Frozen corpus, native coverage, and
manifests are repository-checked and are obtained by branch sync plus
qwen-prepare, not by copying a local .tools tree.
No SSH or SCP operation is performed with --dry-run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hpc4-config-file)
      HPC4_CONFIG_FILE="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
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

command -v python >/dev/null
python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" validate

LOGIN_NODE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get login_node)"
REMOTE_REPO_DIR="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"

declare -a input_names=()
declare -a local_paths=()
declare -a remote_paths=()
declare -a expected_hashes=()
while IFS=$'\t' read -r name local_path remote_path expected_sha256; do
  input_names+=("$name")
  local_paths+=("$local_path")
  remote_paths+=("$remote_path")
  expected_hashes+=("$expected_sha256")
done < <(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" upload-inputs)

for index in "${!input_names[@]}"; do
  if [[ ! -f "${local_paths[$index]}" ]]; then
    echo "ERROR: Missing ${input_names[$index]} input: ${local_paths[$index]}" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  for index in "${!input_names[@]}"; do
    echo "DRY RUN: ${local_paths[$index]} -> $LOGIN_NODE:$REMOTE_REPO_DIR/${remote_paths[$index]}"
  done
  exit 0
fi

command -v sha256sum >/dev/null
command -v ssh >/dev/null
command -v scp >/dev/null

for index in "${!input_names[@]}"; do
  actual_output="$(sha256sum "${local_paths[$index]}")"
  actual_sha256="${actual_output%% *}"
  if [[ "$actual_sha256" != "${expected_hashes[$index]}" ]]; then
    echo "ERROR: SHA256 mismatch for ${local_paths[$index]}" >&2
    exit 1
  fi
done

declare -a remote_directories=()
for remote_path in "${remote_paths[@]}"; do
  remote_directories+=("$(dirname "$remote_path")")
done

ssh "$LOGIN_NODE" bash -s -- "$REMOTE_REPO_DIR" "${remote_directories[@]}" <<'REMOTE_EOF'
set -euo pipefail

repo_dir="$1"
shift
for directory in "$@"; do
  mkdir -p "$repo_dir/$directory"
done
REMOTE_EOF

for index in "${!input_names[@]}"; do
  scp -- \
    "${local_paths[$index]}" \
    "$LOGIN_NODE:$REMOTE_REPO_DIR/${remote_paths[$index]}"
done

declare -a remote_checks=()
for index in "${!input_names[@]}"; do
  remote_checks+=("${expected_hashes[$index]}" "${remote_paths[$index]}")
done

ssh "$LOGIN_NODE" bash -s -- "$REMOTE_REPO_DIR" "${remote_checks[@]}" <<'REMOTE_EOF'
set -euo pipefail

repo_dir="$1"
shift
command -v sha256sum >/dev/null
while [[ $# -gt 0 ]]; do
  expected_sha256="$1"
  relative_path="$2"
  shift 2
  actual_output="$(sha256sum "$repo_dir/$relative_path")"
  actual_sha256="${actual_output%% *}"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "ERROR: Remote SHA256 mismatch for $relative_path" >&2
    exit 1
  fi
done
REMOTE_EOF

echo "Uploaded and verified ${#input_names[@]} Qwen inputs."
