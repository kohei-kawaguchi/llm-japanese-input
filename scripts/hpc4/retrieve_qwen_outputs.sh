#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HPC4_CONFIG_FILE="${HPC4_CONFIG_FILE:-scripts/config/hpc4.json}"
SETUP_JOB_ID=""
SCORE_JOB_ID=""
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc4/retrieve_qwen_outputs.sh \
    --setup-job-id <job-id> \
    --score-job-id <job-id> \
    [--hpc4-config-file <path>] \
    [--dry-run]

Retrieves both Slurm logs, both Qwen score runs, and their SHA256 manifest.
It then verifies the retrieved hashes and byte equality. No SCP operation is
performed with --dry-run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-job-id)
      SETUP_JOB_ID="${2:-}"; shift 2 ;;
    --score-job-id)
      SCORE_JOB_ID="${2:-}"; shift 2 ;;
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

if [[ ! "$SETUP_JOB_ID" =~ ^[0-9]+$ || ! "$SCORE_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Numeric setup and score job IDs are required." >&2
  exit 1
fi

command -v python >/dev/null
python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" validate

LOGIN_NODE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get login_node)"
REMOTE_REPO_DIR="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"
LOG_DIRECTORY="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.logs.directory)"
SETUP_LOG_PREFIX="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.logs.setup_prefix)"
SCORE_LOG_PREFIX="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.logs.score_prefix)"
LOG_SUFFIX="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.logs.suffix)"
CHECKSUM_OUTPUT="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get qwen.checksum_output)"

SETUP_LOG="$LOG_DIRECTORY/$SETUP_LOG_PREFIX$SETUP_JOB_ID$LOG_SUFFIX"
SCORE_LOG="$LOG_DIRECTORY/$SCORE_LOG_PREFIX$SCORE_JOB_ID$LOG_SUFFIX"

declare -a output_names=()
declare -a output_paths=()
while IFS=$'\t' read -r name path; do
  output_names+=("$name")
  output_paths+=("$path")
done < <(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" outputs)

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: $LOGIN_NODE:$REMOTE_REPO_DIR/$SETUP_LOG -> $SETUP_LOG"
  echo "DRY RUN: $LOGIN_NODE:$REMOTE_REPO_DIR/$SCORE_LOG -> $SCORE_LOG"
  for index in "${!output_names[@]}"; do
    echo "DRY RUN: $LOGIN_NODE:$REMOTE_REPO_DIR/${output_paths[$index]} -> ${output_paths[$index]}"
  done
  exit 0
fi

command -v scp >/dev/null
command -v sha256sum >/dev/null
command -v cmp >/dev/null

mkdir -p "$LOG_DIRECTORY"
scp -- "$LOGIN_NODE:$REMOTE_REPO_DIR/$SETUP_LOG" "$SETUP_LOG"
scp -- "$LOGIN_NODE:$REMOTE_REPO_DIR/$SCORE_LOG" "$SCORE_LOG"

for index in "${!output_names[@]}"; do
  mkdir -p "$(dirname "${output_paths[$index]}")"
  scp -- \
    "$LOGIN_NODE:$REMOTE_REPO_DIR/${output_paths[$index]}" \
    "${output_paths[$index]}"
done

sha256sum --check --strict "$CHECKSUM_OUTPUT"
cmp -- "${output_paths[0]}" "${output_paths[2]}"
cmp -- "${output_paths[1]}" "${output_paths[3]}"

echo "Retrieved and verified Qwen outputs and Slurm logs."
