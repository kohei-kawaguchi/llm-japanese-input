#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/run.sh {qwen-prepare|hpc4-sync|qwen-hpc4-upload|qwen-hpc4-submit|qwen-hpc4-retrieve} [arguments]" >&2
  exit 2
fi

stage="$1"
shift

case "$stage" in
  qwen-prepare)
    exec bash scripts/prepare_qwen_model.sh "$@"
    ;;
  hpc4-sync)
    exec bash scripts/hpc4/sync_branch.sh "$@"
    ;;
  qwen-hpc4-upload)
    exec bash scripts/hpc4/upload_qwen_inputs.sh "$@"
    ;;
  qwen-hpc4-submit)
    exec bash scripts/hpc4/submit_qwen_scores.sh "$@"
    ;;
  qwen-hpc4-retrieve)
    exec bash scripts/hpc4/retrieve_qwen_outputs.sh "$@"
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    echo "Expected one of: qwen-prepare, hpc4-sync, qwen-hpc4-upload, qwen-hpc4-submit, qwen-hpc4-retrieve" >&2
    exit 2
    ;;
esac
