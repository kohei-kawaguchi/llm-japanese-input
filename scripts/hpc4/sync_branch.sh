#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HPC4_CONFIG_FILE="${HPC4_CONFIG_FILE:-scripts/config/hpc4.json}"
BRANCH_OVERRIDE=""
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc4/sync_branch.sh \
    [--hpc4-config-file <path>] \
    [--branch <name>] \
    [--dry-run]

The branch must be committed and match its origin/<branch> tracking ref.
This command performs SSH, fetch, switch, and fast-forward pull operations
unless --dry-run is supplied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hpc4-config-file)
      HPC4_CONFIG_FILE="${2:-}"; shift 2 ;;
    --branch)
      BRANCH_OVERRIDE="${2:-}"; shift 2 ;;
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
command -v git >/dev/null
python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" validate

LOGIN_NODE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get login_node)"
REMOTE_REPO_DIR="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Commit all changes before syncing the HPC4 branch." >&2
  exit 1
fi

for required_file in \
  scripts/run.sh \
  scripts/prepare_qwen_model.sh \
  scripts/config/hpc4.json \
  scripts/config/qwen_prepare.json \
  scripts/config/qwen_prepare_requirements.txt \
  scripts/hpc4/hpc4_config.py \
  scripts/hpc4/qwen_prepare_config.py \
  scripts/hpc4/sync_branch.sh \
  scripts/hpc4/upload_qwen_inputs.sh \
  scripts/hpc4/submit_qwen_scores.sh \
  scripts/hpc4/retrieve_qwen_outputs.sh \
  scripts/hpc4/qwen_setup_job.sbatch \
  scripts/hpc4/qwen_score_job.sbatch \
  src/engine/evaluation/checked/quality_regression_development_frozen_corpus.pb \
  src/engine/evaluation/checked/quality_regression_development_native_coverage.pb \
  src/engine/evaluation/checked/manifest-f16-structured-with-target-v1.textproto \
  src/engine/evaluation/checked/manifest-f16-structured-without-target-v1.textproto \
  src/engine/evaluation/checked/manifest-f16-natural-text-only-v1.textproto; do
  if ! git ls-files --error-unmatch "$required_file" >/dev/null 2>&1; then
    echo "ERROR: HPC4 workflow file is not committed: $required_file" >&2
    exit 1
  fi
done

local_branch="$BRANCH_OVERRIDE"
if [[ -z "$local_branch" ]]; then
  local_branch="$(git rev-parse --abbrev-ref HEAD)"
fi
if [[ "$local_branch" == "HEAD" ]]; then
  echo "ERROR: Cannot sync a detached HEAD." >&2
  exit 1
fi

local_commit="$(git rev-parse "refs/heads/$local_branch")"
origin_commit="$(git rev-parse "refs/remotes/origin/$local_branch")"
if [[ "$local_commit" != "$origin_commit" ]]; then
  echo "ERROR: Local $local_branch does not match origin/$local_branch; push the committed branch first." >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: sync origin/$local_branch to $LOGIN_NODE:$REMOTE_REPO_DIR"
  exit 0
fi

command -v ssh >/dev/null
ssh "$LOGIN_NODE" bash -s -- "$REMOTE_REPO_DIR" "$local_branch" <<'REMOTE_EOF'
set -euo pipefail

repo_dir="$1"
branch="$2"

cd "$repo_dir"
git fetch origin --prune
git show-ref --verify --quiet "refs/remotes/origin/$branch"

if git show-ref --verify --quiet "refs/heads/$branch"; then
  git merge-base --is-ancestor "$branch" "origin/$branch"
  git switch "$branch"
else
  git switch -c "$branch" --track "origin/$branch"
fi

git pull --ff-only origin "$branch"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$branch")"
REMOTE_EOF

echo "Synced origin/$local_branch on HPC4."
