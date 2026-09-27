#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HPC4_CONFIG_FILE="${HPC4_CONFIG_FILE:-scripts/config/hpc4.json}"
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc4/submit_qwen_scores.sh \
    [--hpc4-config-file <path>] \
    [--dry-run]

Submits the Qwen setup job and one dependent score job. The setup job runs
qwen-prepare and the opt score-binary build. The score job runs
qwen-base-run1 and qwen-base-run2 sequentially on one allocation.
Run branch sync before this command. GGUF upload is optional because
prepare rebuilds the cache GGUF on the cluster.
No SSH or Slurm submission is performed with --dry-run.
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

for required_file in \
  scripts/hpc4/qwen_setup_job.sbatch \
  scripts/hpc4/qwen_score_job.sbatch; do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: Required job script is missing: $required_file" >&2
    exit 1
  fi
done

LOGIN_NODE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get login_node)"
REMOTE_REPO_DIR="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get repo_dir)"
ACCOUNT="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" get account)"
SETUP_RESOURCE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" resource qwen_setup)"
SCORE_RESOURCE="$(python scripts/hpc4/hpc4_config.py "$HPC4_CONFIG_FILE" resource qwen_score)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: submit qwen_setup on $LOGIN_NODE:$REMOTE_REPO_DIR"
  echo "DRY RUN: account=$ACCOUNT resource=$SETUP_RESOURCE"
  echo "DRY RUN: submit qwen_score with dependency afterok:<qwen_setup_job_id>"
  echo "DRY RUN: account=$ACCOUNT resource=$SCORE_RESOURCE"
  exit 0
fi

command -v ssh >/dev/null
ssh "$LOGIN_NODE" bash -s -- "$REMOTE_REPO_DIR" "$HPC4_CONFIG_FILE" <<'REMOTE_EOF'
set -euo pipefail

repo_dir="$1"
hpc4_config_file="$2"

cd "$repo_dir"
command -v python >/dev/null
command -v sbatch >/dev/null
python scripts/hpc4/hpc4_config.py "$hpc4_config_file" validate

account="$(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" get account)"
log_directory="$(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" get qwen.logs.directory)"
setup_log_prefix="$(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" get qwen.logs.setup_prefix)"
score_log_prefix="$(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" get qwen.logs.score_prefix)"
log_suffix="$(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" get qwen.logs.suffix)"
IFS=$'\t' read -r \
  setup_partition setup_nodes setup_tasks setup_cpus setup_time setup_memory \
  < <(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" resource qwen_setup)
IFS=$'\t' read -r \
  score_partition score_nodes score_tasks score_cpus score_time score_memory \
  < <(python scripts/hpc4/hpc4_config.py "$hpc4_config_file" resource qwen_score)

parse_job_id() {
  local output="$1"
  local job_id="${output%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch did not return a job ID: $output" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

setup_output="$(sbatch \
  --parsable \
  --job-name=qwen_setup \
  --partition="$setup_partition" \
  --account="$account" \
  --nodes="$setup_nodes" \
  --ntasks="$setup_tasks" \
  --cpus-per-task="$setup_cpus" \
  --time="$setup_time" \
  --mem="$setup_memory" \
  --chdir="$repo_dir" \
  --output="$repo_dir/$log_directory/${setup_log_prefix}%j${log_suffix}" \
  --export="ALL,HPC4_CONFIG_FILE=$hpc4_config_file" \
  scripts/hpc4/qwen_setup_job.sbatch)"
setup_job_id="$(parse_job_id "$setup_output")"
echo "Submitted Qwen setup job $setup_job_id."

score_output="$(sbatch \
  --parsable \
  --job-name=qwen_score \
  --partition="$score_partition" \
  --account="$account" \
  --nodes="$score_nodes" \
  --ntasks="$score_tasks" \
  --cpus-per-task="$score_cpus" \
  --time="$score_time" \
  --mem="$score_memory" \
  --chdir="$repo_dir" \
  --output="$repo_dir/$log_directory/${score_log_prefix}%j${log_suffix}" \
  --dependency="afterok:$setup_job_id" \
  --kill-on-invalid-dep=yes \
  --export="ALL,HPC4_CONFIG_FILE=$hpc4_config_file" \
  scripts/hpc4/qwen_score_job.sbatch)"
score_job_id="$(parse_job_id "$score_output")"

echo "Submitted dependent Qwen score job $score_job_id."
echo "Retrieve after completion with:"
echo "  bash scripts/hpc4/retrieve_qwen_outputs.sh --setup-job-id $setup_job_id --score-job-id $score_job_id"
REMOTE_EOF
