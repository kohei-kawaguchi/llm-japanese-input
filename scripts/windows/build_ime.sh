#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="scripts/config/windows_build.json"
DRY_RUN=()

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/windows/build_ime.sh [--config <path>] [--dry-run]

python, bazelisk, and the Qt build PATH are read from the device config.
The default config is scripts/config/windows_build.json.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=(--dry-run); shift ;;
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

python scripts/windows/windows_build_config.py "$CONFIG" validate
PYTHON="$(python scripts/windows/windows_build_config.py "$CONFIG" get python)"
test -f "$PYTHON"
exec "$PYTHON" scripts/windows/build_ime.py --config "$CONFIG" "${DRY_RUN[@]}"
