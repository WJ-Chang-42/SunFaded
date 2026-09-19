#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 2 ]; then
  printf 'Usage: bash train.sh DATA_DIRECTORY NEW_OUTPUT_DIRECTORY [additional train.py arguments]\n' >&2
  exit 2
fi
task_data="$1"
task_output="$2"
shift 2
exec python "$task_root/train.py" -s "$task_data" -m "$task_output" -r 1 --config "$task_root/config/training_config.json" --seed 0 "$@"
