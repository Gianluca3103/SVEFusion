#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: bash tools/scripts/eval_vod_comparison.sh SVE_ROOT EXPORT_ROOT CHECKPOINT PYTHON" >&2
  exit 2
fi
SVE=$(realpath "$1")
EXPORT=$(realpath "$2")
CHECKPOINT=$(realpath "$3")
PYTHON=$4
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
cd "$SVE/tools"
for condition in clean faulty reconstructed; do
  [[ -f "$EXPORT/lidar/$condition/vod_infos_val.pkl" ]] || {
    echo "Missing $condition validation infos" >&2; exit 1;
  }
  "$PYTHON" test.py --cfg_file "cfgs/VoD_models/SVEFusion_vod_${condition}.yaml" \
    --ckpt "$CHECKPOINT" --batch_size 1 --workers 2 \
    --extra_tag vod_clean_trained --eval_tag "$condition" --save_to_file
done
