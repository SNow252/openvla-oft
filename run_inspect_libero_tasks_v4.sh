#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/openvla-oft

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate openvla-oft || true
fi

export MUJOCO_GL=egl
export PYTHONPATH=$PWD/LIBERO:$PWD:$PYTHONPATH

export INSPECT_OUT_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/task_object_inspection_v4_$(date +%Y%m%d_%H%M%S)
mkdir -p "$INSPECT_OUT_DIR"

python tools/inspect_libero_task_objects_v4.py \
  --task_suite_name libero_spatial \
  --task_ids 1 3 8 9 \
  --init_indices 0 1 2 \
  --settle_steps 30 \
  --out_dir "$INSPECT_OUT_DIR" \
  2>&1 | tee "$INSPECT_OUT_DIR/inspect.log"

echo
echo "[INFO] Output dir:"
echo "$INSPECT_OUT_DIR"
echo
echo "[INFO] Markdown:"
cat "$INSPECT_OUT_DIR/task_object_inspection.md"