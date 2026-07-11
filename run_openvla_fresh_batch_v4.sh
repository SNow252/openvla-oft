#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/openvla-oft

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/LIBERO:$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

export OPENVLA_CKPT=/home/asd/projects/vla_lang_generalization/hf_models/openvla-7b-oft-finetuned-libero-spatial

# Old dump is only used to recover the exact custom language strings.
export OLD_DUMP_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/openvla_mismatch_v2_dumps_20260707_220151

export FRESH_ROOT=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/openvla_multicond_10seeds_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FRESH_ROOT/runs"

echo "[INFO] FRESH_ROOT=$FRESH_ROOT"

get_old_lang() {
  local cond="$1"
  python - "$OLD_DUMP_DIR" "$cond" <<'PY'
import sys, json
from pathlib import Path

root = Path(sys.argv[1])
cond = sys.argv[2]

for p in sorted(root.glob("*.jsonl")):
    try:
        with open(p) as f:
            meta = json.loads(f.readline())
        if meta.get("dump_condition", "") == cond:
            print(meta.get("custom_language", ""), end="")
            sys.exit(0)
    except Exception:
        pass

print("", end="")
sys.exit(0)
PY
}

run_one_condition() {
  local cond="$1"
  local task_id="$2"
  local use_custom="$3"

  export OPENVLA_TASK_IDS="$task_id"

  if [[ "$use_custom" == "1" ]]; then
    local lang
    lang="$(get_old_lang "$cond")"
    export OPENVLA_CUSTOM_LANGUAGE="$lang"
    echo "[INFO] condition=$cond task_id=$task_id custom_language=[$OPENVLA_CUSTOM_LANGUAGE]"
  else
    unset OPENVLA_CUSTOM_LANGUAGE
    echo "[INFO] condition=$cond task_id=$task_id using native task language"
  fi

  local out_dir="$FRESH_ROOT/runs/$cond"
  mkdir -p "$out_dir"

  python experiments/robot/libero/run_libero_eval_fresh_dump_v4.py \
    --model_family openvla \
    --pretrained_checkpoint "$OPENVLA_CKPT" \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --num_open_loop_steps 8 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --center_crop True \
    --lora_rank 32 \
    --load_in_8bit False \
    --load_in_4bit False \
    --env_img_res 256 \
    --fresh_dump_dir "$out_dir" \
    --fresh_policy_name openvla_oft \
    --fresh_condition "$cond" \
    2>&1 | tee "$out_dir/run.log"
}

# task8-family conditions
run_one_condition task8_original 8 0
run_one_condition task8_empty_language 8 1
run_one_condition task8_nonsense_language 8 1
run_one_condition task8_unrelated_language 8 1
run_one_condition task8_wrong_source_language_only 8 1
run_one_condition task8_wrong_target_language_only 8 1
run_one_condition task8_wrong_source_wrong_target_language_only 8 1

# positive-control native task
run_one_condition task1_native_next_to_ramekin 1 0

# Merge all policy chunks into one replay directory.
export MERGED_CHUNK_DIR="$FRESH_ROOT/policy_chunks_merged"
mkdir -p "$MERGED_CHUNK_DIR"

python - "$FRESH_ROOT" "$MERGED_CHUNK_DIR" <<'PY'
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
merged = Path(sys.argv[2])
merged.mkdir(parents=True, exist_ok=True)

files = sorted((root / "runs").glob("*/policy_chunks/init_*/*.npz"))
print("[INFO] merging npz files:", len(files))

for src in files:
    init_dir = src.parent.name
    dst_dir = merged / init_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Use symlink to avoid copying large files.
    os.symlink(src, dst)

print("[INFO] merged dir:", merged)
for d in sorted(merged.glob("init_*")):
    print(d.name, len(list(d.glob("*.npz"))))
PY

echo "[INFO] Batch finished."
echo "[INFO] FRESH_ROOT=$FRESH_ROOT"
echo "[INFO] MERGED_CHUNK_DIR=$MERGED_CHUNK_DIR"