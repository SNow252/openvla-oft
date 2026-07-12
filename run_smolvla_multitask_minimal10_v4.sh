#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/lerobot_code_snapshot

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate lerobot || true
fi

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export SMOLVLA_CKPT=/home/asd/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de

export FRESH_ROOT=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/smolvla_multitask_minimal10_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FRESH_ROOT/runs"

echo "[INFO] FRESH_ROOT=$FRESH_ROOT"

run_one_condition() {
  local cond="$1"
  local task_id="$2"
  local mode="$3"

  export SMOLVLA_FRESH_POLICY_NAME=smolvla
  export SMOLVLA_FRESH_CONDITION="$cond"
  export SMOLVLA_FRESH_TASK_ID="$task_id"
  export SMOLVLA_FRESH_START_INIT_STATE_IDX=0

  if [[ "$mode" == "native" ]]; then
    unset LIBERO_LANGUAGE_MODE
    unset LIBERO_CUSTOM_LANGUAGE
    unset LIBERO_CONSTANT_LANGUAGE
    export SMOLVLA_FRESH_LANGUAGE=""
    echo "[INFO] condition=$cond task_id=$task_id using native task language"
  elif [[ "$mode" == "empty" ]]; then
    export LIBERO_LANGUAGE_MODE=custom
    export LIBERO_CUSTOM_LANGUAGE=""
    export SMOLVLA_FRESH_LANGUAGE=""
    echo "[INFO] condition=$cond task_id=$task_id custom_language=[]"
  elif [[ "$mode" == "nonsense" ]]; then
    export LIBERO_LANGUAGE_MODE=custom
    export LIBERO_CUSTOM_LANGUAGE="dax blicket wug zorp"
    export SMOLVLA_FRESH_LANGUAGE="$LIBERO_CUSTOM_LANGUAGE"
    echo "[INFO] condition=$cond task_id=$task_id custom_language=[$LIBERO_CUSTOM_LANGUAGE]"
  else
    echo "[ERROR] unknown mode=$mode"
    exit 1
  fi

  local out_dir="$FRESH_ROOT/runs/$cond"
  mkdir -p "$out_dir"
  export SMOLVLA_FRESH_DUMP_DIR="$out_dir"

  lerobot-eval \
    --policy.type=smolvla \
    --policy.pretrained_path="$SMOLVLA_CKPT" \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids="[$task_id]" \
    --env.max_parallel_tasks=1 \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --eval.use_async_envs=False \
    --policy.device=cuda \
    --policy.use_amp=true \
    --job_name="smolvla_${cond}_fresh_minimal10" \
    2>&1 | tee "$out_dir/run.log"
}

run_one_condition task3_original 3 native
run_one_condition task3_empty_language 3 empty
run_one_condition task3_nonsense_language 3 nonsense

run_one_condition task9_original 9 native
run_one_condition task9_empty_language 9 empty
run_one_condition task9_nonsense_language 9 nonsense

export MERGED_CHUNK_DIR="$FRESH_ROOT/policy_chunks_merged"
mkdir -p "$MERGED_CHUNK_DIR"

python - "$FRESH_ROOT" "$MERGED_CHUNK_DIR" <<'PY'
import os
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

    os.symlink(src, dst)

print("[INFO] merged dir:", merged)
for d in sorted(merged.glob("init_*")):
    print(d.name, len(list(d.glob("*.npz"))))
PY

echo "[INFO] Batch finished."
echo "[INFO] FRESH_ROOT=$FRESH_ROOT"
echo "[INFO] MERGED_CHUNK_DIR=$MERGED_CHUNK_DIR"