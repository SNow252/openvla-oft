#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/lerobot_code_snapshot

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export SMOLVLA_CKPT=/home/asd/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de

export OLD_DUMP_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/openvla_mismatch_v2_dumps_20260707_220151

export FRESH_ROOT=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/smolvla_multicond_10seeds_$(date +%Y%m%d_%H%M%S)
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
PY
}

run_one_condition() {
  local cond="$1"
  local task_id="$2"
  local use_custom="$3"

  export SMOLVLA_FRESH_POLICY_NAME=smolvla
  export SMOLVLA_FRESH_CONDITION="$cond"
  export SMOLVLA_FRESH_TASK_ID="$task_id"
  export SMOLVLA_FRESH_START_INIT_STATE_IDX=0

  if [[ "$use_custom" == "1" ]]; then
    local lang
    lang="$(get_old_lang "$cond")"
    export LIBERO_LANGUAGE_MODE=custom
    export LIBERO_CUSTOM_LANGUAGE="$lang"
    export SMOLVLA_FRESH_LANGUAGE="$lang"
    echo "[INFO] condition=$cond task_id=$task_id custom_language=[$LIBERO_CUSTOM_LANGUAGE]"
  else
    unset LIBERO_LANGUAGE_MODE
    unset LIBERO_CUSTOM_LANGUAGE
    unset LIBERO_CONSTANT_LANGUAGE
    export SMOLVLA_FRESH_LANGUAGE=""
    echo "[INFO] condition=$cond task_id=$task_id using native task language"
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
    --job_name="smolvla_${cond}_fresh_10seeds" \
    2>&1 | tee "$out_dir/run.log"
}

run_one_condition task8_original 8 0
run_one_condition task8_empty_language 8 1
run_one_condition task8_nonsense_language 8 1
run_one_condition task8_unrelated_language 8 1
run_one_condition task8_wrong_source_language_only 8 1
run_one_condition task8_wrong_target_language_only 8 1
run_one_condition task8_wrong_source_wrong_target_language_only 8 1

# positive-control native task
run_one_condition task1_native_next_to_ramekin 1 0

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