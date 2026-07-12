#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/openvla-oft

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate openvla-oft || true
fi

export MUJOCO_GL=egl
export PYTHONPATH=$PWD/LIBERO:$PWD:$PYTHONPATH

export SMOLVLA_MT10_ROOT=$(ls -td /home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/smolvla_multitask_minimal10_* | head -1)
export SMOLVLA_MT10_MERGED="$SMOLVLA_MT10_ROOT/policy_chunks_merged"

if [ ! -d "$SMOLVLA_MT10_MERGED" ]; then
  echo "[ERROR] Missing merged dir: $SMOLVLA_MT10_MERGED"
  exit 1
fi

echo "[INFO] SMOLVLA_MT10_ROOT=$SMOLVLA_MT10_ROOT"
echo "[INFO] SMOLVLA_MT10_MERGED=$SMOLVLA_MT10_MERGED"

run_replay_one_task() {
  local task_id="$1"
  local task_key="task${task_id}"

  export REPLAY_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_replay_v4/smolvla_${task_key}_minimal10_replay_140_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$REPLAY_DIR"

  echo
  echo "======================================================================"
  echo "[INFO] Replay $task_key"
  echo "[INFO] REPLAY_DIR=$REPLAY_DIR"
  echo "======================================================================"

  python tools/replay_policy_action_chunks_v3.py \
    --task_suite_name libero_spatial \
    --task_id "$task_id" \
    --start_init_state_idx 0 \
    --num_seeds 10 \
    --chunk_dir "$SMOLVLA_MT10_MERGED" \
    --chunk_glob "*task${task_id}*.npz" \
    --out_dir "$REPLAY_DIR" \
    --settle_steps 30 \
    --gripper_open_value -1 \
    --contact_threshold 0.10 \
    --move_threshold 0.020 \
    --lift_threshold 0.010 \
    --max_steps 140 \
    2>&1 | tee "$REPLAY_DIR/replay.log"

  python tools/rebuild_replay_dual_source_labels_v3.py \
    --data_dir "$REPLAY_DIR" \
    --input_csv summary.csv \
    --output_csv summary_dual_source.csv \
    --contact_threshold 0.10 \
    --move_threshold 0.020 \
    --lift_threshold 0.010 \
    2>&1 | tee "$REPLAY_DIR/dual_source.log"

  python tools/alias_dual_source_to_object_pair_v4.py \
    --data_dir "$REPLAY_DIR" \
    --input_csv summary_dual_source.csv \
    --output_csv summary_object_pair.csv \
    --config_json configs/libero_spatial_object_mapping_v4.json \
    --task_suite_name libero_spatial \
    --task_key "$task_key" \
    2>&1 | tee "$REPLAY_DIR/object_pair_alias.log"

  python - <<'PY'
import csv, os, math
from collections import defaultdict
import numpy as np

d = os.environ["REPLAY_DIR"]
path = os.path.join(d, "summary_object_pair.csv")
md_path = os.path.join(d, "policy_consequence_table.md")
csv_path = os.path.join(d, "policy_consequence_table.csv")

rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        rr = {}
        for k, v in r.items():
            try:
                rr[k] = float(v)
            except Exception:
                rr[k] = v
        rows.append(rr)

def vals(rs, k):
    out = []
    for r in rs:
        v = r.get(k)
        if isinstance(v, float) and math.isfinite(v):
            out.append(v)
    return out

by = defaultdict(list)
for r in rows:
    by[r.get("condition", "unknown")].append(r)

fields = [
    "condition", "n",
    "primary_grasp", "primary_q",
    "alternative_grasp", "alternative_q",
    "alternative_minus_primary_q",
]

table = []
for cond, rs in sorted(by.items()):
    def mean(k):
        x = vals(rs, k)
        return float(np.mean(x)) if x else float("nan")
    table.append({
        "condition": cond,
        "n": len(rs),
        "primary_grasp": mean("primary_grasp_success_proxy"),
        "primary_q": mean("primary_grasp_quality_score"),
        "alternative_grasp": mean("alternative_grasp_success_proxy"),
        "alternative_q": mean("alternative_grasp_quality_score"),
        "alternative_minus_primary_q": mean("alternative_minus_primary_quality"),
    })

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(table)

lines = []
lines.append("| Condition | n | Primary grasp | Primary q | Alt grasp | Alt q | Alt-Primary q |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in table:
    lines.append(
        f"| {r['condition']} | {r['n']} | "
        f"{r['primary_grasp']:.2f} | {r['primary_q']:+.3f} | "
        f"{r['alternative_grasp']:.2f} | {r['alternative_q']:+.3f} | "
        f"{r['alternative_minus_primary_q']:+.3f} |"
    )

open(md_path, "w").write("\n".join(lines) + "\n")
print("[saved]", csv_path)
print("[saved]", md_path)
print()
print("\n".join(lines))
PY

  echo
  echo "[INFO] table:"
  cat "$REPLAY_DIR/policy_consequence_table.md"
}

run_replay_one_task 3
run_replay_one_task 9

echo
echo "[INFO] Done all SmolVLA multitask minimal replays."