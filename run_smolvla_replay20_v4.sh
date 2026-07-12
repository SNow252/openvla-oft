#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/openvla-oft
conda activate openvla-oft

export MUJOCO_GL=egl
export PYTHONPATH=$PWD/LIBERO:$PWD:$PYTHONPATH

export SMOLVLA_FRESH20_ROOT=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/smolvla_multicond_20seeds_20260712_172705
export SMOLVLA_MERGED20="$SMOLVLA_FRESH20_ROOT/policy_chunks_merged"

if [ ! -d "$SMOLVLA_MERGED20" ]; then
  echo "[ERROR] Missing merged chunk dir: $SMOLVLA_MERGED20"
  exit 1
fi

export SMOLVLA_REPLAY20_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_replay_v4/smolvla_multicond_20seeds_replay_140_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SMOLVLA_REPLAY20_DIR"

echo "[INFO] SMOLVLA_FRESH20_ROOT=$SMOLVLA_FRESH20_ROOT"
echo "[INFO] SMOLVLA_MERGED20=$SMOLVLA_MERGED20"
echo "[INFO] SMOLVLA_REPLAY20_DIR=$SMOLVLA_REPLAY20_DIR"

echo "[INFO] Checking chunks..."
for d in "$SMOLVLA_MERGED20"/init_*; do
  echo "$(basename "$d") $(ls "$d"/*.npz 2>/dev/null | wc -l)"
done

echo "[INFO] Start replay..."
python tools/replay_policy_action_chunks_v3.py \
  --task_suite_name libero_spatial \
  --task_id 8 \
  --start_init_state_idx 0 \
  --num_seeds 20 \
  --chunk_dir "$SMOLVLA_MERGED20" \
  --chunk_glob "*.npz" \
  --out_dir "$SMOLVLA_REPLAY20_DIR" \
  --settle_steps 30 \
  --gripper_open_value -1 \
  --contact_threshold 0.10 \
  --move_threshold 0.020 \
  --lift_threshold 0.010 \
  --max_steps 140 \
  2>&1 | tee "$SMOLVLA_REPLAY20_DIR/replay.log"

echo "[INFO] Rebuild dual-source labels..."
python tools/rebuild_replay_dual_source_labels_v3.py \
  --data_dir "$SMOLVLA_REPLAY20_DIR" \
  --input_csv summary.csv \
  --output_csv summary_dual_source.csv \
  --contact_threshold 0.10 \
  --move_threshold 0.020 \
  --lift_threshold 0.010 \
  2>&1 | tee "$SMOLVLA_REPLAY20_DIR/dual_source.log"

echo "[INFO] Build policy consequence table..."
python - <<'PY'
import csv, os, math
from collections import defaultdict
import numpy as np

out_dir = os.environ["SMOLVLA_REPLAY20_DIR"]
path = os.path.join(out_dir, "summary_dual_source.csv")
save_path = os.path.join(out_dir, "smolvla_20seeds_policy_consequence_table.csv")
md_path = os.path.join(out_dir, "smolvla_20seeds_policy_consequence_table.md")

rows = []
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        out = {}
        for k, v in r.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
        rows.append(out)

def vals(rs, k):
    a = []
    for r in rs:
        v = r.get(k)
        if isinstance(v, float) and not math.isnan(v):
            a.append(v)
    return a

by_cond = defaultdict(list)
for r in rows:
    by_cond[r.get("condition", "unknown")].append(r)

fields = [
    "condition",
    "n",
    "bowl1_grasp",
    "bowl1_lifted",
    "bowl1_quality",
    "bowl2_grasp",
    "bowl2_lifted",
    "bowl2_quality",
    "bowl2_minus_bowl1_grasp",
    "bowl2_minus_bowl1_quality",
]

table = []
for cond, rs in sorted(by_cond.items()):
    def mean(k):
        a = vals(rs, k)
        return float(np.mean(a)) if a else float("nan")
    table.append({
        "condition": cond,
        "n": len(rs),
        "bowl1_grasp": mean("bowl1_grasp_success_proxy"),
        "bowl1_lifted": mean("bowl1_source_lifted"),
        "bowl1_quality": mean("bowl1_grasp_quality_score"),
        "bowl2_grasp": mean("bowl2_grasp_success_proxy"),
        "bowl2_lifted": mean("bowl2_source_lifted"),
        "bowl2_quality": mean("bowl2_grasp_quality_score"),
        "bowl2_minus_bowl1_grasp": mean("bowl2_minus_bowl1_grasp_success_proxy"),
        "bowl2_minus_bowl1_quality": mean("bowl2_minus_bowl1_quality"),
    })

with open(save_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in table:
        writer.writerow(r)

lines = []
lines.append("| Condition | n | B1 grasp | B1 q | B2 grasp | B2 q | B2-B1 q |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in table:
    lines.append(
        f"| {r['condition']} | {r['n']} | "
        f"{r['bowl1_grasp']:.2f} | {r['bowl1_quality']:+.3f} | "
        f"{r['bowl2_grasp']:.2f} | {r['bowl2_quality']:+.3f} | "
        f"{r['bowl2_minus_bowl1_quality']:+.3f} |"
    )

with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("saved:", save_path)
print("saved:", md_path)
print()
print("\n".join(lines))
PY

echo "[INFO] Done."
echo "[INFO] Replay dir: $SMOLVLA_REPLAY20_DIR"
echo
echo "[INFO] Last dual-source summary:"
tail -180 "$SMOLVLA_REPLAY20_DIR/dual_source.log"
echo
echo "[INFO] Table:"
cat "$SMOLVLA_REPLAY20_DIR/smolvla_20seeds_policy_consequence_table.md"
SH