from pathlib import Path
import argparse
import csv
import json
import numpy as np


PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"


def read_index(path: Path):
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def get_vec(obs, key):
    value = obs.get(key, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def safe_vec(x, dim=3):
    if x is None:
        return np.full(dim, np.nan, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def dist(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(a - b))


def infer_goal(row, first_obs):
    from goal_inference_utils import infer_goal_for_row
    return infer_goal_for_row(row, first_obs)


def parse_jsonl(path: Path):
    meta = {}
    steps = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "step":
                steps.append(obj)

    return meta, steps


def add_vec_features(row, prefix, vec):
    vec = safe_vec(vec, dim=3)
    row[f"{prefix}_x"] = float(vec[0])
    row[f"{prefix}_y"] = float(vec[1])
    row[f"{prefix}_z"] = float(vec[2])


def build_rows_for_episode(index_row, horizon):
    jsonl_path = Path(index_row["file"])
    _, steps = parse_jsonl(jsonl_path)

    if not steps:
        return []

    first_obs = steps[0].get("obs_summary", {})
    lang_source, lang_target, default_source, default_target = infer_goal(index_row, first_obs)

    out_rows = []

    for t, step in enumerate(steps):
        obs = step.get("obs_summary", {})
        t2 = min(t + horizon, len(steps) - 1)
        obs2 = steps[t2].get("obs_summary", {})

        eef = get_vec(obs, "robot0_eef_pos")
        lang_source_pos = get_vec(obs, f"{lang_source}_pos")
        lang_target_pos = get_vec(obs, f"{lang_target}_pos")
        default_source_pos = get_vec(obs, f"{default_source}_pos")
        default_target_pos = get_vec(obs, f"{default_target}_pos")

        eef2 = get_vec(obs2, "robot0_eef_pos")
        lang_source_pos2 = get_vec(obs2, f"{lang_source}_pos")
        lang_target_pos2 = get_vec(obs2, f"{lang_target}_pos")
        default_source_pos2 = get_vec(obs2, f"{default_source}_pos")
        default_target_pos2 = get_vec(obs2, f"{default_target}_pos")

        gripper_qpos = get_vec(obs, "robot0_gripper_qpos")
        if gripper_qpos is None:
            gripper_width = np.nan
        else:
            gripper_width = float(np.abs(gripper_qpos).sum())

        action = np.asarray(step.get("processed_action", []), dtype=np.float32)

        d_eef_lang_source = dist(eef, lang_source_pos)
        d_lang_source_lang_target = dist(lang_source_pos, lang_target_pos)
        d_default_source_default_target = dist(default_source_pos, default_target_pos)
        d_lang_source_default_target = dist(lang_source_pos, default_target_pos)

        d_eef_lang_source2 = dist(eef2, lang_source_pos2)
        d_lang_source_lang_target2 = dist(lang_source_pos2, lang_target_pos2)
        d_default_source_default_target2 = dist(default_source_pos2, default_target_pos2)
        d_lang_source_default_target2 = dist(lang_source_pos2, default_target_pos2)

        source_approach_progress = d_eef_lang_source - d_eef_lang_source2
        language_target_progress = d_lang_source_lang_target - d_lang_source_lang_target2
        default_target_progress = d_default_source_default_target - d_default_source_default_target2
        lang_source_to_default_target_progress = (
            d_lang_source_default_target - d_lang_source_default_target2
        )
        default_minus_language_progress = (
            lang_source_to_default_target_progress - language_target_progress
        )

        row = {
            "file_name": index_row["file_name"],
            "condition": index_row["condition"],
            "task_ids": index_row["task_ids"],
            "custom_language": index_row["custom_language"],
            "step": t,
            "horizon": horizon,
            "language_source": lang_source,
            "language_target": lang_target,
            "default_source": default_source,
            "default_target": default_target,
            "reward": step.get("reward"),
            "done": step.get("done"),
            "feat_d_eef_language_source": d_eef_lang_source,
            "feat_d_language_source_language_target": d_lang_source_lang_target,
            "feat_d_default_source_default_target": d_default_source_default_target,
            "feat_d_language_source_default_target": d_lang_source_default_target,
            "feat_gripper_width": gripper_width,
            "label_source_approach_progress_h": source_approach_progress,
            "label_language_target_progress_h": language_target_progress,
            "label_default_target_progress_h": default_target_progress,
            "label_default_minus_language_progress_h": default_minus_language_progress,
            "y_language_progress_pos": int(language_target_progress > 0.001),
            "y_default_over_language": int(default_minus_language_progress > 0.005),
        }

        add_vec_features(row, "feat_rel_eef_to_language_source", safe_vec(lang_source_pos) - safe_vec(eef))
        add_vec_features(row, "feat_rel_language_source_to_language_target", safe_vec(lang_target_pos) - safe_vec(lang_source_pos))
        add_vec_features(row, "feat_rel_language_source_to_default_target", safe_vec(default_target_pos) - safe_vec(lang_source_pos))

        for i in range(7):
            row[f"feat_action_{i}"] = float(action[i]) if i < len(action) else np.nan

        out_rows.append(row)

    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing dump_index.csv: {index_csv}")

    index_rows = read_index(index_csv)

    all_rows = []
    for index_row in index_rows:
        all_rows.extend(build_rows_for_episode(index_row, args.horizon))

    out_csv = dump_dir / f"progress_probe_dataset_h{args.horizon}.csv"

    fieldnames = list(all_rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote: {out_csv}")
    print(f"num_rows: {len(all_rows)}")
    print("conditions:")
    for cond in sorted(set(r["condition"] for r in all_rows)):
        print(cond, sum(r["condition"] == cond for r in all_rows))


if __name__ == "__main__":
    main()