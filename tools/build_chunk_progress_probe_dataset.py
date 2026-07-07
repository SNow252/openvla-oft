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


def parse_jsonl(path: Path):
    chunks = []
    steps = {}

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "chunk":
                chunks.append(obj)

            elif typ == "step":
                steps[int(obj["step"])] = obj

    return chunks, steps


def infer_goal(index_row, first_obs):
    from goal_inference_utils import infer_goal_for_row
    return infer_goal_for_row(index_row, first_obs)


def add_vec_features(row, prefix, vec):
    vec = safe_vec(vec, dim=3)
    row[f"{prefix}_x"] = float(vec[0])
    row[f"{prefix}_y"] = float(vec[1])
    row[f"{prefix}_z"] = float(vec[2])


def add_chunk_features(row, chunk):
    chunk = np.asarray(chunk, dtype=np.float32)

    if chunk.ndim != 2:
        raise ValueError(f"Bad chunk shape: {chunk.shape}")

    # Flatten 8x7 action chunk.
    flat = chunk.reshape(-1)
    for i, v in enumerate(flat):
        row[f"feat_chunk_flat_{i}"] = float(v)

    mean = chunk.mean(axis=0)
    std = chunk.std(axis=0)
    first = chunk[0]
    last = chunk[-1]

    for i in range(chunk.shape[1]):
        row[f"feat_chunk_mean_{i}"] = float(mean[i])
        row[f"feat_chunk_std_{i}"] = float(std[i])
        row[f"feat_chunk_first_{i}"] = float(first[i])
        row[f"feat_chunk_last_{i}"] = float(last[i])

    diffs = np.diff(chunk, axis=0)
    row["feat_chunk_l2_sum"] = float(np.linalg.norm(chunk, axis=1).sum())
    row["feat_chunk_diff_l2_sum"] = float(np.linalg.norm(diffs, axis=1).sum())


def build_rows_for_episode(index_row, horizon):
    jsonl_path = Path(index_row["file"])
    chunks, steps = parse_jsonl(jsonl_path)

    rows = []

    for chunk_obj in chunks:
        start_step = int(chunk_obj["step"])
        query_obs = chunk_obj.get("query_obs_summary", None)

        if query_obs is None:
            continue

        end_step = min(start_step + horizon - 1, max(steps.keys()))
        end_obj = steps.get(end_step, None)

        if end_obj is None:
            continue

        after_obs = end_obj.get("obs_after_summary", None)
        if after_obs is None:
            after_obs = end_obj.get("obs_summary", None)

        if after_obs is None:
            continue

        lang_source, lang_target, default_source, default_target = infer_goal(index_row, query_obs)

        eef = get_vec(query_obs, "robot0_eef_pos")
        lang_source_pos = get_vec(query_obs, f"{lang_source}_pos")
        lang_target_pos = get_vec(query_obs, f"{lang_target}_pos")
        default_source_pos = get_vec(query_obs, f"{default_source}_pos")
        default_target_pos = get_vec(query_obs, f"{default_target}_pos")

        eef2 = get_vec(after_obs, "robot0_eef_pos")
        lang_source_pos2 = get_vec(after_obs, f"{lang_source}_pos")
        lang_target_pos2 = get_vec(after_obs, f"{lang_target}_pos")
        default_source_pos2 = get_vec(after_obs, f"{default_source}_pos")
        default_target_pos2 = get_vec(after_obs, f"{default_target}_pos")

        gripper_qpos = get_vec(query_obs, "robot0_gripper_qpos")
        if gripper_qpos is None:
            gripper_width = np.nan
        else:
            gripper_width = float(np.abs(gripper_qpos).sum())

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
            "chunk_start_step": start_step,
            "chunk_end_step": end_step,
            "horizon": horizon,
            "language_source": lang_source,
            "language_target": lang_target,
            "default_source": default_source,
            "default_target": default_target,
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

        add_vec_features(
            row,
            "feat_rel_eef_to_language_source",
            safe_vec(lang_source_pos) - safe_vec(eef),
        )
        add_vec_features(
            row,
            "feat_rel_language_source_to_language_target",
            safe_vec(lang_target_pos) - safe_vec(lang_source_pos),
        )
        add_vec_features(
            row,
            "feat_rel_language_source_to_default_target",
            safe_vec(default_target_pos) - safe_vec(lang_source_pos),
        )

        add_chunk_features(row, chunk_obj["raw_actions_chunk"])

        rows.append(row)

    return rows


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

    out_csv = dump_dir / f"chunk_progress_probe_dataset_h{args.horizon}.csv"

    if not all_rows:
        raise RuntimeError("No rows generated. Check whether query_obs_summary exists in chunk records.")

    fieldnames = list(all_rows[0].keys())

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote: {out_csv}")
    print(f"num_rows: {len(all_rows)}")

    print("\ncondition counts:")
    for cond in sorted(set(r["condition"] for r in all_rows)):
        print(cond, sum(r["condition"] == cond for r in all_rows))


if __name__ == "__main__":
    main()