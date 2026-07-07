from pathlib import Path
import argparse
import csv
import json
import numpy as np


PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"


CLASS_TO_ID = {
    "language_goal_completed": 0,
    "default_goal_completed_against_language": 1,
    "timeout_or_repeated_failure": 2,
    "failure_or_no_goal_completed": 3,
}


def read_csv(path: Path):
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
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] != dim:
        return np.zeros(dim, dtype=np.float32)
    return arr


def dist(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def parse_jsonl(path: Path):
    meta = {}
    chunks = []
    steps = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "meta":
                meta = obj

            elif typ == "chunk":
                chunks.append(obj)

            elif typ == "step":
                steps.append(obj)

    return meta, chunks, steps


def infer_goal(index_row, first_obs):
    from goal_inference_utils import infer_goal_for_row
    return infer_goal_for_row(index_row, first_obs)


def add_vec(row, prefix, vec):
    vec = safe_vec(vec, dim=3)
    row[f"{prefix}_x"] = float(vec[0])
    row[f"{prefix}_y"] = float(vec[1])
    row[f"{prefix}_z"] = float(vec[2])


def get_initial_state_features(first_obs, language_source, language_target, default_source, default_target):
    row = {}

    eef = get_vec(first_obs, "robot0_eef_pos")
    lang_source_pos = get_vec(first_obs, f"{language_source}_pos")
    lang_target_pos = get_vec(first_obs, f"{language_target}_pos")
    default_source_pos = get_vec(first_obs, f"{default_source}_pos")
    default_target_pos = get_vec(first_obs, f"{default_target}_pos")

    gripper_qpos = get_vec(first_obs, "robot0_gripper_qpos")
    if gripper_qpos is None:
        gripper_width = 0.0
    else:
        gripper_width = float(np.abs(gripper_qpos).sum())

    row["feat_state_d_eef_language_source"] = dist(eef, lang_source_pos)
    row["feat_state_d_language_source_language_target"] = dist(lang_source_pos, lang_target_pos)
    row["feat_state_d_default_source_default_target"] = dist(default_source_pos, default_target_pos)
    row["feat_state_d_language_source_default_target"] = dist(lang_source_pos, default_target_pos)
    row["feat_state_gripper_width"] = gripper_width

    add_vec(row, "feat_state_rel_eef_to_language_source", safe_vec(lang_source_pos) - safe_vec(eef))
    add_vec(row, "feat_state_rel_language_source_to_language_target", safe_vec(lang_target_pos) - safe_vec(lang_source_pos))
    add_vec(row, "feat_state_rel_language_source_to_default_target", safe_vec(default_target_pos) - safe_vec(lang_source_pos))

    return row


def get_chunk_features(chunks, prefix_chunks, max_chunks):
    row = {}

    action_dim = 7
    chunk_len = 8
    total_len = max_chunks * chunk_len * action_dim

    used_chunks = chunks[:prefix_chunks]

    flat_all = np.zeros(total_len, dtype=np.float32)
    mask = np.zeros(max_chunks, dtype=np.float32)

    used_actions = []

    cursor = 0
    for ci in range(max_chunks):
        if ci < len(used_chunks):
            raw = np.asarray(used_chunks[ci].get("raw_actions_chunk", []), dtype=np.float32)

            if raw.ndim == 2 and raw.shape[1] == action_dim:
                raw = raw[:chunk_len]

                padded = np.zeros((chunk_len, action_dim), dtype=np.float32)
                padded[: raw.shape[0], :] = raw

                flat = padded.reshape(-1)

                flat_all[cursor : cursor + len(flat)] = flat
                mask[ci] = 1.0
                used_actions.append(raw)

        cursor += chunk_len * action_dim

    for i, v in enumerate(flat_all):
        row[f"feat_chunk_flat_{i}"] = float(v)

    for i, v in enumerate(mask):
        row[f"feat_chunk_mask_{i}"] = float(v)

    if used_actions:
        concat = np.concatenate(used_actions, axis=0)
    else:
        concat = np.zeros((1, action_dim), dtype=np.float32)

    mean = concat.mean(axis=0)
    std = concat.std(axis=0)
    first = concat[0]
    last = concat[-1]

    diffs = np.diff(concat, axis=0)
    if len(diffs) == 0:
        diffs = np.zeros_like(concat[:1])

    for i in range(action_dim):
        row[f"feat_chunk_summary_mean_{i}"] = float(mean[i])
        row[f"feat_chunk_summary_std_{i}"] = float(std[i])
        row[f"feat_chunk_summary_first_{i}"] = float(first[i])
        row[f"feat_chunk_summary_last_{i}"] = float(last[i])

    row["feat_chunk_summary_l2_sum"] = float(np.linalg.norm(concat, axis=1).sum())
    row["feat_chunk_summary_diff_l2_sum"] = float(np.linalg.norm(diffs, axis=1).sum())
    row["feat_chunk_summary_num_actions"] = float(len(concat))

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--prefix_chunks", type=str, default="1,2,4")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"
    episode_csv = dump_dir / f"progress_episode_classified_h{args.horizon}.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing: {index_csv}")

    if not episode_csv.exists():
        raise FileNotFoundError(f"Missing: {episode_csv}")

    index_rows = read_csv(index_csv)
    episode_rows = read_csv(episode_csv)

    episode_class_by_file = {
        r["file_name"]: r["episode_class"]
        for r in episode_rows
    }

    prefix_list = [int(x) for x in args.prefix_chunks.split(",") if x.strip()]

    all_rows = []

    for index_row in index_rows:
        jsonl_path = Path(index_row["file"])
        meta, chunks, steps = parse_jsonl(jsonl_path)

        if not chunks:
            continue

        first_obs = None

        if "query_obs_summary" in chunks[0]:
            first_obs = chunks[0]["query_obs_summary"]

        if first_obs is None and steps:
            first_obs = steps[0].get("obs_before_summary", None)

        if first_obs is None and steps:
            first_obs = steps[0].get("obs_summary", None)

        if first_obs is None:
            continue

        language_source, language_target, default_source, default_target = infer_goal(index_row, first_obs)

        episode_class = episode_class_by_file.get(index_row["file_name"], "unknown")
        class_id = CLASS_TO_ID.get(episode_class, -1)

        for prefix_chunks in prefix_list:
            row = {
                "file_name": index_row["file_name"],
                "condition": index_row["condition"],
                "task_ids": index_row["task_ids"],
                "custom_language": index_row["custom_language"],
                "prefix_chunks": prefix_chunks,
                "max_chunks": args.max_chunks,
                "num_total_chunks": len(chunks),
                "steps": index_row["steps"],
                "success_by_reward": index_row["success_by_reward"],
                "episode_class": episode_class,
                "class_id": class_id,
                "language_source": language_source,
                "language_target": language_target,
                "default_source": default_source,
                "default_target": default_target,
            }

            row.update(
                get_initial_state_features(
                    first_obs,
                    language_source,
                    language_target,
                    default_source,
                    default_target,
                )
            )

            row.update(
                get_chunk_features(
                    chunks=chunks,
                    prefix_chunks=prefix_chunks,
                    max_chunks=args.max_chunks,
                )
            )

            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No rows generated.")

    out_csv = dump_dir / f"episode_early_warning_dataset_h{args.horizon}_max{args.max_chunks}.csv"

    fieldnames = list(all_rows[0].keys())

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("wrote:", out_csv)
    print("num_rows:", len(all_rows))

    print("\nClass counts by prefix:")
    for prefix in prefix_list:
        sub = [r for r in all_rows if int(r["prefix_chunks"]) == prefix]
        print("\nprefix_chunks =", prefix, "num_rows =", len(sub))

        counts = {}
        for r in sub:
            counts[r["episode_class"]] = counts.get(r["episode_class"], 0) + 1

        for k, v in sorted(counts.items()):
            print(k, v)

    print("\nCondition counts:")
    cond_counts = {}
    for r in all_rows:
        cond_counts[r["condition"]] = cond_counts.get(r["condition"], 0) + 1

    for k, v in sorted(cond_counts.items()):
        print(k, v)


if __name__ == "__main__":
    main()