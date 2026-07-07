from pathlib import Path
import argparse
import csv
import json
import numpy as np

from goal_inference_utils import infer_goal_for_row


GOAL_THRESHOLD = 0.05


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
        return np.full(dim, np.nan, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] != dim:
        return np.full(dim, np.nan, dtype=np.float32)
    return arr


def dist(a, b):
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def displacement(p0, p1):
    if p0 is None or p1 is None:
        return float("nan")
    return float(np.linalg.norm(p1 - p0))


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


def add_vec_features(row, prefix, vec):
    vec = safe_vec(vec, dim=3)
    row[f"{prefix}_x"] = float(vec[0])
    row[f"{prefix}_y"] = float(vec[1])
    row[f"{prefix}_z"] = float(vec[2])


def add_state_features(row, obs, language_source, language_target, default_source, default_target):
    eef = get_vec(obs, "robot0_eef_pos")

    lang_source_pos = get_vec(obs, f"{language_source}_pos")
    lang_target_pos = get_vec(obs, f"{language_target}_pos")
    default_source_pos = get_vec(obs, f"{default_source}_pos")
    default_target_pos = get_vec(obs, f"{default_target}_pos")

    gripper_qpos = get_vec(obs, "robot0_gripper_qpos")
    if gripper_qpos is None:
        gripper_width = float("nan")
    else:
        gripper_width = float(np.abs(gripper_qpos).sum())

    row["feat_state_d_eef_language_source"] = dist(eef, lang_source_pos)
    row["feat_state_d_eef_default_source"] = dist(eef, default_source_pos)

    row["feat_state_d_language_source_language_target"] = dist(lang_source_pos, lang_target_pos)
    row["feat_state_d_default_source_default_target"] = dist(default_source_pos, default_target_pos)

    row["feat_state_d_language_source_default_target"] = dist(lang_source_pos, default_target_pos)
    row["feat_state_d_default_source_language_target"] = dist(default_source_pos, lang_target_pos)

    row["feat_state_gripper_width"] = gripper_width

    add_vec_features(
        row,
        "feat_state_rel_eef_to_language_source",
        safe_vec(lang_source_pos) - safe_vec(eef),
    )
    add_vec_features(
        row,
        "feat_state_rel_eef_to_default_source",
        safe_vec(default_source_pos) - safe_vec(eef),
    )
    add_vec_features(
        row,
        "feat_state_rel_language_source_to_language_target",
        safe_vec(lang_target_pos) - safe_vec(lang_source_pos),
    )
    add_vec_features(
        row,
        "feat_state_rel_default_source_to_default_target",
        safe_vec(default_target_pos) - safe_vec(default_source_pos),
    )


def add_chunk_summary_features(row, chunks, prefix_chunks, max_chunks):
    action_dim = 7
    chunk_len = 8

    used_chunks = chunks[:prefix_chunks]
    used_actions = []
    mask = np.zeros(max_chunks, dtype=np.float32)

    for ci in range(max_chunks):
        if ci < len(used_chunks):
            raw = np.asarray(used_chunks[ci].get("raw_actions_chunk", []), dtype=np.float32)

            if raw.ndim == 2 and raw.shape[1] == action_dim:
                raw = raw[:chunk_len]
                used_actions.append(raw)
                mask[ci] = 1.0

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


def classify_episode(final_d_language_goal, final_d_default_goal, steps):
    language_done = final_d_language_goal < GOAL_THRESHOLD
    default_done = final_d_default_goal < GOAL_THRESHOLD
    timeout_like = steps >= 200

    if language_done:
        return "language_goal_completed"

    if default_done and not language_done:
        return "default_goal_completed_against_language"

    if timeout_like:
        return "timeout_or_repeated_failure"

    return "failure_or_no_goal_completed"


def compute_episode_summary(index_row, chunks, steps):
    if not chunks or not steps:
        return None

    first_obs = chunks[0].get("query_obs_summary")
    if first_obs is None:
        first_obs = steps[0].get("obs_before_summary", steps[0].get("obs_summary", {}))

    final_obs = steps[-1].get("obs_after_summary", steps[-1].get("obs_summary", {}))

    language_source, language_target, default_source, default_target = infer_goal_for_row(index_row, first_obs)

    lang_source_init = get_vec(first_obs, f"{language_source}_pos")
    default_source_init = get_vec(first_obs, f"{default_source}_pos")

    lang_source_final = get_vec(final_obs, f"{language_source}_pos")
    lang_target_final = get_vec(final_obs, f"{language_target}_pos")

    default_source_final = get_vec(final_obs, f"{default_source}_pos")
    default_target_final = get_vec(final_obs, f"{default_target}_pos")

    final_d_language_goal = dist(lang_source_final, lang_target_final)
    final_d_default_goal = dist(default_source_final, default_target_final)

    lang_source_disp = displacement(lang_source_init, lang_source_final)
    default_source_disp = displacement(default_source_init, default_source_final)

    episode_class = classify_episode(final_d_language_goal, final_d_default_goal, len(steps))

    row = {
        "file_name": index_row["file_name"],
        "condition": index_row["condition"],
        "task_ids": index_row["task_ids"],
        "custom_language": index_row["custom_language"],
        "steps": len(steps),
        "chunks": len(chunks),
        "success_by_reward": index_row["success_by_reward"],
        "language_source": language_source,
        "language_target": language_target,
        "default_source": default_source,
        "default_target": default_target,
        "final_d_language_goal": final_d_language_goal,
        "final_d_default_goal": final_d_default_goal,
        "language_source_displacement_total": lang_source_disp,
        "default_source_displacement_total": default_source_disp,
        "source_displacement_advantage_total": default_source_disp - lang_source_disp,
        "episode_class": episode_class,
        "y_episode_default_against_language": int(episode_class == "default_goal_completed_against_language"),
        "y_episode_timeout": int(episode_class in {"timeout_or_repeated_failure", "failure_or_no_goal_completed"}),
    }

    return row


def compute_chunk_rows(index_row, chunks, steps, episode_row, horizon):
    if not chunks or not steps or episode_row is None:
        return []

    steps_by_id = {}
    for step in steps:
        steps_by_id[int(step["step"])] = step

    max_step = max(steps_by_id.keys())

    out_rows = []

    for chunk_obj in chunks:
        start_step = int(chunk_obj["step"])
        end_step = min(start_step + horizon - 1, max_step)

        start_obs = chunk_obj.get("query_obs_summary")
        if start_obs is None:
            continue

        end_step_obj = steps_by_id.get(end_step)
        if end_step_obj is None:
            continue

        end_obs = end_step_obj.get("obs_after_summary", end_step_obj.get("obs_summary"))
        if end_obs is None:
            continue

        language_source = episode_row["language_source"]
        language_target = episode_row["language_target"]
        default_source = episode_row["default_source"]
        default_target = episode_row["default_target"]

        eef_start = get_vec(start_obs, "robot0_eef_pos")
        eef_end = get_vec(end_obs, "robot0_eef_pos")

        lang_source_start = get_vec(start_obs, f"{language_source}_pos")
        lang_target_start = get_vec(start_obs, f"{language_target}_pos")
        default_source_start = get_vec(start_obs, f"{default_source}_pos")
        default_target_start = get_vec(start_obs, f"{default_target}_pos")

        lang_source_end = get_vec(end_obs, f"{language_source}_pos")
        lang_target_end = get_vec(end_obs, f"{language_target}_pos")
        default_source_end = get_vec(end_obs, f"{default_source}_pos")
        default_target_end = get_vec(end_obs, f"{default_target}_pos")

        d_eef_lang_source_start = dist(eef_start, lang_source_start)
        d_eef_lang_source_end = dist(eef_end, lang_source_end)

        d_eef_default_source_start = dist(eef_start, default_source_start)
        d_eef_default_source_end = dist(eef_end, default_source_end)

        d_lang_goal_start = dist(lang_source_start, lang_target_start)
        d_lang_goal_end = dist(lang_source_end, lang_target_end)

        d_default_goal_start = dist(default_source_start, default_target_start)
        d_default_goal_end = dist(default_source_end, default_target_end)

        language_source_approach_progress_h = d_eef_lang_source_start - d_eef_lang_source_end
        default_source_approach_progress_h = d_eef_default_source_start - d_eef_default_source_end

        language_goal_progress_h = d_lang_goal_start - d_lang_goal_end
        default_goal_progress_h = d_default_goal_start - d_default_goal_end

        language_source_displacement_h = displacement(lang_source_start, lang_source_end)
        default_source_displacement_h = displacement(default_source_start, default_source_end)

        source_approach_advantage_h = default_source_approach_progress_h - language_source_approach_progress_h
        source_displacement_advantage_h = default_source_displacement_h - language_source_displacement_h
        default_goal_advantage_h = default_goal_progress_h - language_goal_progress_h

        row = {
            "file_name": index_row["file_name"],
            "condition": index_row["condition"],
            "task_ids": index_row["task_ids"],
            "custom_language": index_row["custom_language"],
            "chunk_start_step": start_step,
            "chunk_end_step": end_step,
            "horizon": horizon,
            "language_source": language_source,
            "language_target": language_target,
            "default_source": default_source,
            "default_target": default_target,
            "episode_class": episode_row["episode_class"],
            "y_episode_default_against_language": episode_row["y_episode_default_against_language"],
            "y_episode_timeout": episode_row["y_episode_timeout"],
            "label_language_source_approach_progress_h": language_source_approach_progress_h,
            "label_default_source_approach_progress_h": default_source_approach_progress_h,
            "label_source_approach_advantage_h": source_approach_advantage_h,
            "label_language_source_displacement_h": language_source_displacement_h,
            "label_default_source_displacement_h": default_source_displacement_h,
            "label_source_displacement_advantage_h": source_displacement_advantage_h,
            "label_language_goal_progress_h": language_goal_progress_h,
            "label_default_goal_progress_h": default_goal_progress_h,
            "label_default_goal_advantage_h": default_goal_advantage_h,
            "y_chunk_default_goal_advantage": int(default_goal_advantage_h > 0.005),
            "y_chunk_source_approach_advantage": int(source_approach_advantage_h > 0.005),
            "y_chunk_source_displacement_advantage": int(source_displacement_advantage_h > 0.002),
        }

        add_state_features(row, start_obs, language_source, language_target, default_source, default_target)
        add_chunk_summary_features(row, [chunk_obj], prefix_chunks=1, max_chunks=1)

        out_rows.append(row)

    return out_rows


def compute_early_rows(index_row, chunks, steps, episode_row, max_chunks, prefix_list):
    if not chunks or not steps or episode_row is None:
        return []

    first_obs = chunks[0].get("query_obs_summary")
    if first_obs is None:
        first_obs = steps[0].get("obs_before_summary", steps[0].get("obs_summary", {}))

    language_source = episode_row["language_source"]
    language_target = episode_row["language_target"]
    default_source = episode_row["default_source"]
    default_target = episode_row["default_target"]

    out_rows = []

    for prefix_chunks in prefix_list:
        row = {
            "file_name": index_row["file_name"],
            "condition": index_row["condition"],
            "task_ids": index_row["task_ids"],
            "custom_language": index_row["custom_language"],
            "prefix_chunks": prefix_chunks,
            "max_chunks": max_chunks,
            "num_total_chunks": len(chunks),
            "steps": len(steps),
            "success_by_reward": index_row["success_by_reward"],
            "episode_class": episode_row["episode_class"],
            "y_episode_default_against_language": episode_row["y_episode_default_against_language"],
            "y_episode_timeout": episode_row["y_episode_timeout"],
            "language_source": language_source,
            "language_target": language_target,
            "default_source": default_source,
            "default_target": default_target,
        }

        add_state_features(row, first_obs, language_source, language_target, default_source, default_target)
        add_chunk_summary_features(row, chunks, prefix_chunks=prefix_chunks, max_chunks=max_chunks)

        out_rows.append(row)

    return out_rows


def write_csv(path: Path, rows):
    if not rows:
        print(f"No rows to write: {path}")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--prefix_chunks", type=str, default="1,2,4")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing dump_index.csv: {index_csv}")

    index_rows = read_csv(index_csv)
    prefix_list = [int(x) for x in args.prefix_chunks.split(",") if x.strip()]

    episode_rows = []
    chunk_rows = []
    early_rows = []

    for index_row in index_rows:
        meta, chunks, steps = parse_jsonl(Path(index_row["file"]))

        episode_row = compute_episode_summary(index_row, chunks, steps)
        if episode_row is None:
            continue

        episode_rows.append(episode_row)
        chunk_rows.extend(compute_chunk_rows(index_row, chunks, steps, episode_row, args.horizon))
        early_rows.extend(compute_early_rows(index_row, chunks, steps, episode_row, args.max_chunks, prefix_list))

    episode_csv = dump_dir / f"source_aware_episode_summary_h{args.horizon}.csv"
    chunk_csv = dump_dir / f"source_aware_chunk_dataset_h{args.horizon}.csv"
    early_csv = dump_dir / f"source_aware_early_dataset_h{args.horizon}_max{args.max_chunks}.csv"

    write_csv(episode_csv, episode_rows)
    write_csv(chunk_csv, chunk_rows)
    write_csv(early_csv, early_rows)

    print("\nEpisode class counts:")
    counts = {}
    for r in episode_rows:
        counts[r["episode_class"]] = counts.get(r["episode_class"], 0) + 1
    for k, v in sorted(counts.items()):
        print(k, v)

    print("\nCondition summary:")
    conds = sorted(set(r["condition"] for r in episode_rows))
    for cond in conds:
        sub = [r for r in episode_rows if r["condition"] == cond]
        class_counts = {}
        for r in sub:
            class_counts[r["episode_class"]] = class_counts.get(r["episode_class"], 0) + 1
        print(cond, class_counts)


if __name__ == "__main__":
    main()