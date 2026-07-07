from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np


BOWL_NAMES = ["akita_black_bowl_1", "akita_black_bowl_2"]
PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"


def load_index(index_csv: Path):
    rows = []
    with index_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_vec(obs, key):
    value = obs.get(key, None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr


def dist(a, b):
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def infer_language_goal(row, first_obs):
    """
    Return:
        language_source_object
        language_target_object
        default_source_object
        default_target_object

    当前先支持已经采集的三类：
    1. task8_original
    2. task8_wrong_target_language_only
    3. task1_native_next_to_ramekin
    """
    condition = row["condition"]
    task_ids = str(row.get("task_ids", ""))
    custom_language = str(row.get("custom_language", "")).lower()

    bowl1_pos = get_vec(first_obs, "akita_black_bowl_1_pos")
    bowl2_pos = get_vec(first_obs, "akita_black_bowl_2_pos")
    plate_pos = get_vec(first_obs, "plate_1_pos")
    ramekin_pos = get_vec(first_obs, "glazed_rim_porcelain_ramekin_1_pos")

    # task8: bowl1 near plate, bowl2 near ramekin
    if task_ids == "8":
        default_source = "akita_black_bowl_1"
        default_target = PLATE

        if "ramekin" in custom_language and "place" in custom_language:
            # wrong target language: source still bowl next to plate, target becomes ramekin
            language_source = "akita_black_bowl_1"
            language_target = RAMEKIN
        else:
            language_source = "akita_black_bowl_1"
            language_target = PLATE

        return language_source, language_target, default_source, default_target

    # task1: pick bowl next to ramekin and place on plate.
    # Robustly infer source as the bowl closer to ramekin.
    if task_ids == "1":
        d1 = dist(bowl1_pos, ramekin_pos)
        d2 = dist(bowl2_pos, ramekin_pos)

        if d1 <= d2:
            language_source = "akita_black_bowl_1"
        else:
            language_source = "akita_black_bowl_2"

        language_target = PLATE
        default_source = language_source
        default_target = PLATE

        return language_source, language_target, default_source, default_target

    # fallback
    return "akita_black_bowl_1", PLATE, "akita_black_bowl_1", PLATE


def parse_episode(jsonl_path: Path):
    meta = {}
    steps = []

    with jsonl_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "meta":
                meta = obj

            elif typ == "step":
                steps.append(obj)

    return meta, steps


def compute_step_rows(row, horizon=8):
    jsonl_path = Path(row["file"])
    meta, steps = parse_episode(jsonl_path)

    if not steps:
        return [], {}

    first_obs = steps[0].get("obs_summary", {})
    lang_source, lang_target, default_source, default_target = infer_language_goal(row, first_obs)

    records = []

    for t, step in enumerate(steps):
        obs = step.get("obs_summary", {})

        eef = get_vec(obs, "robot0_eef_pos")
        lang_source_pos = get_vec(obs, f"{lang_source}_pos")
        lang_target_pos = get_vec(obs, f"{lang_target}_pos")
        default_source_pos = get_vec(obs, f"{default_source}_pos")
        default_target_pos = get_vec(obs, f"{default_target}_pos")

        gripper_qpos = get_vec(obs, "robot0_gripper_qpos")
        gripper_width = float(np.abs(gripper_qpos).sum()) if gripper_qpos is not None else float("nan")

        d_eef_lang_source = dist(eef, lang_source_pos)
        d_lang_source_target = dist(lang_source_pos, lang_target_pos)
        d_default_source_target = dist(default_source_pos, default_target_pos)
        d_lang_source_default_target = dist(lang_source_pos, default_target_pos)

        # Future horizon index
        t2 = min(t + horizon, len(steps) - 1)
        obs2 = steps[t2].get("obs_summary", {})

        eef2 = get_vec(obs2, "robot0_eef_pos")
        lang_source_pos2 = get_vec(obs2, f"{lang_source}_pos")
        lang_target_pos2 = get_vec(obs2, f"{lang_target}_pos")
        default_source_pos2 = get_vec(obs2, f"{default_source}_pos")
        default_target_pos2 = get_vec(obs2, f"{default_target}_pos")

        d_eef_lang_source2 = dist(eef2, lang_source_pos2)
        d_lang_source_target2 = dist(lang_source_pos2, lang_target_pos2)
        d_default_source_target2 = dist(default_source_pos2, default_target_pos2)
        d_lang_source_default_target2 = dist(lang_source_pos2, default_target_pos2)

        # Positive means progress
        source_approach_progress_h = d_eef_lang_source - d_eef_lang_source2
        language_target_progress_h = d_lang_source_target - d_lang_source_target2
        default_target_progress_h = d_default_source_target - d_default_source_target2
        lang_source_to_default_target_progress_h = (
            d_lang_source_default_target - d_lang_source_default_target2
        )

        # This is useful for wrong-target language:
        # positive means action is more default-goal-progressing than language-goal-progressing.
        default_minus_language_progress_h = (
            lang_source_to_default_target_progress_h - language_target_progress_h
        )

        action = step.get("processed_action", [])
        action = np.asarray(action, dtype=np.float32)

        records.append(
            {
                "file_name": row["file_name"],
                "condition": row["condition"],
                "task_ids": row["task_ids"],
                "custom_language": row["custom_language"],
                "step": t,
                "horizon": horizon,
                "reward": step.get("reward"),
                "done": step.get("done"),
                "language_source": lang_source,
                "language_target": lang_target,
                "default_source": default_source,
                "default_target": default_target,
                "d_eef_language_source": d_eef_lang_source,
                "d_language_source_to_language_target": d_lang_source_target,
                "d_default_source_to_default_target": d_default_source_target,
                "d_language_source_to_default_target": d_lang_source_default_target,
                "source_approach_progress_h": source_approach_progress_h,
                "language_target_progress_h": language_target_progress_h,
                "default_target_progress_h": default_target_progress_h,
                "language_source_to_default_target_progress_h": lang_source_to_default_target_progress_h,
                "default_minus_language_progress_h": default_minus_language_progress_h,
                "gripper_width": gripper_width,
                "action_l2": float(np.linalg.norm(action)) if action.size else float("nan"),
                "action_gripper": float(action[-1]) if action.size else float("nan"),
            }
        )

    final_obs = steps[-1].get("obs_summary", {})
    final_lang_source_pos = get_vec(final_obs, f"{lang_source}_pos")
    final_lang_target_pos = get_vec(final_obs, f"{lang_target}_pos")
    final_default_source_pos = get_vec(final_obs, f"{default_source}_pos")
    final_default_target_pos = get_vec(final_obs, f"{default_target}_pos")

    episode_summary = {
        "file_name": row["file_name"],
        "condition": row["condition"],
        "task_ids": row["task_ids"],
        "custom_language": row["custom_language"],
        "steps": len(steps),
        "success_by_reward": row["success_by_reward"],
        "language_source": lang_source,
        "language_target": lang_target,
        "default_source": default_source,
        "default_target": default_target,
        "final_d_language_source_to_language_target": dist(final_lang_source_pos, final_lang_target_pos),
        "final_d_default_source_to_default_target": dist(final_default_source_pos, final_default_target_pos),
        "mean_source_approach_progress_h": float(
            np.nanmean([r["source_approach_progress_h"] for r in records])
        ),
        "mean_language_target_progress_h": float(
            np.nanmean([r["language_target_progress_h"] for r in records])
        ),
        "mean_default_target_progress_h": float(
            np.nanmean([r["default_target_progress_h"] for r in records])
        ),
        "mean_default_minus_language_progress_h": float(
            np.nanmean([r["default_minus_language_progress_h"] for r in records])
        ),
    }

    return records, episode_summary


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
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing dump_index.csv: {index_csv}")

    index_rows = load_index(index_csv)

    all_step_rows = []
    all_episode_rows = []

    for row in index_rows:
        step_rows, episode_row = compute_step_rows(row, horizon=args.horizon)
        all_step_rows.extend(step_rows)
        all_episode_rows.append(episode_row)

    step_csv = dump_dir / f"progress_step_labels_h{args.horizon}.csv"
    episode_csv = dump_dir / f"progress_episode_summary_h{args.horizon}.csv"

    write_csv(step_csv, all_step_rows)
    write_csv(episode_csv, all_episode_rows)

    print("\nEpisode summaries:")
    for r in all_episode_rows:
        print(
            r["condition"],
            "| steps:",
            r["steps"],
            "| lang:",
            f"{r['language_source']}->{r['language_target']}",
            "| final_d_lang:",
            round(r["final_d_language_source_to_language_target"], 4),
            "| final_d_default:",
            round(r["final_d_default_source_to_default_target"], 4),
            "| mean_lang_progress:",
            round(r["mean_language_target_progress_h"], 5),
            "| mean_default_progress:",
            round(r["mean_default_target_progress_h"], 5),
            "| mean_default_minus_lang:",
            round(r["mean_default_minus_language_progress_h"], 5),
        )


if __name__ == "__main__":
    main()