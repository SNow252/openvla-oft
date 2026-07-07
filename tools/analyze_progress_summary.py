from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict
import numpy as np


METRICS = [
    "steps",
    "final_d_language_source_to_language_target",
    "final_d_default_source_to_default_target",
    "mean_source_approach_progress_h",
    "mean_language_target_progress_h",
    "mean_default_target_progress_h",
    "mean_default_minus_language_progress_h",
]


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_csv(path):
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def classify_episode(row):
    final_lang = to_float(row["final_d_language_source_to_language_target"])
    final_default = to_float(row["final_d_default_source_to_default_target"])
    steps = to_float(row["steps"])

    language_done = final_lang < 0.05
    default_done = final_default < 0.05
    timeout_like = steps >= 200

    if language_done:
        return "language_goal_completed"

    if default_done and (not language_done):
        return "default_goal_completed_against_language"

    if timeout_like:
        return "timeout_or_repeated_failure"

    return "failure_or_no_goal_completed"


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.std(xs)) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    episode_csv = dump_dir / f"progress_episode_summary_h{args.horizon}.csv"

    if not episode_csv.exists():
        raise FileNotFoundError(f"Missing file: {episode_csv}")

    rows = read_csv(episode_csv)

    for row in rows:
        row["episode_class"] = classify_episode(row)

    print("=" * 100)
    print("Episode-level classification")
    print("=" * 100)

    for row in rows:
        print(
            row["condition"],
            "| class:",
            row["episode_class"],
            "| steps:",
            row["steps"],
            "| final_d_lang:",
            row["final_d_language_source_to_language_target"],
            "| final_d_default:",
            row["final_d_default_source_to_default_target"],
            "| mean_lang_progress:",
            row["mean_language_target_progress_h"],
            "| mean_default_progress:",
            row["mean_default_target_progress_h"],
            "| default_minus_lang:",
            row["mean_default_minus_language_progress_h"],
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    print("\n" + "=" * 100)
    print("Condition-level metric summary")
    print("=" * 100)

    summary_rows = []

    for cond, cond_rows in grouped.items():
        class_counts = defaultdict(int)
        for r in cond_rows:
            class_counts[r["episode_class"]] += 1

        print("\nCondition:", cond)
        print("num_episodes:", len(cond_rows))
        print("class_counts:", dict(class_counts))

        summary = {
            "condition": cond,
            "num_episodes": len(cond_rows),
        }

        for cls in [
            "language_goal_completed",
            "default_goal_completed_against_language",
            "timeout_or_repeated_failure",
            "failure_or_no_goal_completed",
        ]:
            summary[f"count_{cls}"] = class_counts[cls]

        for m in METRICS:
            vals = [to_float(r[m]) for r in cond_rows]
            summary[f"{m}_mean"] = mean(vals)
            summary[f"{m}_median"] = median(vals)
            summary[f"{m}_std"] = std(vals)

            print(
                m,
                "| mean:",
                round(summary[f"{m}_mean"], 5),
                "| median:",
                round(summary[f"{m}_median"], 5),
                "| std:",
                round(summary[f"{m}_std"], 5),
            )

        summary_rows.append(summary)

    out_csv = dump_dir / f"progress_condition_summary_h{args.horizon}.csv"

    fieldnames = list(summary_rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    out_episode_csv = dump_dir / f"progress_episode_classified_h{args.horizon}.csv"
    fieldnames_ep = list(rows[0].keys())
    with out_episode_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ep)
        writer.writeheader()
        writer.writerows(rows)

    print("\nWrote:")
    print(out_csv)
    print(out_episode_csv)


if __name__ == "__main__":
    main()