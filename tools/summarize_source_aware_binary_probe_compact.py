from pathlib import Path
import argparse
from collections import defaultdict

from train_source_aware_binary_probe import (
    read_csv,
    get_feature_sets,
    split_by_episode,
    eval_binary,
)


MISMATCH_HOLDOUTS = [
    "task8_wrong_target_language_only",
    "task8_wrong_source_language_only",
    "task8_wrong_source_wrong_target_language_only",
]

STRESS_HOLDOUTS = [
    "task8_empty_language",
    "task8_unrelated_language",
]

FEATURES_TO_SHOW = [
    "state_only",
    "chunk_summary_only",
    "state_chunk_summary",
]

CHUNK_RANDOM_LABELS = [
    "y_chunk_default_goal_advantage",
    "y_chunk_source_approach_advantage",
    "y_chunk_source_displacement_advantage",
    "y_episode_default_against_language",
    "y_episode_timeout",
]

EARLY_RANDOM_LABELS = [
    "y_episode_default_against_language",
    "y_episode_timeout",
]


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "NA"


def class_count_table(rows):
    counts = defaultdict(lambda: defaultdict(int))

    # early dataset has duplicate rows for different prefix chunks.
    # Use prefix=1 only to avoid triple-counting episodes.
    rows = [r for r in rows if to_int(r.get("prefix_chunks", 1)) == 1]

    for r in rows:
        counts[r["condition"]][r["episode_class"]] += 1

    lines = []
    lines.append("## 1. Episode class counts\n")
    lines.append("| condition | language_goal_completed | default_goal_completed_against_language | timeout_or_repeated_failure | failure_or_no_goal_completed |")
    lines.append("|---|---:|---:|---:|---:|")

    for cond in sorted(counts):
        c = counts[cond]
        lines.append(
            f"| {cond} | "
            f"{c.get('language_goal_completed', 0)} | "
            f"{c.get('default_goal_completed_against_language', 0)} | "
            f"{c.get('timeout_or_repeated_failure', 0)} | "
            f"{c.get('failure_or_no_goal_completed', 0)} |"
        )

    return "\n".join(lines)


def result_row(label, split, feature_name, result):
    return (
        f"| {label} | {split} | {feature_name} | "
        f"{fmt(result['pos_rate'])} | "
        f"{fmt(result['acc'])} | "
        f"{fmt(result['f1_pos'])} | "
        f"{fmt(result['bal_acc'])} | "
        f"{fmt(result['auc'])} | "
        f"{fmt(result['pred_pos_rate'])} |"
    )


def chunk_random_summary(chunk_rows, seed):
    feature_sets = get_feature_sets(chunk_rows)
    train_rows, test_rows = split_by_episode(chunk_rows, seed=seed, train_ratio=0.7)

    lines = []
    lines.append("\n## 2. Chunk-level binary probe: random episode split\n")
    lines.append("| label | split | feature_set | test_pos_rate | acc | f1_pos | bal_acc | auc | pred_pos_rate |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    for label in CHUNK_RANDOM_LABELS:
        for feature_name in FEATURES_TO_SHOW:
            if feature_name not in feature_sets:
                continue
            result = eval_binary(train_rows, test_rows, feature_sets[feature_name], label)
            lines.append(result_row(label, "random", feature_name, result))

    return "\n".join(lines)


def early_random_summary(early_rows, seed):
    lines = []
    lines.append("\n## 3. Episode early-warning: random episode split\n")

    prefixes = sorted(set(to_int(r["prefix_chunks"]) for r in early_rows))

    for prefix in prefixes:
        rows = [r for r in early_rows if to_int(r["prefix_chunks"]) == prefix]
        feature_sets = get_feature_sets(rows)
        train_rows, test_rows = split_by_episode(rows, seed=seed, train_ratio=0.7)

        lines.append(f"\n### Prefix chunks = {prefix}\n")
        lines.append("| label | feature_set | test_pos_rate | acc | f1_pos | bal_acc | auc | pred_pos_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

        for label in EARLY_RANDOM_LABELS:
            for feature_name in FEATURES_TO_SHOW:
                if feature_name not in feature_sets:
                    continue
                result = eval_binary(train_rows, test_rows, feature_sets[feature_name], label)
                lines.append(
                    f"| {label} | {feature_name} | "
                    f"{fmt(result['pos_rate'])} | "
                    f"{fmt(result['acc'])} | "
                    f"{fmt(result['f1_pos'])} | "
                    f"{fmt(result['bal_acc'])} | "
                    f"{fmt(result['auc'])} | "
                    f"{fmt(result['pred_pos_rate'])} |"
                )

    return "\n".join(lines)


def early_holdout_summary(early_rows):
    lines = []
    lines.append("\n## 4. Focused leave-condition-out: mismatch detector\n")
    lines.append("Label: `y_episode_default_against_language`\n")
    lines.append("| prefix | holdout | test_pos_rate | feature_set | acc | f1_pos | bal_acc | auc | pred_pos_rate |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")

    prefixes = sorted(set(to_int(r["prefix_chunks"]) for r in early_rows))

    for prefix in prefixes:
        rows = [r for r in early_rows if to_int(r["prefix_chunks"]) == prefix]
        feature_sets = get_feature_sets(rows)

        for holdout in MISMATCH_HOLDOUTS:
            train_rows = [r for r in rows if r["condition"] != holdout]
            test_rows = [r for r in rows if r["condition"] == holdout]

            if not test_rows:
                continue

            for feature_name in FEATURES_TO_SHOW:
                if feature_name not in feature_sets:
                    continue

                result = eval_binary(
                    train_rows,
                    test_rows,
                    feature_sets[feature_name],
                    "y_episode_default_against_language",
                )

                lines.append(
                    f"| {prefix} | {holdout} | "
                    f"{fmt(result['pos_rate'])} | "
                    f"{feature_name} | "
                    f"{fmt(result['acc'])} | "
                    f"{fmt(result['f1_pos'])} | "
                    f"{fmt(result['bal_acc'])} | "
                    f"{fmt(result['auc'])} | "
                    f"{fmt(result['pred_pos_rate'])} |"
                )

    lines.append("\n## 5. Focused leave-condition-out: timeout detector\n")
    lines.append("Label: `y_episode_timeout`\n")
    lines.append("| prefix | holdout | test_pos_rate | feature_set | acc | f1_pos | bal_acc | auc | pred_pos_rate |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")

    for prefix in prefixes:
        rows = [r for r in early_rows if to_int(r["prefix_chunks"]) == prefix]
        feature_sets = get_feature_sets(rows)

        for holdout in STRESS_HOLDOUTS:
            train_rows = [r for r in rows if r["condition"] != holdout]
            test_rows = [r for r in rows if r["condition"] == holdout]

            if not test_rows:
                continue

            for feature_name in FEATURES_TO_SHOW:
                if feature_name not in feature_sets:
                    continue

                result = eval_binary(
                    train_rows,
                    test_rows,
                    feature_sets[feature_name],
                    "y_episode_timeout",
                )

                lines.append(
                    f"| {prefix} | {holdout} | "
                    f"{fmt(result['pos_rate'])} | "
                    f"{feature_name} | "
                    f"{fmt(result['acc'])} | "
                    f"{fmt(result['f1_pos'])} | "
                    f"{fmt(result['bal_acc'])} | "
                    f"{fmt(result['auc'])} | "
                    f"{fmt(result['pred_pos_rate'])} |"
                )

    return "\n".join(lines)


def quick_interpretation():
    return """
## 6. What to check

重点看：

1. `y_episode_default_against_language`
   - holdout `task8_wrong_target_language_only`
   - holdout `task8_wrong_source_language_only`
   - holdout `task8_wrong_source_wrong_target_language_only`

   如果训练集中有其他 mismatch 类型时，这三个 holdout 仍能较好预测，说明 verifier 有跨 mismatch 类型泛化。

2. `y_episode_timeout`
   - holdout `task8_empty_language`
   - holdout `task8_unrelated_language`

   如果能预测 timeout，说明 early-warning 不只是识别 wrong-goal，也能识别 execution instability。

3. feature comparison:
   - `state_only`
   - `chunk_summary_only`
   - `state_chunk_summary`

   如果 `chunk_summary_only` 有预测力，说明 early action chunk 本身含风险信号。
   如果 `state_chunk_summary` 明显优于 `state_only`，说明 chunk 对 verifier 有增益。
"""
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)

    chunk_csv = dump_dir / f"source_aware_chunk_dataset_h{args.horizon}.csv"
    early_csv = dump_dir / f"source_aware_early_dataset_h{args.horizon}_max{args.max_chunks}.csv"

    if not chunk_csv.exists():
        raise FileNotFoundError(f"Missing: {chunk_csv}")

    if not early_csv.exists():
        raise FileNotFoundError(f"Missing: {early_csv}")

    chunk_rows = read_csv(chunk_csv)
    early_rows = read_csv(early_csv)

    report_parts = [
        "# Compact Source-Aware Binary Probe Summary\n",
        f"Dump dir: `{dump_dir}`\n",
        class_count_table(early_rows),
        chunk_random_summary(chunk_rows, args.seed),
        early_random_summary(early_rows, args.seed),
        early_holdout_summary(early_rows),
        quick_interpretation(),
    ]

    report = "\n".join(report_parts)

    out_path = dump_dir / f"source_aware_binary_compact_h{args.horizon}_seed{args.seed}.md"
    out_path.write_text(report)

    print(report)
    print("\n" + "=" * 100)
    print(f"Wrote compact report: {out_path}")


if __name__ == "__main__":
    main()