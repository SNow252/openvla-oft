from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict
import math
import numpy as np


MISMATCH_CONDITIONS = [
    "task8_wrong_target_language_only",
    "task8_wrong_source_language_only",
    "task8_wrong_source_wrong_target_language_only",
]

CLASS_NAMES = {
    0: "language_goal_completed",
    1: "default_goal_completed_against_language",
    2: "timeout_or_repeated_failure",
    3: "failure_or_no_goal_completed",
}


def read_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def to_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def fmt(x, nd=4):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x):
        return "NA"
    return f"{x:.{nd}f}"


def mean(xs):
    xs = [float(x) for x in xs if not math.isnan(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    xs = [float(x) for x in xs if not math.isnan(float(x))]
    return float(np.median(xs)) if xs else float("nan")


def auc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    total = 0
    correct = 0.0

    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                correct += 1
            elif p == n:
                correct += 0.5

    return correct / total


def r2_score(y, pred):
    y = np.asarray(y, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if ss_tot <= 1e-12:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    mean_x = np.nanmean(X_train, axis=0, keepdims=True)
    std_x = np.nanstd(X_train, axis=0, keepdims=True) + 1e-6

    X_train = np.nan_to_num((X_train - mean_x) / std_x)
    X_test = np.nan_to_num((X_test - mean_x) / std_x)

    X_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    eye = np.eye(X_train.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0

    w = np.linalg.solve(X_train.T @ X_train + alpha * eye, X_train.T @ y_train)
    return X_test @ w


def ridge_multiclass_predict(X_train, y_train, X_test, alpha=1.0, num_classes=4):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    mean_x = np.nanmean(X_train, axis=0, keepdims=True)
    std_x = np.nanstd(X_train, axis=0, keepdims=True) + 1e-6

    X_train = np.nan_to_num((X_train - mean_x) / std_x)
    X_test = np.nan_to_num((X_test - mean_x) / std_x)

    Y = np.zeros((len(y_train), num_classes), dtype=np.float32)
    for i, cls in enumerate(y_train):
        if 0 <= cls < num_classes:
            Y[i, cls] = 1.0

    X_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    eye = np.eye(X_train.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0

    W = np.linalg.solve(X_train.T @ X_train + alpha * eye, X_train.T @ Y)
    scores = X_test @ W
    pred = scores.argmax(axis=1)

    return pred, scores


def macro_f1(y_true, y_pred, num_classes=4):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    f1s = []

    for cls in range(num_classes):
        if np.sum(y_true == cls) == 0:
            continue

        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(y_true == y_pred))


def split_by_episode(rows, seed=0, train_ratio=0.7):
    grouped = defaultdict(list)

    for r in rows:
        grouped[r["condition"]].append(r["file_name"])

    train_files = set()
    test_files = set()

    rng = random.Random(seed)

    for cond, files in grouped.items():
        unique_files = sorted(set(files))
        rng.shuffle(unique_files)

        n_train = max(1, int(len(unique_files) * train_ratio))
        train_files.update(unique_files[:n_train])
        test_files.update(unique_files[n_train:])

    train_rows = [r for r in rows if r["file_name"] in train_files]
    test_rows = [r for r in rows if r["file_name"] in test_files]

    return train_rows, test_rows


def build_matrix(rows, feature_cols, label_col):
    X = []
    y = []
    for r in rows:
        X.append([to_float(r.get(c, 0.0), 0.0) for c in feature_cols])
        y.append(to_float(r.get(label_col, 0.0), 0.0))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_class_matrix(rows, feature_cols):
    X = []
    y = []
    for r in rows:
        X.append([to_float(r.get(c, 0.0), 0.0) for c in feature_cols])
        y.append(to_int(r.get("class_id", -1), -1))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def condition_index_summary(dump_dir):
    rows = read_csv(dump_dir / "dump_index.csv")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["condition"]].append(r)

    out = []
    out.append("## 1. Dump / success summary\n")
    out.append("| condition | n | success | success_rate | mean_steps | median_steps | timeout_like_steps>=200 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")

    for cond in sorted(grouped):
        rs = grouped[cond]
        n = len(rs)
        succ = sum(to_int(r["success_by_reward"]) for r in rs)
        steps = [to_float(r["steps"]) for r in rs]
        timeout = sum(1 for s in steps if s >= 200)

        out.append(
            f"| {cond} | {n} | {succ} | {succ / n:.2f} | "
            f"{mean(steps):.1f} | {median(steps):.1f} | {timeout} |"
        )

    return "\n".join(out)


def progress_condition_summary(dump_dir, horizon):
    path = dump_dir / f"progress_condition_summary_h{horizon}.csv"
    rows = read_csv(path)

    out = []
    out.append("\n## 2. Progress condition summary\n")

    if not rows:
        out.append(f"Missing: {path}")
        return "\n".join(out)

    out.append(
        "| condition | n | lang_done | default_against_lang | timeout | "
        "final_d_lang_mean | final_d_default_mean | mean_lang_progress | "
        "mean_default_progress | default_minus_lang |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        cond = r["condition"]
        n = to_int(r["num_episodes"])
        lang_done = to_int(r.get("count_language_goal_completed", 0))
        default_against = to_int(r.get("count_default_goal_completed_against_language", 0))
        timeout = to_int(r.get("count_timeout_or_repeated_failure", 0))

        out.append(
            f"| {cond} | {n} | {lang_done} | {default_against} | {timeout} | "
            f"{fmt(r.get('final_d_language_source_to_language_target_mean'))} | "
            f"{fmt(r.get('final_d_default_source_to_default_target_mean'))} | "
            f"{fmt(r.get('mean_language_target_progress_h_mean'))} | "
            f"{fmt(r.get('mean_default_target_progress_h_mean'))} | "
            f"{fmt(r.get('mean_default_minus_language_progress_h_mean'))} |"
        )

    return "\n".join(out)


def get_chunk_feature_sets(rows):
    keys = list(rows[0].keys())

    state_cols = [
        c for c in keys
        if c.startswith("feat_d_")
        or c.startswith("feat_gripper")
        or c.startswith("feat_rel_")
    ]

    chunk_summary_cols = [
        c for c in keys
        if c.startswith("feat_chunk_mean_")
        or c.startswith("feat_chunk_std_")
        or c.startswith("feat_chunk_first_")
        or c.startswith("feat_chunk_last_")
        or c in ["feat_chunk_l2_sum", "feat_chunk_diff_l2_sum"]
    ]

    return {
        "state_only": state_cols,
        "chunk_summary_only": chunk_summary_cols,
        "state_chunk_summary": state_cols + chunk_summary_cols,
    }


def chunk_probe_summary(dump_dir, seed):
    path = dump_dir / "chunk_progress_probe_dataset_h8.csv"
    rows = read_csv(path)

    out = []
    out.append("\n## 3. Chunk-level progress probe summary\n")

    if not rows:
        out.append(f"Missing: {path}")
        return "\n".join(out)

    feature_sets = get_chunk_feature_sets(rows)
    train_rows, test_rows = split_by_episode(rows, seed=seed, train_ratio=0.7)

    out.append("| feature_set | lang_progress_R2 | lang_progress_MAE | default_minus_lang_R2 | y_lang_progress_AUC | y_default_over_lang_AUC |")
    out.append("|---|---:|---:|---:|---:|---:|")

    for fs_name, cols in feature_sets.items():
        X_train, y_train = build_matrix(train_rows, cols, "label_language_target_progress_h")
        X_test, y_test = build_matrix(test_rows, cols, "label_language_target_progress_h")
        pred_lang = ridge_predict(X_train, y_train, X_test)

        X_train2, y_train2 = build_matrix(train_rows, cols, "label_default_minus_language_progress_h")
        X_test2, y_test2 = build_matrix(test_rows, cols, "label_default_minus_language_progress_h")
        pred_default_minus = ridge_predict(X_train2, y_train2, X_test2)

        X_train3, y_train3 = build_matrix(train_rows, cols, "y_language_progress_pos")
        X_test3, y_test3 = build_matrix(test_rows, cols, "y_language_progress_pos")
        score_lang_pos = ridge_predict(X_train3, y_train3, X_test3)

        X_train4, y_train4 = build_matrix(train_rows, cols, "y_default_over_language")
        X_test4, y_test4 = build_matrix(test_rows, cols, "y_default_over_language")
        score_default_over = ridge_predict(X_train4, y_train4, X_test4)

        out.append(
            f"| {fs_name} | "
            f"{r2_score(y_test, pred_lang):.4f} | {mae(y_test, pred_lang):.5f} | "
            f"{r2_score(y_test2, pred_default_minus):.4f} | "
            f"{auc_score(y_test3, score_lang_pos):.4f} | "
            f"{auc_score(y_test4, score_default_over):.4f} |"
        )

    return "\n".join(out)


def get_early_feature_sets(rows):
    keys = list(rows[0].keys())

    state_cols = [c for c in keys if c.startswith("feat_state_")]

    chunk_summary_cols = [
        c for c in keys
        if c.startswith("feat_chunk_summary_")
        or c.startswith("feat_chunk_mask_")
    ]

    return {
        "state_only": state_cols,
        "early_chunk_summary_only": chunk_summary_cols,
        "state_early_chunk_summary": state_cols + chunk_summary_cols,
    }


def multiclass_eval(train_rows, test_rows, feature_cols):
    X_train, y_train = build_class_matrix(train_rows, feature_cols)
    X_test, y_test = build_class_matrix(test_rows, feature_cols)

    pred, _ = ridge_multiclass_predict(X_train, y_train, X_test, num_classes=4)

    return accuracy(y_test, pred), macro_f1(y_test, pred, num_classes=4)


def class_count_string(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["episode_class"]] += 1
    parts = []
    for k in sorted(counts):
        parts.append(f"{k}:{counts[k]}")
    return ", ".join(parts)


def early_warning_summary(dump_dir, seed, focus_holdouts):
    path = dump_dir / "episode_early_warning_dataset_h8_max4.csv"
    rows_all = read_csv(path)

    out = []
    out.append("\n## 4. Episode early-warning summary\n")

    if not rows_all:
        out.append(f"Missing: {path}")
        return "\n".join(out)

    rows_all = [r for r in rows_all if to_int(r.get("class_id", -1), -1) >= 0]
    feature_sets = get_early_feature_sets(rows_all)

    prefixes = sorted(set(to_int(r["prefix_chunks"]) for r in rows_all))

    for prefix in prefixes:
        rows = [r for r in rows_all if to_int(r["prefix_chunks"]) == prefix]
        out.append(f"\n### Prefix chunks = {prefix}\n")
        out.append(f"Class counts: {class_count_string(rows)}\n")

        train_rows, test_rows = split_by_episode(rows, seed=seed, train_ratio=0.7)

        out.append("#### Random episode split\n")
        out.append("| feature_set | acc | macro_F1 |")
        out.append("|---|---:|---:|")

        for fs_name, cols in feature_sets.items():
            acc, f1 = multiclass_eval(train_rows, test_rows, cols)
            out.append(f"| {fs_name} | {acc:.4f} | {f1:.4f} |")

        out.append("\n#### Focused leave-condition-out\n")
        out.append("| holdout | test_classes | feature_set | acc | macro_F1 |")
        out.append("|---|---|---|---:|---:|")

        for holdout in focus_holdouts:
            train_h = [r for r in rows if r["condition"] != holdout]
            test_h = [r for r in rows if r["condition"] == holdout]

            if not test_h:
                continue

            test_classes = sorted(set(r["episode_class"] for r in test_h))
            test_class_str = ", ".join(test_classes)

            for fs_name, cols in feature_sets.items():
                acc, f1 = multiclass_eval(train_h, test_h, cols)
                out.append(f"| {holdout} | {test_class_str} | {fs_name} | {acc:.4f} | {f1:.4f} |")

    return "\n".join(out)


def interpretation_block():
    return """
## 5. Quick interpretation checklist

重点看三件事：

1. `wrong_source / wrong_source_wrong_target` 是否也产生 `default_goal_completed_against_language`。
   - 如果是，class 1 不再只来自 wrong_target，leave-condition-out 才有意义。

2. Chunk-level probe 中：
   - `state_chunk_summary` 是否明显优于 `state_only`。
   - `chunk_summary_only` 是否仍有预测力。
   - 如果二者成立，说明 action chunk 对 progress verification 有额外信息。

3. Early-warning focused holdout 中：
   - holdout `task8_wrong_target_language_only`
   - holdout `task8_wrong_source_language_only`
   - holdout `task8_wrong_source_wrong_target_language_only`

如果训练中有其他 mismatch 类型时，模型能在 holdout mismatch 上识别 `default_goal_completed_against_language`，说明路线 A 仍可继续。
如果 holdout mismatch 仍然全崩，说明 verifier 很可能学的是 condition/template identity，应降级为诊断工具。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--focus_holdouts",
        type=str,
        default="task8_wrong_target_language_only,task8_wrong_source_language_only,task8_wrong_source_wrong_target_language_only,task8_empty_language,task8_unrelated_language",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    focus_holdouts = [x.strip() for x in args.focus_holdouts.split(",") if x.strip()]

    lines = []
    lines.append("# Compact mismatch-v2 result summary\n")
    lines.append(f"Dump dir: `{dump_dir}`\n")

    lines.append(condition_index_summary(dump_dir))
    lines.append(progress_condition_summary(dump_dir, args.horizon))
    lines.append(chunk_probe_summary(dump_dir, args.seed))
    lines.append(early_warning_summary(dump_dir, args.seed, focus_holdouts))
    lines.append(interpretation_block())

    report = "\n".join(lines)

    out_path = dump_dir / f"compact_summary_h{args.horizon}_seed{args.seed}.md"
    out_path.write_text(report)

    print(report)
    print("\n" + "=" * 100)
    print(f"Wrote compact report: {out_path}")


if __name__ == "__main__":
    main()