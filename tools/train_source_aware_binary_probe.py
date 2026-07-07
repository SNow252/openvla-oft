from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict
import numpy as np


FOCUS_HOLDOUTS = [
    "task8_wrong_target_language_only",
    "task8_wrong_source_language_only",
    "task8_wrong_source_wrong_target_language_only",
    "task8_empty_language",
    "task8_unrelated_language",
]


def read_csv(path: Path):
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def get_feature_sets(rows):
    keys = list(rows[0].keys())

    state_cols = [c for c in keys if c.startswith("feat_state_")]

    chunk_cols = [
        c for c in keys
        if c.startswith("feat_chunk_summary_")
        or c.startswith("feat_chunk_mask_")
    ]

    return {
        "state_only": state_cols,
        "chunk_summary_only": chunk_cols,
        "state_chunk_summary": state_cols + chunk_cols,
    }


def build_matrix(rows, feature_cols, label_col):
    X = []
    y = []

    for r in rows:
        X.append([to_float(r.get(c, 0.0), 0.0) for c in feature_cols])
        y.append(to_int(r.get(label_col, 0), 0))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def standardize(X_train, X_test):
    mean = np.nanmean(X_train, axis=0, keepdims=True)
    std = np.nanstd(X_train, axis=0, keepdims=True) + 1e-6

    X_train = np.nan_to_num((X_train - mean) / std)
    X_test = np.nan_to_num((X_test - mean) / std)

    return X_train, X_test


def ridge_binary_score(X_train, y_train, X_test, alpha=1.0):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    X_train, X_test = standardize(X_train, X_test)

    X_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    eye = np.eye(X_train.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0

    w = np.linalg.solve(X_train.T @ X_train + alpha * eye, X_train.T @ y_train)
    return X_test @ w


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


def binary_metrics(y_true, score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score >= threshold).astype(int)

    acc = float(np.mean(pred == y_true))

    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if np.sum(y_true == 1) > 0:
        tpr = tp / (tp + fn + 1e-9)
    else:
        tpr = float("nan")

    if np.sum(y_true == 0) > 0:
        tnr = tn / (tn + fp + 1e-9)
    else:
        tnr = float("nan")

    if np.isnan(tpr):
        bal_acc = tnr
    elif np.isnan(tnr):
        bal_acc = tpr
    else:
        bal_acc = 0.5 * (tpr + tnr)

    return {
        "acc": acc,
        "f1_pos": float(f1),
        "bal_acc": float(bal_acc),
        "auc": auc_score(y_true, score),
        "pos_rate": float(np.mean(y_true)),
        "pred_pos_rate": float(np.mean(pred)),
    }


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


def eval_binary(train_rows, test_rows, feature_cols, label_col):
    X_train, y_train = build_matrix(train_rows, feature_cols, label_col)
    X_test, y_test = build_matrix(test_rows, feature_cols, label_col)

    score = ridge_binary_score(X_train, y_train, X_test, alpha=1.0)
    return binary_metrics(y_test, score)


def class_counts(rows, label_col):
    pos = sum(to_int(r.get(label_col, 0), 0) for r in rows)
    return pos, len(rows) - pos


def print_table_header():
    print("| label | split | feature_set | n_train | n_test | test_pos_rate | acc | f1_pos | bal_acc | auc | pred_pos_rate |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def print_result(label, split, fs_name, n_train, n_test, result):
    print(
        f"| {label} | {split} | {fs_name} | {n_train} | {n_test} | "
        f"{result['pos_rate']:.3f} | {result['acc']:.3f} | {result['f1_pos']:.3f} | "
        f"{result['bal_acc']:.3f} | {result['auc']:.3f} | {result['pred_pos_rate']:.3f} |"
    )


def run_random(rows, feature_sets, labels, seed):
    train_rows, test_rows = split_by_episode(rows, seed=seed, train_ratio=0.7)

    print("\n## Random episode split\n")
    print_table_header()

    for label in labels:
        for fs_name, cols in feature_sets.items():
            result = eval_binary(train_rows, test_rows, cols, label)
            print_result(label, "random", fs_name, len(train_rows), len(test_rows), result)


def run_holdout(rows, feature_sets, labels):
    print("\n## Focused leave-condition-out\n")
    print_table_header()

    for holdout in FOCUS_HOLDOUTS:
        train_rows = [r for r in rows if r["condition"] != holdout]
        test_rows = [r for r in rows if r["condition"] == holdout]

        if not test_rows:
            continue

        for label in labels:
            train_pos, train_neg = class_counts(train_rows, label)
            test_pos, test_neg = class_counts(test_rows, label)

            split_name = f"holdout={holdout} train_pos={train_pos} test_pos={test_pos}"

            for fs_name, cols in feature_sets.items():
                result = eval_binary(train_rows, test_rows, cols, label)
                print_result(label, split_name, fs_name, len(train_rows), len(test_rows), result)


def run_early_probe(early_rows, seed):
    prefixes = sorted(set(to_int(r["prefix_chunks"]) for r in early_rows))
    labels = [
        "y_episode_default_against_language",
        "y_episode_timeout",
    ]

    print("\n# Episode-level early-warning binary probe\n")

    for prefix in prefixes:
        rows = [r for r in early_rows if to_int(r["prefix_chunks"]) == prefix]
        feature_sets = get_feature_sets(rows)

        print(f"\n# Prefix chunks = {prefix}\n")
        print("condition counts:")
        counts = defaultdict(int)
        for r in rows:
            counts[r["condition"]] += 1
        for k, v in sorted(counts.items()):
            print(f"- {k}: {v}")

        run_random(rows, feature_sets, labels, seed)
        run_holdout(rows, feature_sets, labels)


def run_chunk_probe(chunk_rows, seed):
    labels = [
        "y_chunk_default_goal_advantage",
        "y_chunk_source_approach_advantage",
        "y_chunk_source_displacement_advantage",
        "y_episode_default_against_language",
        "y_episode_timeout",
    ]

    feature_sets = get_feature_sets(chunk_rows)

    print("\n# Chunk-level source-aware binary probe\n")
    print("num_rows:", len(chunk_rows))

    run_random(chunk_rows, feature_sets, labels, seed)
    run_holdout(chunk_rows, feature_sets, labels)


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

    run_chunk_probe(chunk_rows, args.seed)
    run_early_probe(early_rows, args.seed)


if __name__ == "__main__":
    main()