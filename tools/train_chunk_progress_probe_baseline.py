from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict
import numpy as np


def read_csv(path: Path):
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def auc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

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
    y = np.asarray(y)
    pred = np.asarray(pred)

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if ss_tot <= 1e-12:
        return np.nan

    return 1.0 - ss_res / ss_tot


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    mean = np.nanmean(X_train, axis=0, keepdims=True)
    std = np.nanstd(X_train, axis=0, keepdims=True) + 1e-6

    X_train = np.nan_to_num((X_train - mean) / std)
    X_test = np.nan_to_num((X_test - mean) / std)

    X_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    eye = np.eye(X_train.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0

    w = np.linalg.solve(X_train.T @ X_train + alpha * eye, X_train.T @ y_train)
    return X_test @ w


def episode_split(rows, seed=0, train_ratio=0.7):
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
        X.append([to_float(r[c]) for c in feature_cols])
        y.append(to_float(r[label_col]))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def evaluate_regression(rows_train, rows_test, feature_cols, label_col):
    X_train, y_train = build_matrix(rows_train, feature_cols, label_col)
    X_test, y_test = build_matrix(rows_test, feature_cols, label_col)

    pred = ridge_predict(X_train, y_train, X_test, alpha=1.0)

    return {
        "label": label_col,
        "r2": r2_score(y_test, pred),
        "mae": mae(y_test, pred),
    }


def evaluate_binary(rows_train, rows_test, feature_cols, label_col):
    X_train, y_train = build_matrix(rows_train, feature_cols, label_col)
    X_test, y_test = build_matrix(rows_test, feature_cols, label_col)

    score = ridge_predict(X_train, y_train, X_test, alpha=1.0)
    pred = (score >= 0.5).astype(int)

    acc = float(np.mean(pred == y_test))
    auc = auc_score(y_test, score)

    return {
        "label": label_col,
        "acc": acc,
        "auc": auc,
        "positive_rate_test": float(np.mean(y_test)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_csv(Path(args.dataset_csv))

    state_cols = [
        c for c in rows[0].keys()
        if c.startswith("feat_d_")
        or c.startswith("feat_gripper")
        or c.startswith("feat_rel_")
    ]

    chunk_flat_cols = [
        c for c in rows[0].keys()
        if c.startswith("feat_chunk_flat_")
    ]

    chunk_summary_cols = [
        c for c in rows[0].keys()
        if c.startswith("feat_chunk_mean_")
        or c.startswith("feat_chunk_std_")
        or c.startswith("feat_chunk_first_")
        or c.startswith("feat_chunk_last_")
        or c in ["feat_chunk_l2_sum", "feat_chunk_diff_l2_sum"]
    ]

    feature_sets = {
        "state_only": state_cols,
        "chunk_summary_only": chunk_summary_cols,
        "chunk_flat_only": chunk_flat_cols,
        "state_chunk_summary": state_cols + chunk_summary_cols,
        "state_chunk_flat": state_cols + chunk_flat_cols,
    }

    train_rows, test_rows = episode_split(rows, seed=args.seed, train_ratio=0.7)

    print("num_rows:", len(rows))
    print("train_rows:", len(train_rows))
    print("test_rows:", len(test_rows))

    print("\nTest condition counts:")
    counts = defaultdict(int)
    for r in test_rows:
        counts[r["condition"]] += 1

    for k, v in sorted(counts.items()):
        print(k, v)

    regression_labels = [
        "label_language_target_progress_h",
        "label_default_minus_language_progress_h",
    ]

    binary_labels = [
        "y_language_progress_pos",
        "y_default_over_language",
    ]

    print("\n" + "=" * 100)
    print("Regression probes")
    print("=" * 100)

    for fs_name, cols in feature_sets.items():
        print("\nFeature set:", fs_name, "num_features:", len(cols))

        for label in regression_labels:
            result = evaluate_regression(train_rows, test_rows, cols, label)
            print(
                label,
                "| R2:",
                round(result["r2"], 4),
                "| MAE:",
                round(result["mae"], 6),
            )

    print("\n" + "=" * 100)
    print("Binary probes")
    print("=" * 100)

    for fs_name, cols in feature_sets.items():
        print("\nFeature set:", fs_name, "num_features:", len(cols))

        for label in binary_labels:
            result = evaluate_binary(train_rows, test_rows, cols, label)
            print(
                label,
                "| ACC:",
                round(result["acc"], 4),
                "| AUC:",
                round(result["auc"], 4),
                "| positive_rate_test:",
                round(result["positive_rate_test"], 4),
            )


if __name__ == "__main__":
    main()