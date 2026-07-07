from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict
import numpy as np


CLASS_NAMES = {
    0: "language_goal_completed",
    1: "default_goal_completed_against_language",
    2: "timeout_or_repeated_failure",
    3: "failure_or_no_goal_completed",
}


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
        return 0.0


def get_feature_cols(rows):
    keys = list(rows[0].keys())

    state_cols = [
        c for c in keys
        if c.startswith("feat_state_")
    ]

    chunk_summary_cols = [
        c for c in keys
        if c.startswith("feat_chunk_summary_")
        or c.startswith("feat_chunk_mask_")
    ]

    chunk_flat_cols = [
        c for c in keys
        if c.startswith("feat_chunk_flat_")
        or c.startswith("feat_chunk_mask_")
    ]

    return {
        "state_only": state_cols,
        "early_chunk_summary_only": chunk_summary_cols,
        "early_chunk_flat_only": chunk_flat_cols,
        "state_early_chunk_summary": state_cols + chunk_summary_cols,
        "state_early_chunk_flat": state_cols + chunk_flat_cols,
    }


def build_matrix(rows, feature_cols):
    X = []
    y = []

    for r in rows:
        X.append([to_float(r[c]) for c in feature_cols])
        y.append(int(float(r["class_id"])))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def standardize(X_train, X_test):
    mean = np.nanmean(X_train, axis=0, keepdims=True)
    std = np.nanstd(X_train, axis=0, keepdims=True) + 1e-6

    X_train = np.nan_to_num((X_train - mean) / std)
    X_test = np.nan_to_num((X_test - mean) / std)

    return X_train, X_test


def ridge_multiclass_predict(X_train, y_train, X_test, alpha=1.0, num_classes=4):
    X_train, X_test = standardize(X_train, X_test)

    Y = np.zeros((len(y_train), num_classes), dtype=np.float32)
    for i, cls in enumerate(y_train):
        if 0 <= cls < num_classes:
            Y[i, cls] = 1.0

    X_train_aug = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test_aug = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    eye = np.eye(X_train_aug.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0

    W = np.linalg.solve(
        X_train_aug.T @ X_train_aug + alpha * eye,
        X_train_aug.T @ Y,
    )

    scores = X_test_aug @ W
    pred = scores.argmax(axis=1)

    return pred, scores


def confusion_matrix(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm


def macro_f1(y_true, y_pred, num_classes=4):
    f1s = []

    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        if np.sum(y_true == cls) == 0:
            continue

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)

    if not f1s:
        return 0.0

    return float(np.mean(f1s))


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def random_episode_split(rows, seed=0, train_ratio=0.7):
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


def evaluate_split(train_rows, test_rows, feature_cols, split_name):
    if not train_rows or not test_rows:
        return None

    X_train, y_train = build_matrix(train_rows, feature_cols)
    X_test, y_test = build_matrix(test_rows, feature_cols)

    train_classes = sorted(set(y_train.tolist()))
    test_classes = sorted(set(y_test.tolist()))

    pred, _ = ridge_multiclass_predict(
        X_train,
        y_train,
        X_test,
        alpha=1.0,
        num_classes=4,
    )

    result = {
        "split": split_name,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_classes": train_classes,
        "test_classes": test_classes,
        "acc": accuracy(y_test, pred),
        "macro_f1": macro_f1(y_test, pred, num_classes=4),
        "cm": confusion_matrix(y_test, pred, num_classes=4),
    }

    return result


def print_result(result):
    print(
        result["split"],
        "| n_train:",
        result["n_train"],
        "| n_test:",
        result["n_test"],
        "| train_classes:",
        result["train_classes"],
        "| test_classes:",
        result["test_classes"],
        "| ACC:",
        round(result["acc"], 4),
        "| macro-F1:",
        round(result["macro_f1"], 4),
    )

    print("confusion matrix rows=true, cols=pred")
    print(result["cm"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows_all = read_csv(Path(args.dataset_csv))
    rows_all = [r for r in rows_all if int(float(r["class_id"])) >= 0]

    feature_sets = get_feature_cols(rows_all)

    prefix_values = sorted(set(int(float(r["prefix_chunks"])) for r in rows_all))

    print("num_rows_total:", len(rows_all))
    print("prefix_values:", prefix_values)

    print("\nClass names:")
    for k, v in CLASS_NAMES.items():
        print(k, v)

    for prefix in prefix_values:
        rows = [r for r in rows_all if int(float(r["prefix_chunks"])) == prefix]

        print("\n" + "=" * 120)
        print("PREFIX CHUNKS =", prefix, "| num_rows =", len(rows))
        print("=" * 120)

        class_counts = defaultdict(int)
        condition_counts = defaultdict(int)

        for r in rows:
            class_counts[r["episode_class"]] += 1
            condition_counts[r["condition"]] += 1

        print("Class counts:", dict(class_counts))
        print("Condition counts:", dict(condition_counts))

        train_rows, test_rows = random_episode_split(
            rows,
            seed=args.seed,
            train_ratio=0.7,
        )

        print("\n" + "-" * 100)
        print("Random episode split")
        print("-" * 100)

        for fs_name, cols in feature_sets.items():
            result = evaluate_split(
                train_rows,
                test_rows,
                cols,
                split_name=f"random/{fs_name}",
            )

            print_result(result)

        print("\n" + "-" * 100)
        print("Leave-condition-out split")
        print("-" * 100)

        conditions = sorted(set(r["condition"] for r in rows))

        for holdout in conditions:
            train_rows_lco = [r for r in rows if r["condition"] != holdout]
            test_rows_lco = [r for r in rows if r["condition"] == holdout]

            print("\nHOLDOUT CONDITION:", holdout)

            for fs_name, cols in feature_sets.items():
                result = evaluate_split(
                    train_rows_lco,
                    test_rows_lco,
                    cols,
                    split_name=f"holdout={holdout}/{fs_name}",
                )

                print_result(result)


if __name__ == "__main__":
    main()