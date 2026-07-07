from pathlib import Path
import argparse
import numpy as np


def zscore(X, eps=1e-6):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def l2(a, b):
    return float(np.linalg.norm(a - b))


def nearest_centroid(x, centroids):
    dists = {name: l2(x, c) for name, c in centroids.items()}
    pred = min(dists, key=dists.get)
    return pred, dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_npz", type=str, required=True)
    parser.add_argument(
        "--feature_type",
        type=str,
        default="X_all",
        choices=["X_all", "X_flat", "X_summary"],
    )
    args = parser.parse_args()

    data = np.load(args.feature_npz, allow_pickle=True)

    X = data[args.feature_type].astype(np.float32)
    conditions = data["conditions"].astype(str)
    file_names = data["file_names"].astype(str)

    X = zscore(X)

    template_labels = []
    known_indices = []

    for i, cond in enumerate(conditions):
        if cond == "task8_original":
            template_labels.append("task8_template")
            known_indices.append(i)
        elif cond == "task1_native_next_to_ramekin":
            template_labels.append("task1_template")
            known_indices.append(i)

    known_indices = np.asarray(known_indices)
    template_labels = np.asarray(template_labels)

    print("=" * 100)
    print("Known-template leave-one-out nearest-centroid check")
    print("=" * 100)

    correct = 0
    total = 0

    for idx in known_indices:
        true_label = (
            "task8_template"
            if conditions[idx] == "task8_original"
            else "task1_template"
        )

        train_mask = known_indices != idx
        train_indices = known_indices[train_mask]
        train_labels = template_labels[train_mask]

        centroids = {}
        for label in sorted(set(train_labels)):
            label_indices = train_indices[train_labels == label]
            centroids[label] = X[label_indices].mean(axis=0)

        pred, dists = nearest_centroid(X[idx], centroids)
        ok = pred == true_label

        correct += int(ok)
        total += 1

        print(
            f"idx={idx:02d}",
            f"true={true_label}",
            f"pred={pred}",
            f"ok={ok}",
            f"d_task8={dists.get('task8_template', float('nan')):.3f}",
            f"d_task1={dists.get('task1_template', float('nan')):.3f}",
            f"file={file_names[idx]}",
        )

    acc = correct / total if total else 0.0
    print(f"\nKnown-template LOOCV accuracy: {correct}/{total} = {acc:.3f}")

    print("\n" + "=" * 100)
    print("Classify task8_wrong_target_language_only")
    print("=" * 100)

    task8_indices = np.where(conditions == "task8_original")[0]
    task1_indices = np.where(conditions == "task1_native_next_to_ramekin")[0]
    wrong_indices = np.where(conditions == "task8_wrong_target_language_only")[0]

    centroids = {
        "task8_template": X[task8_indices].mean(axis=0),
        "task1_template": X[task1_indices].mean(axis=0),
    }

    pred_counts = {}

    for idx in wrong_indices:
        pred, dists = nearest_centroid(X[idx], centroids)
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

        margin = dists["task1_template"] - dists["task8_template"]

        print(
            f"idx={idx:02d}",
            f"pred={pred}",
            f"d_task8={dists['task8_template']:.3f}",
            f"d_task1={dists['task1_template']:.3f}",
            f"margin_task1_minus_task8={margin:.3f}",
            f"file={file_names[idx]}",
        )

    print("\nWrong-target prediction counts:")
    for k, v in sorted(pred_counts.items()):
        print(k, v)

    print("\nInterpretation:")
    print(
        "If most wrong-target episodes are classified as task8_template, "
        "then action features support the hypothesis that wrong-target language "
        "still executes the task8 default template."
    )


if __name__ == "__main__":
    main()