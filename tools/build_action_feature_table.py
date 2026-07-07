from pathlib import Path
import argparse
import csv
import json
import numpy as np


def load_index(index_csv: Path):
    rows = []
    with index_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_processed_actions(jsonl_path: Path):
    actions = []

    with jsonl_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            if obj.get("type") == "step":
                actions.append(np.asarray(obj["processed_action"], dtype=np.float32))

    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"Bad action shape {actions.shape} in {jsonl_path}")

    return actions


def resample_actions(actions, fixed_steps=100):
    old_t = np.linspace(0.0, 1.0, len(actions))
    new_t = np.linspace(0.0, 1.0, fixed_steps)

    out = []
    for d in range(actions.shape[1]):
        out.append(np.interp(new_t, old_t, actions[:, d]))

    return np.stack(out, axis=1).astype(np.float32)


def summarize_actions(actions):
    n = len(actions)

    early = actions[: max(1, n // 3)]
    middle = actions[n // 3 : max(n // 3 + 1, 2 * n // 3)]
    late = actions[2 * n // 3 :]

    diffs = np.diff(actions, axis=0)
    if len(diffs) == 0:
        diffs = np.zeros_like(actions[:1])

    feats = []

    feats.extend(actions.mean(axis=0))
    feats.extend(actions.std(axis=0))
    feats.extend(actions.min(axis=0))
    feats.extend(actions.max(axis=0))

    feats.extend(early.mean(axis=0))
    feats.extend(middle.mean(axis=0))
    feats.extend(late.mean(axis=0))

    feats.append(float(np.linalg.norm(actions, axis=1).sum()))
    feats.append(float(np.linalg.norm(diffs, axis=1).sum()))
    feats.append(float(len(actions)))

    return np.asarray(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--fixed_steps", type=int, default=100)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing dump_index.csv: {index_csv}")

    rows = load_index(index_csv)

    X_flat = []
    X_summary = []
    conditions = []
    file_names = []
    steps = []
    success = []

    for row in rows:
        path = Path(row["file"])
        actions = load_processed_actions(path)

        fixed = resample_actions(actions, fixed_steps=args.fixed_steps)
        summary = summarize_actions(actions)

        X_flat.append(fixed.reshape(-1))
        X_summary.append(summary)
        conditions.append(row["condition"])
        file_names.append(row["file_name"])
        steps.append(len(actions))
        success.append(int(row["success_by_reward"]))

    X_flat = np.asarray(X_flat, dtype=np.float32)
    X_summary = np.asarray(X_summary, dtype=np.float32)
    X_all = np.concatenate([X_flat, X_summary], axis=1)

    out_npz = dump_dir / f"action_features_fixed{args.fixed_steps}.npz"
    np.savez(
        out_npz,
        X_flat=X_flat,
        X_summary=X_summary,
        X_all=X_all,
        conditions=np.asarray(conditions),
        file_names=np.asarray(file_names),
        steps=np.asarray(steps),
        success=np.asarray(success),
    )

    out_csv = dump_dir / f"episode_feature_index_fixed{args.fixed_steps}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "condition",
                "file_name",
                "steps",
                "success",
            ],
        )
        writer.writeheader()

        for i, (cond, fn, st, suc) in enumerate(
            zip(conditions, file_names, steps, success)
        ):
            writer.writerow(
                {
                    "idx": i,
                    "condition": cond,
                    "file_name": fn,
                    "steps": st,
                    "success": suc,
                }
            )

    print(f"wrote npz: {out_npz}")
    print(f"wrote csv: {out_csv}")
    print(f"X_flat shape: {X_flat.shape}")
    print(f"X_summary shape: {X_summary.shape}")
    print(f"X_all shape: {X_all.shape}")

    print("\nCondition counts:")
    for cond in sorted(set(conditions)):
        print(cond, conditions.count(cond))


if __name__ == "__main__":
    main()