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


def load_actions(jsonl_path: Path):
    processed_actions = []
    raw_actions = []
    chunks = []

    with jsonl_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "chunk":
                arr = np.asarray(obj["raw_actions_chunk"], dtype=np.float32)
                chunks.append(arr)

            elif typ == "step":
                processed_actions.append(
                    np.asarray(obj["processed_action"], dtype=np.float32)
                )
                raw_actions.append(
                    np.asarray(obj["raw_action_before_process"], dtype=np.float32)
                )

    processed_actions = np.asarray(processed_actions, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)

    return {
        "processed_actions": processed_actions,
        "raw_actions": raw_actions,
        "chunks": chunks,
    }


def flatten_prefix(actions, max_steps=None):
    if max_steps is not None:
        actions = actions[:max_steps]
    return actions.reshape(-1)


def cosine(a, b, eps=1e-8):
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)


def l2_mean(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


def l2_total(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(a[:n] - b[:n]))


def summarize_episode(actions):
    return {
        "steps": len(actions),
        "mean_abs_action": float(np.mean(np.abs(actions))),
        "std_action": float(np.std(actions)),
        "action_l2_path": float(np.linalg.norm(actions, axis=1).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "dump_index.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing dump_index.csv: {index_csv}")

    rows = load_index(index_csv)

    episodes = []
    for row in rows:
        path = Path(row["file"])
        data = load_actions(path)
        actions = data["processed_actions"]

        if actions.ndim != 2:
            print(f"WARNING: bad action shape {actions.shape} in {path}")
            continue

        ep = {
            "condition": row["condition"],
            "file_name": row["file_name"],
            "file": str(path),
            "actions": actions,
            "summary": summarize_episode(actions),
        }
        episodes.append(ep)

    print("=" * 100)
    print("Episode summary")
    print("=" * 100)

    for ep in episodes:
        s = ep["summary"]
        print(
            ep["condition"],
            "| steps:", s["steps"],
            "| mean_abs_action:", round(s["mean_abs_action"], 4),
            "| std_action:", round(s["std_action"], 4),
            "| action_l2_path:", round(s["action_l2_path"], 4),
            "| file:", ep["file_name"],
        )

    print("\n" + "=" * 100)
    print("Pairwise similarity, aligned by min length")
    print("=" * 100)

    pair_rows = []
    for i in range(len(episodes)):
        for j in range(i + 1, len(episodes)):
            a = episodes[i]["actions"]
            b = episodes[j]["actions"]
            n = min(len(a), len(b))

            af = flatten_prefix(a, n)
            bf = flatten_prefix(b, n)

            row = {
                "cond_i": episodes[i]["condition"],
                "cond_j": episodes[j]["condition"],
                "file_i": episodes[i]["file_name"],
                "file_j": episodes[j]["file_name"],
                "aligned_steps": n,
                "cosine": cosine(af, bf),
                "mean_step_l2": l2_mean(a, b),
                "total_l2": l2_total(a, b),
            }
            pair_rows.append(row)

            print(
                row["cond_i"],
                "<->",
                row["cond_j"],
                "| steps:",
                row["aligned_steps"],
                "| cosine:",
                round(row["cosine"], 4),
                "| mean_step_l2:",
                round(row["mean_step_l2"], 4),
                "| total_l2:",
                round(row["total_l2"], 4),
            )

    out_csv = dump_dir / "action_pairwise_similarity.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cond_i",
                "cond_j",
                "file_i",
                "file_j",
                "aligned_steps",
                "cosine",
                "mean_step_l2",
                "total_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    print("\nWrote:", out_csv)

    print("\n" + "=" * 100)
    print("Condition-level mean pairwise distance")
    print("=" * 100)

    grouped = {}
    for row in pair_rows:
        key = tuple(sorted([row["cond_i"], row["cond_j"]]))
        grouped.setdefault(key, []).append(row)

    for key, vals in grouped.items():
        mean_l2 = np.mean([v["mean_step_l2"] for v in vals])
        mean_cos = np.mean([v["cosine"] for v in vals])
        print(
            key,
            "| mean cosine:",
            round(float(mean_cos), 4),
            "| mean step L2:",
            round(float(mean_l2), 4),
            "| n_pairs:",
            len(vals),
        )


if __name__ == "__main__":
    main()