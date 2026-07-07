from pathlib import Path
import argparse
import csv
import json


def infer_condition(meta):
    dump_condition = str(meta.get("dump_condition", ""))
    task_ids = str(meta.get("task_ids", ""))
    custom_language = str(meta.get("custom_language", ""))

    if dump_condition:
        return dump_condition

    if task_ids == "8" and custom_language == "":
        return "task8_original"

    if task_ids == "8" and "ramekin" in custom_language.lower():
        return "task8_wrong_target_language_only"

    if task_ids == "1" and custom_language == "":
        return "task1_native_next_to_ramekin"

    return "unknown"


def summarize_jsonl(path: Path):
    meta = {}
    n_chunk = 0
    n_step = 0
    rewards = []
    dones = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "meta":
                meta = obj

            elif typ == "chunk":
                n_chunk += 1

            elif typ == "step":
                n_step += 1
                rewards.append(obj.get("reward"))
                dones.append(obj.get("done"))

    reward_values = []
    for r in rewards:
        try:
            reward_values.append(float(r))
        except Exception:
            pass

    max_reward = max(reward_values) if reward_values else 0.0
    success = max_reward > 0.0
    done_true = any(str(d) == "True" or d is True for d in dones)

    return {
        "file": str(path),
        "file_name": path.name,
        "condition": infer_condition(meta),
        "dump_condition": meta.get("dump_condition", ""),
        "task_ids": meta.get("task_ids", ""),
        "custom_language": meta.get("custom_language", ""),
        "num_open_loop_steps": meta.get("num_open_loop_steps", ""),
        "chunks": n_chunk,
        "steps": n_step,
        "success_by_reward": int(success),
        "done_true": int(done_true),
        "max_reward": max_reward,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"))

    rows = [summarize_jsonl(p) for p in files]

    out_csv = dump_dir / "dump_index.csv"

    fieldnames = [
        "file",
        "file_name",
        "condition",
        "dump_condition",
        "task_ids",
        "custom_language",
        "num_open_loop_steps",
        "chunks",
        "steps",
        "success_by_reward",
        "done_true",
        "max_reward",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"dump_dir: {dump_dir}")
    print(f"num_jsonl: {len(files)}")
    print(f"wrote: {out_csv}")

    print("\nSummary:")
    for row in rows:
        print(
            row["condition"],
            "| task:",
            row["task_ids"],
            "| steps:",
            row["steps"],
            "| chunks:",
            row["chunks"],
            "| success:",
            row["success_by_reward"],
            "| max_reward:",
            row["max_reward"],
            "| custom_language:",
            repr(row["custom_language"]),
            "| file:",
            row["file_name"],
        )


if __name__ == "__main__":
    main()