#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter


IMPORTANT_KEYS = [
    "init_state_idx",
    "initial_state_idx",
    "episode_idx",
    "episode_id",
    "seed",
    "task_id",
    "task_suite_name",
    "task_language",
    "language",
    "instruction",
    "condition",
    "condition_name",
    "variant",
    "success",
    "done",
    "reward",
    "timestep",
    "step",
    "raw_actions_chunk",
    "processed_action",
    "raw_action_before_process",
    "query_obs_summary",
    "obs_before_summary",
    "obs_after_summary",
    "obs_summary",
]


def read_jsonl(path, max_lines):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception as e:
                print(f"[json error] line={i}, error={e}")
    return records


def shape_of(x):
    try:
        import numpy as np
        return tuple(np.asarray(x).shape)
    except Exception:
        return None


def short_value(x, max_len=160):
    if isinstance(x, (int, float, bool)) or x is None:
        return repr(x)
    if isinstance(x, str):
        return repr(x[:max_len])
    s = repr(x)
    return s[:max_len]


def flatten_keys(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.append((p, v))
            out.extend(flatten_keys(v, p))
    elif isinstance(obj, list):
        # avoid expanding long numeric arrays
        if len(obj) > 0 and all(isinstance(z, (int, float, bool)) for z in obj[:min(len(obj), 20)]):
            return out
        for i, v in enumerate(obj[:10]):
            p = f"{prefix}[{i}]"
            out.append((p, v))
            out.extend(flatten_keys(v, p))
    return out


def last_key(path):
    return path.split(".")[-1].split("[")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--max_lines", type=int, default=12)
    args = parser.parse_args()

    files = sorted(Path(args.dump_dir).glob("*.jsonl"))[:args.max_files]

    print("=" * 120)
    print("DUMP_DIR:", args.dump_dir)
    print("FILES:", len(files))
    print("=" * 120)

    for fi, path in enumerate(files):
        print("\n" + "#" * 120)
        print(f"[file {fi}] {path.name}")
        print("#" * 120)

        records = read_jsonl(path, args.max_lines)
        print("num_records_read:", len(records))

        for li, rec in enumerate(records):
            print("\n" + "-" * 100)
            print(f"[line {li}] top-level keys:")
            print(sorted(rec.keys()))

            flat = flatten_keys(rec)

            print("\nImportant fields:")
            found = False
            for p, v in flat:
                lk = last_key(p)
                if lk in IMPORTANT_KEYS:
                    found = True
                    sh = shape_of(v)
                    if sh is not None and sh != ():
                        print(f"  {p:60s} shape={str(sh):14s} value={short_value(v)}")
                    else:
                        print(f"  {p:60s} value={short_value(v)}")
            if not found:
                print("  <none>")

            print("\nAction-like fields:")
            for p, v in flat:
                sh = shape_of(v)
                if sh in [(7,), (8, 7)]:
                    print(f"  {p:60s} shape={sh}")

            print("\nSummary-like fields:")
            for p, v in flat:
                lk = last_key(p)
                if lk.endswith("_summary") or lk in [
                    "robot0_eef_pos",
                    "akita_black_bowl_1_pos",
                    "akita_black_bowl_2_pos",
                    "plate_1_pos",
                    "glazed_rim_porcelain_ramekin_1_pos",
                ]:
                    sh = shape_of(v)
                    print(f"  {p:60s} shape={sh} value={short_value(v, 80)}")


if __name__ == "__main__":
    main()