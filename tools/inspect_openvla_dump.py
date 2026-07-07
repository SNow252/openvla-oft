from pathlib import Path
import json
import argparse
import numpy as np


def shape_of(x):
    try:
        return np.asarray(x).shape
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="/tmp/openvla_oft_traj_dump_test",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"))

    print(f"dump_dir = {dump_dir}")
    print(f"num_files = {len(files)}")

    for p in files:
        n_meta = 0
        n_chunk = 0
        n_step = 0

        chunk_shapes = []
        raw_action_shapes = []
        processed_action_shapes = []
        rewards = []
        dones = []

        print("\n" + "=" * 80)
        print(p)

        with p.open("r", errors="ignore") as f:
            for line in f:
                obj = json.loads(line)
                typ = obj.get("type")

                if typ == "meta":
                    n_meta += 1
                    print("META:", obj)

                elif typ == "chunk":
                    n_chunk += 1
                    arr = obj.get("raw_actions_chunk")
                    chunk_shapes.append(shape_of(arr))

                elif typ == "step":
                    n_step += 1
                    raw_action_shapes.append(shape_of(obj.get("raw_action_before_process")))
                    processed_action_shapes.append(shape_of(obj.get("processed_action")))
                    rewards.append(obj.get("reward"))
                    dones.append(obj.get("done"))

        print(f"meta={n_meta}, chunks={n_chunk}, steps={n_step}")
        print("unique chunk shapes:", sorted(set(map(str, chunk_shapes))))
        print("unique raw action shapes:", sorted(set(map(str, raw_action_shapes))))
        print("unique processed action shapes:", sorted(set(map(str, processed_action_shapes))))
        print("reward values:", sorted(set(map(str, rewards))))
        print("done values:", sorted(set(map(str, dones))))


if __name__ == "__main__":
    main()