from pathlib import Path
import argparse
import json
import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=2)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"))

    print(f"dump_dir = {dump_dir}")
    print(f"num_jsonl = {len(files)}")

    if not files:
        raise FileNotFoundError(f"No jsonl files found in {dump_dir}")

    p = files[0]
    print("\n" + "=" * 100)
    print(f"Inspecting file: {p}")
    print("=" * 100)

    n_steps_printed = 0

    with p.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            typ = obj.get("type")

            if typ == "meta":
                print("\n[META]")
                pprint.pprint(obj, width=120)

            elif typ == "chunk":
                arr = obj.get("raw_actions_chunk")
                print("\n[FIRST CHUNK]")
                try:
                    print(f"chunk length = {len(arr)}")
                    print(f"first action length = {len(arr[0])}")
                    print(f"first action = {arr[0]}")
                except Exception as e:
                    print(f"Cannot parse chunk shape: {e}")

            elif typ == "step" and n_steps_printed < args.max_steps:
                print("\n" + "-" * 100)
                print(f"[STEP {obj.get('step')}]")

                print("\nprocessed_action:")
                pprint.pprint(obj.get("processed_action"), width=120)

                print("\nreward / done:")
                print("reward =", obj.get("reward"))
                print("done =", obj.get("done"))

                print("\ninfo keys:")
                info = obj.get("info", {})
                if isinstance(info, dict):
                    print(list(info.keys()))
                    pprint.pprint(info, width=120)
                else:
                    print(type(info), info)

                print("\nobs_summary keys:")
                obs = obj.get("obs_summary", {})
                if isinstance(obs, dict):
                    print(list(obs.keys()))
                    for k, v in obs.items():
                        print(f"\nobs_summary[{k}]:")
                        pprint.pprint(v, width=120)
                else:
                    print(type(obs), obs)

                n_steps_printed += 1

            if n_steps_printed >= args.max_steps:
                break


if __name__ == "__main__":
    main()