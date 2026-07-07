from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"))

    print("dump_dir:", dump_dir)
    print("num_jsonl:", len(files))

    if not files:
        raise FileNotFoundError(f"No jsonl found in {dump_dir}")

    p = files[0]
    print("file:", p)

    found_chunk = False
    found_step = False

    with p.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            if obj.get("type") == "chunk" and not found_chunk:
                print("\nFIRST CHUNK KEYS:")
                print(sorted(obj.keys()))
                print("has query_obs_summary:", "query_obs_summary" in obj)

                if "query_obs_summary" in obj:
                    print("query_obs_summary keys:")
                    print(sorted(obj["query_obs_summary"].keys())[:20])

                arr = obj.get("raw_actions_chunk")
                print("chunk len:", len(arr))
                print("action dim:", len(arr[0]))

                found_chunk = True

            if obj.get("type") == "step" and not found_step:
                print("\nFIRST STEP KEYS:")
                print(sorted(obj.keys()))
                print("has obs_before_summary:", "obs_before_summary" in obj)
                print("has obs_after_summary:", "obs_after_summary" in obj)

                if "obs_before_summary" in obj:
                    print("obs_before_summary keys:")
                    print(sorted(obj["obs_before_summary"].keys())[:20])

                if "obs_after_summary" in obj:
                    print("obs_after_summary keys:")
                    print(sorted(obj["obs_after_summary"].keys())[:20])

                found_step = True

            if found_chunk and found_step:
                break


if __name__ == "__main__":
    main()