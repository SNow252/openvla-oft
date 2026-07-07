from pathlib import Path
import re


PATCHES = [
    {
        "path": "tools/build_progress_labels.py",
        "func_name": "infer_language_goal",
        "signature": "row, first_obs",
        "next_func": "parse_episode",
        "wrapper": (
            "def infer_language_goal(row, first_obs):\n"
            "    from goal_inference_utils import infer_goal_for_row\n"
            "    return infer_goal_for_row(row, first_obs)\n"
        ),
    },
    {
        "path": "tools/build_progress_probe_dataset.py",
        "func_name": "infer_goal",
        "signature": "row, first_obs",
        "next_func": "parse_jsonl",
        "wrapper": (
            "def infer_goal(row, first_obs):\n"
            "    from goal_inference_utils import infer_goal_for_row\n"
            "    return infer_goal_for_row(row, first_obs)\n"
        ),
    },
    {
        "path": "tools/build_chunk_progress_probe_dataset.py",
        "func_name": "infer_goal",
        "signature": "index_row, first_obs",
        "next_func": "add_vec_features",
        "wrapper": (
            "def infer_goal(index_row, first_obs):\n"
            "    from goal_inference_utils import infer_goal_for_row\n"
            "    return infer_goal_for_row(index_row, first_obs)\n"
        ),
    },
    {
        "path": "tools/build_episode_early_warning_dataset.py",
        "func_name": "infer_goal",
        "signature": "index_row, first_obs",
        "next_func": "add_vec",
        "wrapper": (
            "def infer_goal(index_row, first_obs):\n"
            "    from goal_inference_utils import infer_goal_for_row\n"
            "    return infer_goal_for_row(index_row, first_obs)\n"
        ),
    },
]


def patch_file(item):
    path = Path(item["path"])

    if not path.exists():
        print(f"SKIP missing file: {path}")
        return

    text = path.read_text()

    func_name = item["func_name"]
    next_func = item["next_func"]
    wrapper = item["wrapper"]

    pattern = rf"def {func_name}\(.*?\):\n.*?(?=\n\ndef {next_func}\()"

    new_text, n = re.subn(pattern, wrapper, text, flags=re.S)

    if n != 1:
        raise RuntimeError(f"Expected to replace 1 function in {path}, replaced {n}")

    path.write_text(new_text)
    print(f"Patched {path}: {func_name} -> shared goal_inference_utils")


def main():
    for item in PATCHES:
        patch_file(item)


if __name__ == "__main__":
    main()