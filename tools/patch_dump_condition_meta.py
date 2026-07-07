from pathlib import Path


path = Path("experiments/robot/libero/run_libero_eval_dump.py")
text = path.read_text()

if '"dump_condition":' in text:
    print("dump_condition already exists. Nothing to patch.")
else:
    old = '                "custom_language": os.environ.get("OPENVLA_CUSTOM_LANGUAGE", ""),\n'
    new = (
        '                "dump_condition": os.environ.get("OPENVLA_DUMP_CONDITION", ""),\n'
        '                "custom_language": os.environ.get("OPENVLA_CUSTOM_LANGUAGE", ""),\n'
    )

    if old not in text:
        raise RuntimeError("Cannot find custom_language line in meta dump block.")

    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"Patched dump_condition into {path}")