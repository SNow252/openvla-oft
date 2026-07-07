from pathlib import Path


path = Path("experiments/robot/libero/run_libero_eval_dump.py")
text = path.read_text()

# 1. Save obs summary at policy query time.
if "_vla_query_obs_summary" not in text:
    old = "                # Query model to get action\n"
    new = (
        "                # Query model to get action\n"
        "                _vla_query_obs_summary = _vla_obs_summary(obs) if _vla_dump_enabled else None\n"
    )

    if old not in text:
        raise RuntimeError("Cannot find query-action comment block.")

    text = text.replace(old, new, 1)


# 2. Add query_obs_summary to chunk records.
if '"query_obs_summary":' not in text:
    old = '                            "raw_actions_chunk": _vla_dump_sanitize(_vla_np.asarray(actions)),\n'
    new = (
        '                            "raw_actions_chunk": _vla_dump_sanitize(_vla_np.asarray(actions)),\n'
        '                            "query_obs_summary": _vla_query_obs_summary,\n'
    )

    if old not in text:
        raise RuntimeError("Cannot find raw_actions_chunk dump block.")

    text = text.replace(old, new, 1)


# 3. Save obs before each step.
if "_vla_obs_before_summary" not in text:
    old = (
        "            action = action_queue.popleft()\n"
        "            _vla_raw_action_before_process = _vla_np.asarray(action).copy()\n"
    )

    new = (
        "            action = action_queue.popleft()\n"
        "            _vla_raw_action_before_process = _vla_np.asarray(action).copy()\n"
        "            _vla_obs_before_summary = _vla_obs_summary(obs) if _vla_dump_enabled else None\n"
    )

    if old not in text:
        raise RuntimeError("Cannot find action_queue.popleft block.")

    text = text.replace(old, new, 1)


# 4. Add obs_before_summary and obs_after_summary to step records.
if '"obs_after_summary":' not in text:
    old = '                        "obs_summary": _vla_obs_summary(obs),\n'
    new = (
        '                        "obs_before_summary": _vla_obs_before_summary,\n'
        '                        "obs_after_summary": _vla_obs_summary(obs),\n'
        '                        "obs_summary": _vla_obs_summary(obs),\n'
    )

    if old not in text:
        raise RuntimeError("Cannot find obs_summary dump block.")

    text = text.replace(old, new, 1)


path.write_text(text)
print(f"Patched dump script: {path}")
print("Now chunk records contain query_obs_summary, and step records contain obs_before_summary / obs_after_summary.")