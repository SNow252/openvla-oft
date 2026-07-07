from pathlib import Path
import re


path = Path("experiments/robot/libero/run_libero_eval_dump.py")
text = path.read_text()

new_func = r'''def _vla_dump_sanitize(x):
    """Convert numpy / torch / Python objects into JSON-safe values."""
    try:
        if x is None or isinstance(x, (str, int, float, bool)):
            return x

        if isinstance(x, dict):
            return {str(k): _vla_dump_sanitize(v) for k, v in x.items()}

        if isinstance(x, (list, tuple)):
            return [_vla_dump_sanitize(v) for v in x]

        # torch.Tensor-like
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            try:
                x = x.detach().cpu().numpy()
            except Exception:
                x = x.detach().cpu().tolist()
                return _vla_dump_sanitize(x)

        # numpy ndarray
        if isinstance(x, _vla_np.ndarray):
            return _vla_dump_sanitize(x.tolist())

        # numpy scalar
        if isinstance(x, _vla_np.generic):
            return x.item()

        # generic array-like
        if hasattr(x, "tolist"):
            return _vla_dump_sanitize(x.tolist())

        # scalar-like
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass

        return str(x)

    except Exception as e:
        return {"dump_error": str(e), "type": str(type(x))}
'''


pattern = r"def _vla_dump_sanitize\(x\):.*?(?=\ndef _vla_obs_summary\(obs\):)"
new_text, n = re.subn(pattern, new_func + "\n", text, flags=re.S)

if n != 1:
    raise RuntimeError(f"Expected to replace 1 sanitize function, replaced {n}.")

path.write_text(new_text)
print(f"Fixed _vla_dump_sanitize in {path}")