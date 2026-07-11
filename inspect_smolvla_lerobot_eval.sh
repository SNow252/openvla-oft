#!/usr/bin/env bash
set -euo pipefail

echo "==== env ===="
which python || true
which lerobot-eval || true
python - <<'PY'
import sys, os
print("python:", sys.executable)
try:
    import lerobot
    print("lerobot package:", lerobot.__file__)
except Exception as e:
    print("import lerobot failed:", repr(e))
PY

echo
echo "==== locate lerobot source ===="
LEROBOT_SRC="$(python - <<'PY'
import os
try:
    import lerobot
    p = os.path.dirname(lerobot.__file__)
    print(p)
except Exception:
    print("")
PY
)"

echo "LEROBOT_SRC=$LEROBOT_SRC"

if [[ -z "$LEROBOT_SRC" ]]; then
  echo "[ERROR] Cannot import lerobot."
  exit 1
fi

echo
echo "==== grep eval entrypoints ===="
grep -R "def eval" -n "$LEROBOT_SRC" | head -80 || true
grep -R "def eval_policy" -n "$LEROBOT_SRC" | head -80 || true
grep -R "class.*Eval" -n "$LEROBOT_SRC" | head -80 || true

echo
echo "==== grep action selection ===="
grep -R "select_action" -n "$LEROBOT_SRC" | head -120 || true
grep -R "predict_action" -n "$LEROBOT_SRC" | head -120 || true
grep -R "policy(.*obs" -n "$LEROBOT_SRC" | head -80 || true
grep -R "policy.forward" -n "$LEROBOT_SRC" | head -80 || true

echo
echo "==== grep env step ===="
grep -R "env.step" -n "$LEROBOT_SRC" | head -120 || true
grep -R ".step(action" -n "$LEROBOT_SRC" | head -120 || true
grep -R "actions" -n "$LEROBOT_SRC/scripts" "$LEROBOT_SRC/common" 2>/dev/null | head -120 || true

echo
echo "==== grep save episode / rollout ===="
grep -R "save.*episode" -n "$LEROBOT_SRC" | head -120 || true
grep -R "rollout" -n "$LEROBOT_SRC" | head -120 || true
grep -R "video" -n "$LEROBOT_SRC/scripts" "$LEROBOT_SRC/common" 2>/dev/null | head -120 || true

echo
echo "==== candidate files content snippets ===="
for f in \
  "$LEROBOT_SRC/scripts/eval.py" \
  "$LEROBOT_SRC/scripts/evaluate.py" \
  "$LEROBOT_SRC/common/evaluation/eval.py" \
  "$LEROBOT_SRC/common/evaluation/utils.py" \
  "$LEROBOT_SRC/common/envs/factory.py" \
  "$LEROBOT_SRC/common/policies/factory.py"
do
  if [[ -f "$f" ]]; then
    echo
    echo "################################################################################"
    echo "# FILE: $f"
    echo "################################################################################"
    sed -n '1,260p' "$f"
  fi
done