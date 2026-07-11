#!/usr/bin/env python3
from pathlib import Path
import argparse
import datetime

def read(path):
    path = Path(path)
    if not path.exists():
        return f"\n[Missing file: {path}]\n"
    return path.read_text()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla_replay_dir", required=True)
    ap.add_argument("--smolvla_replay_dir", required=True)
    ap.add_argument("--compare_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    openvla = Path(args.openvla_replay_dir)
    smolvla = Path(args.smolvla_replay_dir)
    compare = Path(args.compare_dir)

    md = []
    md.append("# Current Evidence Summary: Same-State Interventional Policy Replay\n")
    md.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")

    md.append("## Paths\n")
    md.append(f"- OpenVLA replay: `{openvla}`")
    md.append(f"- SmolVLA replay: `{smolvla}`")
    md.append(f"- Compare dir: `{compare}`\n")

    md.append("## 1. Cross-policy consequence table\n")
    md.append(read(compare / "paper_policy_consequence_table.md"))

    md.append("\n## 2. OpenVLA pre-action Kinematic++ baseline\n")
    md.append(read(openvla / "preaction_kinematicpp_best_by_method.md"))

    md.append("\n## 3. SmolVLA pre-action Kinematic++ baseline\n")
    md.append(read(smolvla / "preaction_kinematicpp_best_by_method.md"))

    md.append("\n## 4. OpenVLA post-hoc Kinematic oracle\n")
    md.append(read(openvla / "kinematicpp_metrics.md"))

    md.append("\n## 5. SmolVLA post-hoc Kinematic oracle\n")
    md.append(read(smolvla / "kinematicpp_metrics.md"))

    md.append("""
## Current interpretation

The current evidence supports the following thesis:

Same-state interventional replay exposes policy-specific object-level consequence profiles.
OpenVLA-OFT preserves a stable default-source manipulation template across language perturbations, while SmolVLA is more language-sensitive and often collapses under corrupted language. Strong pre-action and post-hoc kinematic baselines show that these consequences are physically grounded and partly encoded in action geometry.

The paper should not claim that learned evaluators beat kinematics at this stage. The stronger and safer claim is that interventional replay provides a controlled evaluation protocol for exposing what object a learned robot policy action actually affects.
""")

    out_path = out_dir / "current_evidence_summary.md"
    out_path.write_text("\n".join(md))
    print("[saved]", out_path)

if __name__ == "__main__":
    main()