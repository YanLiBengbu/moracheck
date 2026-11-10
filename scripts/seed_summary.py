"""Mean and range over three seeds for the full model, the key ablations and the
fixed-factor duration control, from the final checkpoint of each run's log."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"MoraCheck": ["moracheck_s0", "moracheck_s1", "moracheck_s2"],
          "without segment features": ["abl_no_seg_s0", "abl_no_seg_s1", "abl_no_seg_s2"],
          "without duration edits": ["abl_no_dur_edits_s0", "abl_no_dur_edits_s1", "abl_no_dur_edits_s2"],
          "without text flips": ["abl_no_flips_s0", "abl_no_flips_s1", "abl_no_flips_s2"],
          "fixed duration factors": ["ctl_fixed_dur_s0", "ctl_fixed_dur_s1", "ctl_fixed_dur_s2"]}
SETS = ["flip", "lex", "dur", "tts"]


def last(run):
    p = ROOT / f"results/runs/{run}"
    if not (p / "done.json").exists():
        return None
    line = [l for l in open(p / "log.txt") if l.startswith("EVAL")][-1]
    return {k: float(v) for k, v in re.findall(r"(\w+):utt=([0-9.]+)", line)}


out = {}
for name, runs in GROUPS.items():
    vals = [last(r) for r in runs]
    vals = [v for v in vals if v]
    if not vals:
        continue
    out[name] = {s: dict(mean=sum(v[s] for v in vals) / len(vals), lo=min(v[s] for v in vals), hi=max(v[s] for v in vals),
                         n=len(vals)) for s in SETS}
    print(f"{name:26s} n={len(vals)} " + " ".join(f"{s}={out[name][s]['mean']:.3f}[{out[name][s]['lo']:.3f},{out[name][s]['hi']:.3f}]" for s in SETS))
json.dump(out, open(ROOT / "results/seed_summary.json", "w"), indent=1)
