"""Ablation table of MoraCheck on the native validation sets, from the training logs.

Each row is the checkpoint after the last step of one run (seed 0). The full
model row is the seed-0 run, so every row has the same seed and training budget.
Prints LaTeX rows and writes results/ablation_summary.json.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("moracheck_s0", "MoraCheck"),
        ("abl_no_lr_s0", "Remove the likelihood ratios."),
        ("abl_no_dur_s0", "Remove the durations."),
        ("abl_no_seg_s0", "Remove the segment features."),
        ("abl_no_xattn_s0", "Remove the cross-attention."),
        ("abl_flip_only_s0", "Train on text flips only."),
        ("abl_no_tts_s0", "Remove the TTS swaps."),
        ("abl_no_dur_edits_s0", "Remove the duration edits."),
        ("abl_no_flips_s0", "Remove the text flips.")]
SETS = ["flip", "lex", "dur", "tts"]


def best(run):
    p = ROOT / f"results/runs/{run}/log.txt"
    if not (ROOT / f"results/runs/{run}/done.json").exists():
        return None
    lines = [l for l in open(p) if l.startswith("EVAL")]
    b = lines[-1]  # the checkpoint after the last step; no selection on the reported sets
    return {k: float(v) for k, v in re.findall(r"(\w+):utt=([0-9.]+)", b)}


out = {}
full = best(ROWS[0][0])
for run, name in ROWS:
    r = best(run)
    if r is None:
        print(f"% {name} pending")
        continue
    out[run] = r
    cells = []
    for s in SETS:
        v = r[s]
        if run == ROWS[0][0]:
            cells.append(f"\\textbf{{{v:.3f}}}")
        else:
            dlt = round(v - full[s], 3)
            sign = "$-$" if dlt < 0 else ("+" if dlt > 0 else "")
            cells.append(f"{v:.3f} ({sign}{abs(dlt):.3f})")
    print(f"{name} & " + " & ".join(cells) + " \\\\")
json.dump(out, open(ROOT / "results/ablation_summary.json", "w"), indent=1)
