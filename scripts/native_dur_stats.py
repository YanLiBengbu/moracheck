"""Native duration statistics for the duration-rule baseline: mean and standard deviation of
log(d / mean mora duration) for short vowels, long vowels (both halves), geminates and moraic
nasals, from the CTC alignments of the native training utterances."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from jphon import VOWELS

def mora_count(ph):
    return sum(1 for p in ph if p in VOWELS or p in ("N", "cl"))

cats = {"short_v": [], "long_v": [], "cl": [], "N": []}
for l in open("data/native_clean.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["split"] != "train":
        continue
    ph, sp = it["phones"], it["spans"]
    d = [max(b - a, 1) for a, b in sp]
    mu = sum(d) / max(mora_count(ph), 1)
    for i, p in enumerate(ph):
        if p in VOWELS:
            if i + 1 < len(ph) and ph[i + 1] == p:
                cats["long_v"].append(np.log((d[i] + d[i + 1]) / mu))
            elif i == 0 or ph[i - 1] != p:
                cats["short_v"].append(np.log(d[i] / mu))
        elif p in ("cl", "N"):
            cats[p].append(np.log(d[i] / mu))
stats = {k: [float(np.mean(v)), float(np.std(v)), len(v)] for k, v in cats.items()}
json.dump(stats, open("results/native_dur_stats.json", "w"), indent=1)
print(stats)
