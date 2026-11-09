"""Frozen validation set of text flips on held-out native speakers (CV test speakers)."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from jphon import flip_candidates, phone_labels, GROUP

rng = random.Random(123)
items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
items = [it for it in items if it["split"] == "valid"]
rng.shuffle(items)
groups = sorted(set(GROUP.values()))
out = []
for it in items[:2000]:
    R = it["phones"]
    rec = dict(wav=it["wav"], C=R, labels=[0] * len(R), R=R, type="none", src=it["id"], spk=it["spk"])
    if rng.random() < 0.5:
        by_g = {}
        for c in flip_candidates(R):
            by_g.setdefault(GROUP[c[0]], []).append(c)
        if by_g:
            g = rng.choice([g for g in groups if g in by_g])
            et, C, _, pos = rng.choice(by_g[g])
            rec.update(C=C, labels=phone_labels(C, pos), type=et)
    out.append(rec)
os.makedirs("data/sim", exist_ok=True)
with open("data/sim/val_flip.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(GROUP.get(r["type"], "none") for r in out))
