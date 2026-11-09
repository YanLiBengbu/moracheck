"""Phone-boundary accuracy of the CTC alignment against the JSUT labels (Julius times),
and the filter that writes data/native_clean.jsonl."""
import json, sys
import numpy as np

items = [json.loads(l) for l in open("data/native_aligned.jsonl", encoding="utf-8")]
errs_start, errs_end, by_ph = [], [], {}
for it in items:
    if it["corpus"] != "jsut" or "spans" not in it:
        continue
    uid = it["id"][5:]
    ref = []
    for line in open(f"data/raw/jsut-label/labels/basic5000/{uid}.lab"):
        s, e, lab = line.split()
        p = lab.split("-")[1].split("+")[0]
        if p in ("sil", "pau"):
            continue
        ref.append((int(s) / 1e7, int(e) / 1e7, p.lower() if p in "AIUEO" else p))
    if len(ref) != len(it["spans"]):
        continue
    for (rs, re_, p), (fs, fe) in zip(ref, it["spans"]):
        es, ee = fs * 0.02 - rs, fe * 0.02 - re_
        errs_start.append(es); errs_end.append(ee)
        by_ph.setdefault(p, []).append((es, ee, re_ - rs, (fe - fs) * 0.02))
es, ee = np.array(errs_start), np.array(errs_end)
print(f"phones {len(es)}  start err median {np.median(es)*1000:.0f} ms, |err| median {np.median(np.abs(es))*1000:.0f} ms, within 40ms {np.mean(np.abs(es)<0.04):.3f}")
print(f"end err median {np.median(ee)*1000:.0f} ms, |err| median {np.median(np.abs(ee))*1000:.0f} ms, within 40ms {np.mean(np.abs(ee)<0.04):.3f}")
for p in ["a", "o", "N", "cl", "k", "t", "s", "sh", "g", "d", "b"]:
    v = np.array(by_ph.get(p, [(0, 0, 0, 0)]))
    print(f"  {p:3s} n={len(v):6d} start|err| {np.median(np.abs(v[:,0]))*1000:4.0f}ms end|err| {np.median(np.abs(v[:,1]))*1000:4.0f}ms  ref dur {np.median(v[:,2])*1000:4.0f}ms ctc dur {np.median(v[:,3])*1000:4.0f}ms")

# filter: alignment found, recognition close to the canonical phones
pers = np.array([it["per"] for it in items])
print("PER quantiles", np.quantile(pers, [0.5, 0.75, 0.9, 0.95]).round(3))
keep = [it for it in items if "spans" in it and it["per"] <= 0.2]
with open("data/native_clean.jsonl", "w", encoding="utf-8") as f:
    for it in keep:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
from collections import Counter
print("kept", len(keep), "of", len(items), Counter((it["corpus"], it["split"]) for it in keep))
print("hours", sum(it["dur"] for it in keep) / 3600)
