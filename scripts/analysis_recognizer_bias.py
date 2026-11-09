"""What the native phoneme recognizer outputs on the duration-edit validation set:
the canonical sequence, the realized (erroneous) sequence, or something else."""
import json, os, sys
import torch
import soundfile as sf
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from ctc_align import CTC

dev = os.environ.get("DEVICE", "cuda")
if dev == "cpu":
    torch.set_num_threads(4)
ctc = CTC(dev)
items = [json.loads(l) for l in open("data/sim/val_dur.jsonl", encoding="utf-8")]
res = defaultdict(Counter)
for it in items:
    x, _ = sf.read(it["wav"], dtype="float32")
    hyp = ctc.greedy(ctc.logprobs(x))
    key = "canonical" if hyp == it["C"] else ("realized" if hyp == it["R"] else "other")
    res[it["type"]][key] += 1
out = {t: dict(c) for t, c in res.items()}
json.dump(out, open("results/recognizer_bias_val_dur.json", "w"), indent=1)
for t, c in sorted(out.items()):
    n = sum(c.values())
    print(f"{t:12s} n={n:4d} canonical={c.get('canonical',0):4d} realized={c.get('realized',0):4d} other={c.get('other',0):4d}")
