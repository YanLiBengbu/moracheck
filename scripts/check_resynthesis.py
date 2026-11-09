"""Processing-trace check: does MoraCheck react to overlap-add resynthesis alone?

800 native validation utterances are resynthesized by PSOLA with a scale of 1.0 on a random
phoneme span (the same processing as the correct controls, without a duration change). The
utterance AUC of MoraCheck for telling resynthesized from original recordings should be near
0.5 if the model does not use processing traces. Usage: check_resynthesis.py CKPT [CKPT ...]
"""
import json, os, random, sys
import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
import make_dur_edits as M
from simdata import FixedSet
import train_v2
from model_v2 import DetectorV2

out_dir = "data/sim/resynth_check"
os.makedirs(out_dir, exist_ok=True)
items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
items = [it for it in items if it["split"] == "valid"]
random.Random(31).shuffle(items)
items = items[:800]
orig, resyn = [], []
rng = random.Random(5)
for it in items:
    x, _ = sf.read(it["wav"], dtype="float32")
    i = rng.randrange(len(it["phones"]))
    t0, t1 = it["spans"][i][0] * 0.02, it["spans"][i][1] * 0.02
    if t1 - t0 < 0.02:
        t0, t1 = 0.0, 0.05
    y = M.psola(x, t0, t1, 1.0)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    p = f"{out_dir}/{it['id']}.wav"
    sf.write(p, y, 16000, subtype="PCM_16")
    orig.append(dict(wav=it["wav"], C=it["phones"], labels=[0] * len(it["phones"])))
    resyn.append(dict(wav=p, C=it["phones"], labels=[0] * len(it["phones"])))
dev = os.environ.get("DEVICE", "cpu")
if dev == "cpu":
    torch.set_num_threads(4)
res = {}
for ck in sys.argv[1:]:
    m = DetectorV2().to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    po = [max(p) for p in train_v2.predict(m, FixedSet(orig), device=dev)]
    pr = [max(p) for p in train_v2.predict(m, FixedSet(resyn), device=dev)]
    auc = roc_auc_score([0] * len(po) + [1] * len(pr), po + pr)
    res[ck] = dict(auc_resynth_vs_orig=float(auc), mean_max_orig=float(np.mean(po)), mean_max_resynth=float(np.mean(pr)))
    print(ck, res[ck], flush=True)
json.dump(res, open("results/resynthesis_check.json", "w"), indent=1)
