"""Word-level evaluation on rated learner speech (UME-JRF style).

Input jsonl, one rated utterance per line:
  wav, text (sentence or word read aloud), target (the rated word, kana or surface form),
  label (1 = judged mispronounced by the expert majority), spk, l1, group (error category)
Each method scores the target word by the maximum phone score over its phones.
Usage: eval_target_words.py ITEMS.jsonl OUT.json --ckpt results/runs/X/best.pt [--baselines]
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from jphon import g2p
from simdata import FixedSet
from model import Detector


def locate(C, W):
    """Phone indices of word W inside sentence C (first exact match, else best overlap)."""
    n, m = len(C), len(W)
    for s in range(n - m + 1):
        if C[s:s + m] == W:
            return list(range(s, s + m))
    best, arg = -1, None
    for s in range(n - m + 1):
        k = sum(a == b for a, b in zip(C[s:s + m], W))
        if k > best:
            best, arg = k, s
    return list(range(arg, arg + m)) if arg is not None else list(range(n))


def prepare(items):
    for it in items:
        C = [p for p in g2p(it["text"]) if p != "pau"]
        W = [p for p in g2p(it["target"]) if p != "pau"]
        it["C"] = C
        it["tidx"] = locate(C, W)
        it.setdefault("labels", [0] * len(C))
    return items


def word_scores(items, phone_scores):
    return [float(max(s[i] for i in it["tidx"])) for it, s in zip(items, phone_scores)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--encoder", default="models/japanese-hubert-base")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    items = prepare([json.loads(l) for l in open(a.items, encoding="utf-8")])
    res = {"n": len(items), "label": [it["label"] for it in items], "spk": [it.get("spk") for it in items],
           "group": [it.get("group") for it in items], "l1": [it.get("l1") for it in items], "scores": {}}
    from train_detector import predict
    import train_v2
    from model_v2 import DetectorV2
    for ck in a.ckpt:
        args = json.load(open(os.path.join(os.path.dirname(ck), "args.json")))
        if "no_lr" in args:
            model = DetectorV2(use_lr=not args["no_lr"], use_dur=not args["no_dur"], use_seg=not args["no_seg"],
                               use_xattn=not args["no_xattn"]).cuda()
            model.load_state_dict(torch.load(ck, map_location="cuda"), strict=False)
            probs = train_v2.predict(model, FixedSet(items))
        else:
            model = Detector(args.get("encoder", a.encoder), finetune_top=args.get("finetune_top", 0)).cuda()
            model.load_state_dict(torch.load(ck, map_location="cuda"), strict=False)
            probs = predict(model, FixedSet(items))
        res["scores"][ck] = word_scores(items, probs)
        del model; torch.cuda.empty_cache()
    if a.baselines:
        from baselines import run
        for m, ps in run(items).items():
            res["scores"][m] = word_scores(items, ps)
    from sklearn.metrics import roc_auc_score
    y = np.array(res["label"])
    for k, s in res["scores"].items():
        if 0 < y.sum() < len(y):
            print(f"{k}: AUC {roc_auc_score(y, s):.4f}")
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
