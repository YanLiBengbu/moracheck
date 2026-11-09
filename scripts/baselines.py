"""Training-free baselines built on the native phoneme CTC recognizer.

gop  : low posterior of the canonical phone in its forced-aligned segment
rec  : free phone recognition, edit-distance alignment to the canonical phones
lr   : catalogue likelihood ratio, max over learner errors at the mora of
       log P(R_k | x) - log P(C | x) (CTC sequence likelihoods)
Each returns one score per canonical phone (higher = more likely mispronounced).
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from ctc_align import CTC
from jphon import realize_errors, mora_of_phone


def seq_ll(ctc, lp, seqs):
    """CTC log-likelihoods of several phone sequences given one utterance, batched."""
    tg = [ctc.ids(s) for s in seqs]
    L = max(len(t) for t in tg)
    tgt = torch.zeros(len(tg), L, dtype=torch.long, device=lp.device)
    for i, t in enumerate(tg):
        tgt[i, :len(t)] = torch.tensor(t)
    T = lp.shape[0]
    lps = lp[:, None].expand(T, len(tg), lp.shape[1])
    loss = F.ctc_loss(lps, tgt, torch.full((len(tg),), T, dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, reduction="none", zero_infinity=True)
    return (-loss).cpu().numpy()


def score_gop(ctc, lp, C):
    res = ctc.align(lp, C)
    if res is None:
        return [0.0] * len(C)
    spans, post = res
    return [-p for p in post]


def score_rec(ctc, lp, C):
    hyp = ctc.greedy(lp)
    n, m = len(C), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (C[i - 1] != hyp[j - 1]))
    bad = [0.0] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (C[i - 1] != hyp[j - 1]):
            if C[i - 1] != hyp[j - 1]:
                bad[i - 1] = 1.0
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            bad[i - 1] = 1.0; i -= 1
        else:
            bad[max(i - 1, 0)] = 1.0; j -= 1  # insertion: mark the preceding canonical phone
    midx = mora_of_phone(C)
    mb = {}
    for k, b in enumerate(bad):
        mb[midx[k]] = max(mb.get(midx[k], 0), b)
    return [mb[midx[k]] for k in range(n)]


def score_lr(ctc, lp, C):
    errs = realize_errors(C)
    if not errs:
        return [-10.0] * len(C)
    seqs = [C] + [e[2] for e in errs]
    ll = seq_ll(ctc, lp, seqs)
    midx = mora_of_phone(C)
    best = {}
    for k, e in enumerate(errs):
        m = midx[e[3]]
        best[m] = max(best.get(m, -1e9), ll[k + 1] - ll[0])
    return [best.get(midx[k], -10.0) for k in range(len(C))]


_DUR = None


def score_dur(ctc, lp, C):
    """Duration rule: z-score of the aligned, rate-normalized duration against native
    statistics; long for a short vowel, short for a long vowel, geminate or moraic nasal."""
    global _DUR
    if _DUR is None:
        _DUR = json.load(open("results/native_dur_stats.json"))
    res = ctc.align(lp, C)
    if res is None:
        return [0.0] * len(C)
    spans, _ = res
    d = [max(b - a, 1) for a, b in spans]
    nm = sum(1 for p in C if p in "aiueo" or p in ("N", "cl"))
    mu = sum(d) / max(nm, 1)
    z = lambda v, k: (np.log(v / mu) - _DUR[k][0]) / _DUR[k][1]
    sc = [0.0] * len(C)
    for i, p in enumerate(C):
        if p in "aiueo":
            if i + 1 < len(C) and C[i + 1] == p:
                sc[i] = sc[i + 1] = -z(d[i] + d[i + 1], "long_v")
            elif i == 0 or C[i - 1] != p:
                sc[i] = z(d[i], "short_v")
        elif p in ("cl", "N"):
            sc[i] = -z(d[i], p)
    midx = mora_of_phone(C)
    best = {}
    for k, v in enumerate(sc):
        best[midx[k]] = max(best.get(midx[k], -1e9), v)
    return [best[midx[k]] for k in range(len(C))]


METHODS = {"gop": score_gop, "rec": score_rec, "lr": score_lr, "dur": score_dur}


def run(items, methods=("gop", "rec", "lr", "dur"), device="cuda"):
    ctc = CTC(device)
    out = {m: [] for m in methods}
    for it in items:
        x, _ = sf.read(it["wav"], dtype="float32")
        if "start" in it:
            x = x[int(it["start"] * 16000): int(it["end"] * 16000)]
        lp = ctc.logprobs(x)
        for m in methods:
            out[m].append(METHODS[m](ctc, lp, it["C"]))
    return out


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    device = os.environ.get("DEVICE", "cuda")
    if device == "cpu":
        torch.set_num_threads(3)
    methods = tuple(os.environ.get("METHODS", "gop,rec,lr,dur").split(","))
    for p in sys.argv[1:]:
        items = [json.loads(l) for l in open(p, encoding="utf-8")]
        res = run(items, methods=methods, device=device)
        old = p.replace(".jsonl", ".baselines.json")
        if os.path.exists(old):
            prev = json.load(open(old))
            prev.update({m: v for m, v in res.items()})
            res_all = prev
        else:
            res_all = res
        for m, probs in res.items():
            y_u = [int(max(it["labels"]) > 0) for it in items]
            s_u = [max(s) for s in probs]
            y_p = [l for it in items for l in it["labels"]]
            s_p = [x for s in probs for x in s]
            print(f"{p} {m}: utt AUC {roc_auc_score(y_u, s_u):.4f}  phone AUC {roc_auc_score(y_p, s_p):.4f}", flush=True)
        res_all = {m: [[float(v) for v in sc] for sc in probs] for m, probs in res_all.items()}
        json.dump(res_all, open(p.replace(".jsonl", ".baselines.json"), "w"))
