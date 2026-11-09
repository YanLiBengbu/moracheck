"""Statistics for word-level results: AUC with speaker-cluster bootstrap intervals,
paired AUC differences, kappa against the expert majority, and per-group AUCs.
Usage: stats.py RESULT.json [REF_KEY]"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score


def cluster_boot(y, scores, spk, B=2000, seed=0):
    """Bootstrap over speakers. scores: dict key -> array. Returns dict key -> AUC draws."""
    rng = np.random.default_rng(seed)
    spks = np.array(sorted(set(spk)))
    idx = {s: np.where(spk == s)[0] for s in spks}
    draws = {k: [] for k in scores}
    for _ in range(B):
        pick = rng.choice(spks, size=len(spks), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if y[ii].min() == y[ii].max():
            continue
        for k, s in scores.items():
            draws[k].append(roc_auc_score(y[ii], s[ii]))
    return {k: np.array(v) for k, v in draws.items()}


def kappa_at(y, s, thr):
    return cohen_kappa_score(y, (s >= thr).astype(int))


def main():
    r = json.load(open(sys.argv[1]))
    y = np.array(r["label"]); spk = np.array([str(s) for s in r["spk"]])
    scores = {k: np.array(v) for k, v in r["scores"].items()}
    ref = sys.argv[2] if len(sys.argv) > 2 else list(scores)[0]
    d = cluster_boot(y, scores, spk)
    print(f"n={len(y)} positives={int(y.sum())} speakers={len(set(spk))}")
    for k, s in scores.items():
        auc = roc_auc_score(y, s)
        lo, hi = np.percentile(d[k], [2.5, 97.5])
        line = f"{k:40s} AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]"
        if k != ref:
            diff = d[ref] - d[k]
            line += f"  ref-minus {roc_auc_score(y, scores[ref]) - auc:+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]"
        print(line)
    groups = np.array([str(g) for g in r["group"]])
    for g in sorted(set(groups)):
        m = groups == g
        if 0 < y[m].sum() < m.sum():
            print(f"  group {g:14s} n={m.sum():5d} pos={int(y[m].sum()):4d} " +
                  " ".join(f"{k.split('/')[-2] if '/' in k else k}={roc_auc_score(y[m], s[m]):.3f}" for k, s in scores.items()))


if __name__ == "__main__":
    main()
