"""Analysis of word scores on rated learner speech (UME-JRF).

Input: a scored result of eval_target_words.py whose items also carry
  ratings (list of per-teacher binary labels, 1 = mispronounced), l1, spk, group.
Steps
  1. split speakers into development (30%) and test (70%), stratified by first language
  2. on development speakers: decision threshold of each method (Youden index) and the
     error rate of each group, which gives the learner-informed group distribution
  3. on test speakers: AUC with speaker-cluster bootstrap intervals, paired differences to
     MoraCheck, Cohen's kappa of each method and of each teacher with the majority of the
     other teachers, AUC per group and per first language
Usage: umejrf_analyze.py SCORED.json ITEMS.jsonl OUT.json [--ref KEY]
"""
import argparse, json, random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score

from stats import cluster_boot


def split_speakers(spk, l1, frac=0.3, seed=2026):
    by = defaultdict(list)
    for s, l in sorted(set(zip(spk, l1))):
        by[l].append(s)
    rng = random.Random(seed)
    dev = set()
    for l, ss in sorted(by.items()):
        rng.shuffle(ss)
        dev.update(ss[:max(1, round(frac * len(ss)))])
    return dev


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored"); ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    r = json.load(open(a.scored))
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    y = np.array(r["label"]); spk = np.array([str(s) for s in r["spk"]]); l1 = np.array([str(x) for x in r["l1"]])
    grp = np.array([str(g) for g in r["group"]])
    scores = {k: np.array(v) for k, v in r["scores"].items()}
    ref = a.ref or next(k for k in scores if "best.pt" in k)
    dev = split_speakers(spk, l1)
    is_dev = np.array([s in dev for s in spk])
    te = ~is_dev
    out = dict(n_items=int(len(y)), n_speakers=int(len(set(spk))), n_dev_speakers=len(dev),
               n_test_items=int(te.sum()), test_positive_rate=float(y[te].mean()))
    # learner-informed group distribution from development speakers
    out["dev_group_error_rate"] = {g: float(y[is_dev & (grp == g)].mean()) for g in sorted(set(grp)) if (is_dev & (grp == g)).any()}
    # thresholds on development speakers, kappa on test speakers
    out["methods"] = {}
    d = cluster_boot(y[te], {k: v[te] for k, v in scores.items()}, spk[te])
    for k, s in scores.items():
        thr = youden(y[is_dev], s[is_dev])
        auc = roc_auc_score(y[te], s[te])
        row = dict(auc=float(auc), ci=[float(x) for x in np.percentile(d[k], [2.5, 97.5])], threshold=thr,
                   kappa=float(cohen_kappa_score(y[te], (s[te] >= thr).astype(int))))
        if k != ref:
            diff = d[ref] - d[k]
            row["diff_to_ref"] = float(roc_auc_score(y[te], scores[ref][te]) - auc)
            row["diff_ci"] = [float(x) for x in np.percentile(diff, [2.5, 97.5])]
        row["per_group"] = {g: float(roc_auc_score(y[te & (grp == g)], s[te & (grp == g)]))
                            for g in sorted(set(grp)) if 0 < y[te & (grp == g)].sum() < (te & (grp == g)).sum()}
        row["per_l1"] = {l: float(roc_auc_score(y[te & (l1 == l)], s[te & (l1 == l)]))
                         for l in sorted(set(l1)) if 0 < y[te & (l1 == l)].sum() < (te & (l1 == l)).sum()}
        out["methods"][k] = row
    # each teacher against the majority of the other teachers, on test items rated by at least three
    rat = [it.get("ratings") for it in items]
    kap = defaultdict(lambda: ([], []))
    for i in np.where(te)[0]:
        rs = rat[i]
        if not rs or len(rs) < 3:
            continue
        for j, v in enumerate(rs):
            others = [x for t, x in enumerate(rs) if t != j]
            maj = int(sum(others) * 2 > len(others)) if sum(others) * 2 != len(others) else None
            if maj is not None:
                kap[j][0].append(v); kap[j][1].append(maj)
    out["teacher_kappa"] = {str(j): float(cohen_kappa_score(a_, b_)) for j, (a_, b_) in kap.items() if len(set(a_)) > 1}
    if out["teacher_kappa"]:
        out["teacher_kappa_mean"] = float(np.mean(list(out["teacher_kappa"].values())))
    json.dump(out, open(a.out, "w"), indent=1)
    for k, row in out["methods"].items():
        print(f"{k:45s} AUC {row['auc']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] kappa {row['kappa']:.3f}")
    print("teacher kappa", out.get("teacher_kappa_mean"))


if __name__ == "__main__":
    main()
