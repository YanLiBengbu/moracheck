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
  4. native control: share of native target words at or above each method's development threshold
  5. D set (optional): Spearman correlation between the mean rating and the word score, test speakers
Label rule (fixed before the data arrived): the label is the majority of the teachers; items
with a 2 to 2 tie among four teachers are left out of the main AUC, and a sensitivity run
counts ties as mispronounced (build the items file with label_ties=1).
Group and first-language AUCs need at least 30 positive and 30 negative test items.
With --summary, the headline numbers for Fig. 1 and the video go to a small JSON
(results/umejrf_summary.json in the real run).
Usage: umejrf_analyze.py SCORED.json ITEMS.jsonl OUT.json [--ref KEY] [--dset SCORED_D.json] [--summary PATH]
"""
import argparse, json, random
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
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
    ap.add_argument("--dset", default=None)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    r = json.load(open(a.scored))
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    y = np.array(r["label"]); spk = np.array([str(s) for s in r["spk"]]); l1 = np.array([str(x) for x in r["l1"]])
    grp = np.array([str(g) for g in r["group"]])
    scores = {k: np.array(v) for k, v in r["scores"].items()}
    ref = a.ref or ("moracheck" if "moracheck" in scores else next(k for k in scores if "last.pt" in k))
    nat = np.array(r.get("native") or [0] * len(y)).astype(bool)
    dev = split_speakers(spk[~nat], l1[~nat])
    is_dev = np.array([s in dev for s in spk]) & ~nat
    te = ~is_dev & ~nat
    out = dict(ref=ref, n_items=int((~nat).sum()), n_speakers=int(len(set(spk[~nat]))), n_dev_speakers=len(dev),
               n_test_items=int(te.sum()), test_positive_rate=float(y[te].mean()), n_native_items=int(nat.sum()))
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
        enough = lambda m: y[m].sum() >= 30 and (1 - y[m]).sum() >= 30
        row["per_group"] = {g: float(roc_auc_score(y[te & (grp == g)], s[te & (grp == g)]))
                            for g in sorted(set(grp[te])) if enough(te & (grp == g))}
        row["per_l1"] = {l: float(roc_auc_score(y[te & (l1 == l)], s[te & (l1 == l)]))
                         for l in sorted(set(l1[te])) if enough(te & (l1 == l))}
        if nat.any():
            row["native_false_alarm"] = float((s[nat] >= thr).mean())
        out["methods"][k] = row
    # like-for-like agreement: hold out one teacher at a time; the held-out teacher and each
    # method are both compared with the majority of the remaining teachers (no ties with three)
    rat = [it.get("ratings") for it in items]
    tk, mk = defaultdict(lambda: ([], [])), {k: defaultdict(lambda: ([], [])) for k in scores}
    thr = {k: out["methods"][k]["threshold"] for k in scores}
    for i in np.where(te)[0]:
        rs = rat[i]
        if not rs or len(rs) < 4:
            continue
        for j, v in enumerate(rs):
            others = [x for t, x in enumerate(rs) if t != j]
            if sum(others) * 2 == len(others):
                continue
            maj = int(sum(others) * 2 > len(others))
            tk[j][0].append(v); tk[j][1].append(maj)
            for k, sc in scores.items():
                mk[k][j][0].append(int(sc[i] >= thr[k])); mk[k][j][1].append(maj)
    kap = lambda a_, b_: float(cohen_kappa_score(a_, b_)) if len(set(a_)) > 1 and len(set(b_)) > 1 else None
    out["teacher_kappa"] = {str(j): kap(*v) for j, v in tk.items()}
    vals = [v for v in out["teacher_kappa"].values() if v is not None]
    out["teacher_kappa_mean"] = float(np.mean(vals)) if vals else None
    for k in scores:
        mv = [kap(*v) for v in mk[k].values()]
        mv = [v for v in mv if v is not None]
        out["methods"][k]["kappa_heldout_mean"] = float(np.mean(mv)) if mv else None
    if a.dset:
        rd = json.load(open(a.dset))
        dspk = np.array([str(x) for x in rd["spk"]])
        dnat = np.array(rd.get("native") or [0] * len(dspk)).astype(bool)
        dte = np.array([x not in dev for x in dspk]) & ~dnat
        rm = np.array([np.nan if v is None else v for v in rd["rating_mean"]], dtype=float)
        ok = dte & ~np.isnan(rm)
        out["dset"] = {"n_test_words": int(ok.sum())}
        for k, v in rd["scores"].items():
            rho, pv = spearmanr(rm[ok], np.array(v)[ok])
            out["dset"][k] = dict(spearman=float(rho), p=float(pv))
    json.dump(out, open(a.out, "w"), indent=1)
    if a.summary:
        m = out["methods"]
        json.dump(dict(moracheck=m[ref]["auc"], moracheck_ci=m[ref]["ci"], lr=m["lr"]["auc"], gop=m["gop"]["auc"],
                       lr_diff_ci=m["lr"]["diff_ci"], kappa_model=m[ref]["kappa_heldout_mean"],
                       kappa_teacher=out["teacher_kappa_mean"], native_false_alarm=m[ref].get("native_false_alarm"),
                       n_test_items=out["n_test_items"], n_speakers=out["n_speakers"]),
                  open(a.summary, "w"), indent=1)
    for k, row in out["methods"].items():
        print(f"{k:45s} AUC {row['auc']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] kappa {row['kappa']:.3f}")
    print("teacher kappa (held-out mean)", out.get("teacher_kappa_mean"))
    for k, row in out["methods"].items():
        print(f"{k:45s} kappa vs held-out majority {row.get('kappa_heldout_mean')}")


if __name__ == "__main__":
    main()
