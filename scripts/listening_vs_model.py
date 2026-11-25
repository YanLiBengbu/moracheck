"""Compare detector scores on the 60 listening clips with what the listeners heard.

Reads the scored clips (eval_target_words.py on data/listening_items.jsonl) and adds to the
listening results: AUC of each method against the acoustic edit label, against the native
listener's percept, on edits heard by both listeners or by neither, and within the edits
(heard against unheard edits).
Intervals are from 2000 bootstrap resamples of clips; differences are paired against MoraCheck.
Usage: listening_vs_model.py SCORED.json ITEMS.jsonl LISTENING.json
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

METHODS = ["moracheck", "lr", "gop", "rec", "dur"]


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def boot(y, S, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: [] for k in S}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for k, s in S.items():
            d[k].append(roc_auc_score(y[i], s[i]))
    return {k: np.array(v) for k, v in d.items()}


def main():
    r = json.load(open(sys.argv[1]))
    items = [json.loads(l) for l in open(sys.argv[2])]
    out = json.load(open(sys.argv[3]))
    S = {k: np.array(r["scores"][k]) for k in METHODS}
    y = np.array([it["label"] for it in items])
    hn = np.array([it["heard_native"] for it in items])
    hl = np.array([it["heard_l2"] for it in items])
    val = np.array([it["split"] == "valid" for it in items])
    both, neither, ctrl = (y == 1) & (hn == 1) & (hl == 1), (y == 1) & (hn == 0) & (hl == 0), y == 0
    res = dict(n_clips=len(y), n_validation_speaker_clips=int(val.sum()), n_both=int(both.sum()), n_neither=int(neither.sum()))
    d = boot(hn, S)
    e = y == 1
    d2 = boot(hn[e], {k: s[e] for k, s in S.items()})
    for k, s in S.items():
        a = auc(hn, s)
        row = dict(edit_vs_control=auc(y, s), native_percept=a, native_percept_ci=[float(x) for x in np.percentile(d[k], [2.5, 97.5])],
                   native_percept_validation_speakers=auc(hn[val], s[val]),
                   heard_by_both_vs_control=auc(y[both | ctrl], s[both | ctrl]),
                   heard_by_neither_vs_control=auc(y[neither | ctrl], s[neither | ctrl]),
                   within_edits_native=auc(hn[e], s[e]), within_edits_native_ci=[float(x) for x in np.percentile(d2[k], [2.5, 97.5])],
                   within_edits_l2=auc(hl[e], s[e]))
        if k != "moracheck":
            diff = d["moracheck"] - d[k]
            row["moracheck_minus_this_ci"] = [float(x) for x in np.percentile(diff, [2.5, 97.5])]
            row["within_edits_moracheck_minus_this_ci"] = [float(x) for x in np.percentile(d2["moracheck"] - d2[k], [2.5, 97.5])]
        res[k] = row
    m = S["moracheck"]
    res["moracheck_median"] = dict(control=float(np.median(m[ctrl])), heard_by_both=float(np.median(m[both])),
                                   heard_by_neither=float(np.median(m[neither])))
    out["model_vs_listeners"] = res
    json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
