"""Score the blind listening check of the duration edits (make_listening_test.py).

Each listener heard 60 clips: 40 duration edits (8 per length error type) and 20 controls that
only went through the same resynthesis, and picked which of two katakana forms was heard.
An edit counts as perceived when the listener picked the error form. A control counts as
correct when the listener picked the canonical form.
Usage: score_listening.py KEY.json OUT.json NAME=RESPONSES.csv [NAME=RESPONSES.csv ...]
"""
import csv, json, sys
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

TYPES = ["long_short", "short_long", "gem_drop", "gem_add", "nas_drop"]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def main():
    key = {x["no"]: x for x in json.load(open(sys.argv[1]))}
    resp = {}
    for arg in sys.argv[3:]:
        name, path = arg.split("=", 1)
        resp[name] = {int(r["no"]): r for r in csv.DictReader(open(path))}
        assert set(resp[name]) == set(key), name
    heard = {n: {i: int(r[i]["answer"] == key[i]["error_option"]) for i in key} for n, r in resp.items()}
    out = dict(n_clips=len(key), listeners={})
    for n, h in heard.items():
        ed = [i for i in key if key[i]["type"] != "none"]
        ct = [i for i in key if key[i]["type"] == "none"]
        k = sum(h[i] for i in ed)
        conf = {i: int(resp[n][i]["confidence"]) for i in key}
        out["listeners"][n] = dict(
            edits_perceived=k, n_edits=len(ed), rate=round(k / len(ed), 3), ci=wilson(k, len(ed)),
            p_vs_chance=float(binomtest(k, len(ed), 0.5, alternative="greater").pvalue),
            controls_canonical=sum(1 - h[i] for i in ct), n_controls=len(ct),
            accuracy=round((k + sum(1 - h[i] for i in ct)) / len(key), 3),
            per_type={t: sum(h[i] for i in ed if key[i]["type"] == t) for t in TYPES},
            confidence_mean=round(float(np.mean(list(conf.values()))), 2),
            confidence_perceived=round(float(np.mean([conf[i] for i in ed if h[i]])), 2),
            confidence_missed=round(float(np.mean([conf[i] for i in ed if not h[i]])), 2) if k < len(ed) else None,
            notes={i: resp[n][i]["note"] for i in key if resp[n][i]["note"]})
    names = list(heard)
    if len(names) == 2:
        a, b = (np.array([heard[m][i] for i in sorted(key)]) for m in names)
        ed = np.array([key[i]["type"] != "none" for i in sorted(key)])
        out["agreement"] = dict(
            percent=round(float((a == b).mean()), 3), kappa=round(float(cohen_kappa_score(a, b)), 3),
            edits_both=int((a & b)[ed].sum()), edits_either=int((a | b)[ed].sum()),
            edits_both_per_type={t: int(sum(heard[names[0]][i] & heard[names[1]][i] for i in key if key[i]["type"] == t)) for t in TYPES},
            controls_both_canonical=int(((1 - a) & (1 - b))[~ed].sum()))
    # mild versus strong edits: the factor is the duration ratio applied to the target segment
    fac = {t: [(key[i]["factor"], [heard[m][i] for m in names]) for i in key if key[i]["type"] == t] for t in TYPES}
    out["factor_by_type"] = {t: sorted(v) for t, v in fac.items()}
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "factor_by_type"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
