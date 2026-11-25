"""Blind listening check of the duration edits for a Japanese teacher.

40 edited clips (8 of each length error type) and 20 controls that only went through the
same resynthesis. For each clip the rater sees the sentence with the target word blanked
and chooses which of two katakana forms was heard. The answer key stays in exp/annotation.
Usage: make_listening_test.py OUT_DIR
"""
import json, os, random, sys
import numpy as np
import soundfile as sf
import pyopenjtalk
sys.path.insert(0, os.path.dirname(__file__))
import make_dur_edits as M
from jphon import g2p, realize_errors, VOWELS

KANA_V = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}
SIMPLE = {"k": "カキクケコ", "g": "ガギグゲゴ", "n": "ナニヌネノ", "h": "ハヒフヘホ", "b": "バビブベボ",
          "p": "パピプペポ", "m": "マミムメモ", "r": "ラリルレロ"}
SPECIAL = {("s", "a"): "サ", ("s", "u"): "ス", ("s", "e"): "セ", ("s", "o"): "ソ", ("s", "i"): "スィ",
           ("z", "a"): "ザ", ("z", "u"): "ズ", ("z", "e"): "ゼ", ("z", "o"): "ゾ", ("z", "i"): "ズィ",
           ("t", "a"): "タ", ("t", "e"): "テ", ("t", "o"): "ト", ("t", "i"): "ティ", ("t", "u"): "トゥ",
           ("d", "a"): "ダ", ("d", "e"): "デ", ("d", "o"): "ド", ("d", "i"): "ディ", ("d", "u"): "ドゥ",
           ("sh", "i"): "シ", ("j", "i"): "ジ", ("ch", "i"): "チ", ("ts", "u"): "ツ", ("f", "u"): "フ", ("f", "a"): "ファ", ("f", "i"): "フィ",
           ("f", "e"): "フェ", ("f", "o"): "フォ",
           ("y", "a"): "ヤ", ("y", "u"): "ユ", ("y", "o"): "ヨ", ("y", "e"): "イェ",
           ("w", "a"): "ワ", ("w", "o"): "ヲ", ("w", "i"): "ウィ", ("w", "e"): "ウェ",
           ("v", "a"): "ヴァ", ("v", "i"): "ヴィ", ("v", "u"): "ヴ", ("v", "e"): "ヴェ", ("v", "o"): "ヴォ"}
YOON_BASE = {"ky": "キ", "gy": "ギ", "ny": "ニ", "hy": "ヒ", "by": "ビ", "py": "ピ", "my": "ミ", "ry": "リ",
             "sh": "シ", "j": "ジ", "ch": "チ", "ty": "テ", "dy": "デ", "f": "フ", "ts": "ツ"}
SMALL = {"a": "ャ", "u": "ュ", "o": "ョ", "e": "ェ", "i": "ィ"}
VIDX = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}


def to_kata(ph):
    out, prev_v, i = [], None, 0
    while i < len(ph):
        p = ph[i]
        if p == "N":
            out.append("ン"); prev_v = None; i += 1
        elif p == "cl":
            out.append("ッ"); prev_v = None; i += 1
        elif p in VOWELS:
            out.append("ー" if p == prev_v else KANA_V[p]); prev_v = p; i += 1
        elif i + 1 < len(ph) and ph[i + 1] in VOWELS:
            v = ph[i + 1]
            if (p, v) in SPECIAL:
                k = SPECIAL[(p, v)]
            elif p in SIMPLE:
                k = SIMPLE[p][VIDX[v]]
            elif p in YOON_BASE:
                k = YOON_BASE[p] + (SMALL[v] if not (p in ("f", "ts") and v == "u") else "")
            else:
                k = p + v
            out.append(k); prev_v = v; i += 2
        else:
            out.append(p); prev_v = None; i += 1
    return "".join(out)


def word_spans(text, n_phones):
    """(start, end, surface, usable) per word. A word that ends in a geminate is merged
    with the next word. Particles, auxiliaries and one-mora words are not usable targets."""
    raw, k = [], 0
    for w in pyopenjtalk.run_frontend(text):
        if w["mora_size"] > 0 and w["pron"]:
            ph = [p for p in g2p(w["pron"].replace("’", "")) if p != "pau"]
            if not ph:
                return None
            raw.append(dict(a=k, b=k + len(ph), s=w["string"], pos=w["pos"], m=w["mora_size"], q=ph[-1] == "cl"))
            k += len(ph)
        elif w["string"].strip():
            raw.append(dict(a=k, b=k, s=w["string"], pos="記号", m=0, q=False))
    if k != n_phones:
        return None
    merged = []
    for r in raw:
        if merged and merged[-1]["q"] and r["m"] > 0:
            m = merged[-1]
            merged[-1] = dict(a=m["a"], b=r["b"], s=m["s"] + r["s"], pos=m["pos"], m=m["m"] + r["m"], q=r["q"])
        else:
            merged.append(dict(r))
    return [(r["a"], r["b"], r["s"], r["pos"] not in ("助詞", "助動詞", "記号", "フィラー") and r["m"] >= 2)
            for r in merged]


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(2026)
    items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
    items = [it for it in items if it["text"] and it["dur"] <= 6.0 and (it["corpus"] == "jsut" or it["split"] == "valid")]
    rng.shuffle(items)
    want = {"long_short": 8, "short_long": 8, "gem_drop": 8, "gem_add": 8, "nas_drop": 8, "none": 20}
    got = {k: [] for k in want}
    for it in items:
        if all(len(got[k]) >= want[k] for k in want):
            break
        ph = it["phones"]
        ws = word_spans(it["text"], len(ph))
        if ws is None:
            continue
        x, _ = sf.read(it["wav"], dtype="float32")
        tv = M.voicing(x)
        sites = M.sites(ph)
        need = [t for t in want if t != "none" and len(got[t]) < want[t]]
        is_ctrl = len(got["none"]) < want["none"] and (not need or rng.random() < 0.33)
        cand = [s for s in sites if (s[0] in need) or is_ctrl]
        if not cand:
            continue
        et, i = rng.choice(cand)
        r = M.edit(x, ph, it["spans"], et, i, rng, tv)
        if r is None:
            continue
        y, C, lab, R, f = r
        d = len(R) - len(C)
        lo, hi = {"long_short": (i - 1, i), "gem_drop": (i, i + 1), "gem_add": (i - 1, i)}.get(et, (i, i))
        wsel = [(a, b, s, u) for (a, b, s, u) in ws if a <= lo and hi < b]
        if not wsel or not wsel[0][3]:
            continue
        a, b, surf, _ = wsel[0]
        canon = to_kata(C[a:b]); err = to_kata(R[a:b + d])
        if canon == err or "?" in canon + err or any(c.isascii() for c in canon + err):
            continue
        if is_ctrl:
            y, _, _, _, f = M.control(x, ph, it["spans"], rng); kind = "none"
        else:
            kind = et
        if len(got[kind]) >= want[kind]:
            continue
        sent = "".join(s if not (aa == a and bb == b) else "【　？　】" for (aa, bb, s, _) in ws)
        got[kind].append(dict(src=it["id"], text=it["text"], blanked=sent, canon=canon, err=err, type=kind,
                              factor=round(float(f), 3), audio=y))
    clips = [c for k in got for c in got[k]]
    rng.shuffle(clips)
    key = []
    for n, c in enumerate(clips, 1):
        swap = rng.random() < 0.5
        A, B = (c["err"], c["canon"]) if swap else (c["canon"], c["err"])
        y = c.pop("audio"); y = y / (np.abs(y).max() + 1e-9) * 0.9
        sf.write(f"{out_dir}/{n:02d}.wav", y, 16000, subtype="PCM_16")
        key.append(dict(no=n, A=A, B=B, error_option="A" if swap else "B", **c))
    os.makedirs("annotation", exist_ok=True)
    json.dump(key, open("annotation/listening_key.json", "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(len(clips), Counter(c["type"] for c in key))
    return key


if __name__ == "__main__":
    main()
