"""Lexical text flips on held-out native speakers: flips that turn one word of the
sentence into another existing word (minimal pairs such as denki -> tenki).
The lexicon is the set of word pronunciations in all native transcripts."""
import json, random, sys, os
import pyopenjtalk
sys.path.insert(0, os.path.dirname(__file__))
from jphon import flip_candidates, phone_labels, GROUP, g2p


def word_phones(text):
    """Phones of each word of a sentence, from the OpenJTalk front end."""
    out = []
    for w in pyopenjtalk.run_frontend(text):
        if w["mora_size"] > 0 and w["pron"]:
            ph = [p for p in g2p(w["pron"].replace("’", "")) if p != "pau"]
            if ph:
                out.append(ph)
    return out


items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
lex = set()
for it in items:
    if it["text"]:
        for ph in word_phones(it["text"]):
            if len(ph) >= 3:
                lex.add(tuple(ph))
print("lexicon", len(lex))
rng = random.Random(7)
valid = [it for it in items if it["split"] == "valid" and it["text"]]
rng.shuffle(valid)
out = []
for it in valid:
    R = it["phones"]
    wp = word_phones(it["text"])
    if sum(len(w) for w in wp) != len(R):
        continue
    starts, k = [], 0
    for w in wp:
        starts.append((k, k + len(w))); k += len(w)
    cands = []
    for c in flip_candidates(R):
        et, C, _, pos = c
        d = len(C) - len(R)
        for (a, b) in starts:
            if a <= pos < b + max(d, 0):
                new = tuple(C[a:b + d])
                if len(new) >= 3 and new in lex and new != tuple(R[a:b]):
                    cands.append(c)
                break
    rec = dict(wav=it["wav"], C=R, labels=[0] * len(R), R=R, type="none", src=it["id"], spk=it["spk"])
    if cands and rng.random() < 0.5:
        et, C, _, pos = rng.choice(cands)
        rec.update(C=C, labels=phone_labels(C, pos), type=et)
    elif not cands:
        continue
    out.append(rec)
    if len(out) >= 1500:
        break
with open("data/sim/val_lex.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(GROUP.get(r["type"], "none") for r in out))
