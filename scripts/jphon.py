"""Japanese phoneme and mora utilities, and the catalogue of learner pronunciation errors.

Phonemes follow OpenJTalk (pyopenjtalk.g2p). Devoiced vowels (A I U E O) are mapped to
lowercase. A long vowel is a vowel-only mora that repeats the previous vowel.

An error operation maps a canonical phoneme sequence C to a realized sequence R and
marks the canonical mora that is mispronounced. Text flips run the operation in
reverse on native speech: the audio realizes R, and the target text is set to C.
"""
import random
import pyopenjtalk

VOWELS = set("aiueo")
SPECIAL = {"N", "cl"}
PAUSE = {"pau", "sil"}

PHONES = ["<pad>", "<unk>", "a", "i", "u", "e", "o", "N", "cl",
          "k", "ky", "g", "gy", "s", "sh", "z", "j", "t", "ch", "ts", "d", "dy", "ty",
          "n", "ny", "h", "hy", "f", "b", "by", "p", "py", "m", "my", "y", "r", "ry",
          "w", "v", "pau"]
PHONE2ID = {p: i for i, p in enumerate(PHONES)}

# error types used everywhere in the paper
ERROR_TYPES = ["voicing", "devoicing", "long_short", "short_long", "gem_drop", "gem_add",
               "nas_drop", "nas_add", "pal_merge", "pal_split", "tsu", "zj", "rd"]
# coarse groups that follow the contrasts of Japanese pronunciation teaching
GROUP = {"voicing": "voicing", "devoicing": "voicing",
         "long_short": "long vowel", "short_long": "long vowel",
         "gem_drop": "geminate", "gem_add": "geminate",
         "nas_drop": "moraic nasal", "nas_add": "moraic nasal",
         "pal_merge": "palatal", "pal_split": "palatal",
         "tsu": "affricate", "zj": "affricate", "rd": "liquid"}

VOICE = {"k": "g", "t": "d", "s": "z", "sh": "j", "ch": "j", "h": "b", "p": "b",
         "ky": "gy", "hy": "by", "py": "by", "f": "b"}
DEVOICE = {"g": "k", "d": "t", "z": "s", "j": "ch", "b": "p", "gy": "ky", "by": "py"}
PAL = {"k": "ky", "g": "gy", "n": "ny", "h": "hy", "b": "by", "p": "py", "m": "my", "r": "ry"}
UNPAL = {v: k for k, v in PAL.items()}
GEM_OK = {"k", "ky", "s", "sh", "t", "ch", "ts", "p", "py"}


def g2p(text):
    """Phonemes of a text, devoicing removed, pauses kept as 'pau'."""
    ph = pyopenjtalk.g2p(text).split()
    return [p.lower() if p in ("A", "I", "U", "E", "O") else p for p in ph]


def kana2phones(kana):
    return g2p(kana)


def moras(ph):
    """Split phonemes into moras. Returns a list of (start, end, kind) over ph indices.
    kind is one of CV, V, R (long vowel), N, Q, P (pause)."""
    out, i, prev_v = [], 0, None
    while i < len(ph):
        p = ph[i]
        if p in PAUSE:
            out.append((i, i + 1, "P")); prev_v = None; i += 1
        elif p == "N":
            out.append((i, i + 1, "N")); prev_v = None; i += 1
        elif p == "cl":
            out.append((i, i + 1, "Q")); prev_v = None; i += 1
        elif p in VOWELS:
            out.append((i, i + 1, "R" if p == prev_v else "V")); prev_v = p; i += 1
        elif i + 1 < len(ph) and ph[i + 1] in VOWELS:
            out.append((i, i + 2, "CV")); prev_v = ph[i + 1]; i += 2
        else:  # stray consonant
            out.append((i, i + 1, "CV")); prev_v = None; i += 1
    return out


def mora_of_phone(ph):
    idx = [0] * len(ph)
    for m, (s, e, _) in enumerate(moras(ph)):
        for j in range(s, e):
            idx[j] = m
    return idx


def to_ids(ph):
    return [PHONE2ID.get(p, 1) for p in ph]


# ---------------------------------------------------------------------------
# error operations. Each candidate is (type, C, R, err_mora_in_C)
# where C is canonical, R is realized. Functions enumerate all sites.
# ---------------------------------------------------------------------------

def _sites_substitution(ph, table, etype):
    out = []
    for i, p in enumerate(ph):
        if p in table and i + 1 < len(ph) and ph[i + 1] in VOWELS:
            r = ph[:i] + [table[p]] + ph[i + 1:]
            out.append((etype, ph, r, i))
    return out


def realize_errors(C):
    """All single-site errors that a learner could make on canonical C.
    Returns list of (type, C, R, phone_index_in_C)."""
    out = []
    out += _sites_substitution(C, VOICE, "voicing")
    out += _sites_substitution(C, DEVOICE, "devoicing")
    for i in range(1, len(C)):
        # long vowel shortened: drop the repeated vowel
        if C[i] in VOWELS and C[i - 1] == C[i] and (i < 2 or C[i - 2] != C[i]):
            out.append(("long_short", C, C[:i] + C[i + 1:], i))
    for i, p in enumerate(C):
        # short vowel lengthened (not already long, not before another same vowel)
        if p in VOWELS and (i + 1 >= len(C) or C[i + 1] != p) and (i == 0 or C[i - 1] != p):
            out.append(("short_long", C, C[:i + 1] + [p] + C[i + 1:], i))
        if p == "cl":
            out.append(("gem_drop", C, C[:i] + C[i + 1:], i))
        if p == "N":
            out.append(("nas_drop", C, C[:i] + C[i + 1:], i))
    for i in range(1, len(C)):
        # geminate inserted before a voiceless obstruent that follows a vowel
        if C[i] in GEM_OK and C[i - 1] in VOWELS:
            out.append(("gem_add", C, C[:i] + ["cl"] + C[i:], i - 1))
        # moraic nasal inserted after a vowel before a consonant
        if C[i - 1] in VOWELS and C[i] not in VOWELS and C[i] not in SPECIAL and C[i] not in PAUSE:
            out.append(("nas_add", C, C[:i] + ["N"] + C[i:], i - 1))
    for i in range(len(C) - 3):
        # C i y V -> Cy V   (biyooiN -> byooiN)
        if C[i] in PAL and C[i + 1] == "i" and C[i + 2] == "y" and C[i + 3] in VOWELS:
            out.append(("pal_merge", C, C[:i] + [PAL[C[i]]] + C[i + 3:], i))
    for i in range(len(C) - 1):
        # Cy V -> C i y V   (byooiN -> biyooiN)
        if C[i] in UNPAL and C[i + 1] in VOWELS:
            out.append(("pal_split", C, C[:i] + [UNPAL[C[i]], "i", "y"] + C[i + 1:], i))
        if C[i] == "ts" and C[i + 1] == "u":
            out.append(("tsu", C, C[:i] + ["ch"] + C[i + 1:], i))
            out.append(("tsu", C, C[:i] + ["s"] + C[i + 1:], i))
        if C[i] == "z" and C[i + 1] in "auo":
            out.append(("zj", C, C[:i] + ["j"] + C[i + 1:], i))
        if C[i] == "j" and C[i + 1] in "auo":
            out.append(("zj", C, C[:i] + ["z"] + C[i + 1:], i))
        if C[i] == "r" and C[i + 1] in VOWELS:
            out.append(("rd", C, C[:i] + ["d"] + C[i + 1:], i))
        if C[i] == "d" and C[i + 1] in "aeo":
            out.append(("rd", C, C[:i] + ["r"] + C[i + 1:], i))
    return out


def flip_candidates(R):
    """Text flips on native audio that realizes R: canonical sequences C such that a
    learner who was asked to say C and said R made one error. Returns
    (type, C, R, phone_index_in_C)."""
    out = []
    # voicing error = canonical voiceless, audio voiced
    for i, p in enumerate(R):
        if i + 1 < len(R) and R[i + 1] in VOWELS:
            if p in DEVOICE:
                out.append(("voicing", R[:i] + [DEVOICE[p]] + R[i + 1:], R, i))
            if p in VOICE and p != "f":
                out.append(("devoicing", R[:i] + [VOICE[p]] + R[i + 1:], R, i))
    for i in range(1, len(R)):
        if R[i] in VOWELS and R[i - 1] == R[i]:
            # audio long, canonical short -> learner lengthened
            out.append(("short_long", R[:i] + R[i + 1:], R, i - 1))
    for i, p in enumerate(R):
        if p in VOWELS and (i + 1 >= len(R) or R[i + 1] != p) and (i == 0 or R[i - 1] != p):
            # audio short, canonical long -> learner shortened
            out.append(("long_short", R[:i + 1] + [p] + R[i + 1:], R, i + 1))
        if p == "cl" and i + 1 < len(R):
            out.append(("gem_add", R[:i] + R[i + 1:], R, max(i - 1, 0)))
        if p == "N":
            out.append(("nas_add", R[:i] + R[i + 1:], R, max(i - 1, 0)))
    for i in range(1, len(R)):
        if R[i] in GEM_OK and R[i - 1] in VOWELS:
            out.append(("gem_drop", R[:i] + ["cl"] + R[i:], R, i))
        if R[i - 1] in VOWELS and R[i] not in VOWELS and R[i] not in SPECIAL and R[i] not in PAUSE:
            out.append(("nas_drop", R[:i] + ["N"] + R[i:], R, i))
    for i in range(len(R) - 1):
        if R[i] in UNPAL and R[i + 1] in VOWELS:
            # audio Cy V, canonical C i y V -> learner merged
            out.append(("pal_merge", R[:i] + [UNPAL[R[i]], "i", "y"] + R[i + 1:], R, i))
        if R[i] == "ch" and R[i + 1] == "u":
            out.append(("tsu", R[:i] + ["ts"] + R[i + 1:], R, i))
        if R[i] == "j" and R[i + 1] in "auo":
            out.append(("zj", R[:i] + ["z"] + R[i + 1:], R, i))
        if R[i] == "z" and R[i + 1] in "auo":
            out.append(("zj", R[:i] + ["j"] + R[i + 1:], R, i))
        if R[i] == "d" and R[i + 1] in "aeo":
            out.append(("rd", R[:i] + ["r"] + R[i + 1:], R, i))
    for i in range(len(R) - 3):
        if R[i] in PAL and R[i + 1] == "i" and R[i + 2] == "y" and R[i + 3] in VOWELS:
            out.append(("pal_split", R[:i] + [PAL[R[i]]] + R[i + 3:], R, i))
    return out


def phone_labels(C, err_phone):
    """Per-phone 0/1 labels on C: all phones of the mora that holds err_phone are 1."""
    midx = mora_of_phone(C)
    m = midx[err_phone]
    return [1 if k == m else 0 for k in midx]


if __name__ == "__main__":
    for t in ["病院へ行きました", "おばあさんが酸っぱい天気だと言った", "全員で合唱する", "通信と王座"]:
        ph = g2p(t)
        print(t, " ".join(ph), [k for _, _, k in moras(ph)])
        rs = realize_errors(ph)
        fs = flip_candidates(ph)
        print("  realize:", len(rs), " flips:", len(fs))
        random.seed(0)
        for e in random.sample(fs, 3):
            print("   flip", e[0], "C=", "".join(e[1]), "R=", "".join(e[2]), "at", e[3], e[1][e[3]])
