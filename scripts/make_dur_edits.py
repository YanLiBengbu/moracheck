"""Duration edits of native speech (PSOLA in Praat via parselmouth).

Errors: long vowel shortened, short vowel lengthened, geminate dropped, geminate added,
moraic nasal dropped. Every output, including the correct controls, passes through the
same overlap-add resynthesis with a scaled segment, so resynthesis is not a cue.
Usage: make_dur_edits.py SPLIT N_PER_UTT OUT_DIR
"""
import json, os, sys, random, zlib
from multiprocessing import Pool
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, os.path.dirname(__file__))
from jphon import VOWELS, GEM_OK, phone_labels

FR = 0.02
# scale factors applied to the refined region (central part of the voiced vowel region,
# or the voiceless closure/frication region between two vowels)
RANGES = {"err_nasdrop": (0.03, 0.12), "ctrl": (0.8, 1.2)}
# native category durations in units of the utterance's mean mora duration, measured on
# the JSUT BASIC5000 labels (interquartile range). An edit samples a target duration from
# the opposite category, so an edited segment falls inside the native range of the
# category the learner produced.
NATIVE = {"short_v": (0.31, 0.60), "long_v": (0.89, 1.22), "single_obs": (0.51, 0.86), "gem_obs": (1.03, 1.34)}


def voicing(x):
    """Voiced/unvoiced decision every 10 ms (Praat pitch)."""
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=16000)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    t = pitch.xs()
    v = pitch.selected_array["frequency"] > 0
    return t, v


def voiced_core(t, v, a, b, keep=0.7):
    """Central part of the voiced frames inside [a, b]."""
    m = (t >= a) & (t <= b) & v
    if m.sum() < 3:
        return None
    va, vb = t[m][0], t[m][-1]
    pad = (vb - va) * (1 - keep) / 2
    return va + pad, vb - pad


def unvoiced_run(t, v, a, b):
    """Longest voiceless run that overlaps [a, b]."""
    best, cur = None, None
    for k in range(len(t)):
        if not v[k]:
            cur = (t[k], t[k]) if cur is None else (cur[0], t[k])
        else:
            if cur is not None and cur[1] >= a and cur[0] <= b:
                if best is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
            cur = None
    if cur is not None and cur[1] >= a and cur[0] <= b:
        if best is None or cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def psola(x, t0, t1, f, extra=()):
    """Overlap-add resynthesis with region [t0, t1] scaled by f; extra holds further
    non-overlapping (t0, t1, f) regions scaled in the same pass."""
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=16000)
    manip = call(snd, "To Manipulation", 0.01, 75, 600)
    dt = call("Create DurationTier", "dt", 0, snd.duration)
    eps = 0.0005
    for a, b, g in sorted([(t0, t1, f)] + list(extra)):
        call(dt, "Add point", max(a - eps, 0), 1.0)
        call(dt, "Add point", a, g)
        call(dt, "Add point", b, g)
        call(dt, "Add point", min(b + eps, snd.duration), 1.0)
    call([manip, dt], "Replace duration tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32)


def sites(ph):
    s = []
    for i in range(1, len(ph)):
        if ph[i] in VOWELS and ph[i - 1] == ph[i]:
            s.append(("long_short", i))
    for i, p in enumerate(ph):
        if p in VOWELS and (i + 1 >= len(ph) or ph[i + 1] != p) and (i == 0 or ph[i - 1] != p):
            s.append(("short_long", i))
        if p == "cl" and i + 1 < len(ph):
            s.append(("gem_drop", i))
        if p == "N":
            s.append(("nas_drop", i))
        if i > 0 and p in GEM_OK and ph[i - 1] in VOWELS:
            s.append(("gem_add", i))
    return s


def mora_dur(ph, spans):
    n = sum(1 for p in ph if p in VOWELS or p in ("N", "cl"))
    return (spans[-1][1] - spans[0][0]) * FR / max(n, 1)


# control condition without calibration: fixed factors around one half and two
FIXED = os.environ.get("DUR_MODE", "calibrated") == "fixed"


def edit(x, ph, spans, etype, i, rng, tv):
    """Returns (y, C, labels, R, factor). C is the native phones (the correct target)."""
    sec = lambda f: f * FR
    t, v = tv
    mu = mora_dur(ph, spans)
    C = list(ph)
    target = lambda cat: rng.uniform(*NATIVE[cat]) * mu
    if etype in ("long_short", "short_long"):
        a0 = sec(spans[i - 1][0]) if etype == "long_short" else sec(spans[i][0])
        m = (t >= a0) & (t <= sec(spans[i][1])) & v
        reg = voiced_core(t, v, a0, sec(spans[i][1]))
        if reg is None:
            return None
        d_voiced = t[m][-1] - t[m][0]
        T = target("short_v" if etype == "long_short" else "long_v")
        f = 1 + (T - d_voiced) / (reg[1] - reg[0])
        if etype == "long_short":
            f = min(max(f, 0.08), 0.7); R = C[:i] + C[i + 1:]
        else:
            f = min(max(f, 1.4), 5.0); R = C[:i + 1] + [C[i]] + C[i + 1:]
        lab_pos = i
    elif etype == "nas_drop":  # delete the moraic nasal
        reg = voiced_core(t, v, sec(spans[i][0]), sec(spans[i][1]), keep=0.8)
        f = rng.uniform(*RANGES["err_nasdrop"])
        R = C[:i] + C[i + 1:]; lab_pos = i
    elif etype == "gem_drop":  # cl + voiceless obstruent: shorten the closure/frication
        reg = unvoiced_run(t, v, sec(spans[i][0]), sec(spans[i + 1][1]))
        if reg is None:
            return None
        f = min(max(target("single_obs") / max(reg[1] - reg[0], 1e-3), 0.15), 0.8)
        R = C[:i] + C[i + 1:]; lab_pos = i
    elif etype == "gem_add":  # voiceless obstruent after a vowel: lengthen closure/frication
        reg = unvoiced_run(t, v, sec(spans[i - 1][1]) - 0.04, sec(spans[i][1]))
        if reg is None:
            return None
        f = min(max(target("gem_obs") / max(reg[1] - reg[0], 1e-3), 1.4), 4.0)
        R = C[:i] + ["cl"] + C[i:]; lab_pos = i - 1
    max_len = 0.3 if etype.startswith("gem") else 0.6
    if reg is None or reg[1] - reg[0] < 0.02 or reg[1] - reg[0] > max_len:
        return None
    if FIXED and etype != "nas_drop":
        f = rng.uniform(0.45, 0.55) if etype in ("long_short", "gem_drop") else rng.uniform(1.8, 2.2)
    # learners who drop a geminate lengthen the following CV, and learners who add one
    # shorten the preceding CV (Yamakawa et al., 2021)
    extra = []
    if etype == "gem_drop" and i + 2 < len(C) and C[i + 2] in VOWELS:
        nb = voiced_core(t, v, sec(spans[i + 2][0]), sec(spans[i + 2][1]))
        if nb is not None and nb[0] > reg[1]:
            extra.append((nb[0], nb[1], rng.uniform(1.1, 1.35)))
    if etype == "gem_add" and C[i - 1] in VOWELS:
        nb = voiced_core(t, v, sec(spans[i - 1][0]), sec(spans[i - 1][1]))
        if nb is not None and nb[1] < reg[0]:
            extra.append((nb[0], nb[1], rng.uniform(0.75, 0.9)))
    y = psola(x, reg[0], reg[1], f, extra)
    return y, C, phone_labels(C, lab_pos), R, f


def control(x, ph, spans, rng):
    i = rng.randrange(len(ph))
    t0, t1 = spans[i][0] * FR, spans[i][1] * FR
    if t1 - t0 < 0.02:
        t0, t1 = 0.0, 0.05
    f = rng.uniform(*RANGES["ctrl"])
    return psola(x, t0, t1, f), list(ph), [0] * len(ph), list(ph), f


def work(args):
    it, k, seed, out_dir = args
    rng = random.Random(seed)
    try:
        x, _ = sf.read(it["wav"], dtype="float32")
        ph, spans = it["phones"], it["spans"]
        tv = voicing(x)
        st = sites(ph)
        res = []
        for j in range(k):
            if rng.random() < 0.4 or not st:
                r = control(x, ph, spans, rng); et = "none"
            else:  # error type uniform first, then a site of that type
                by_t = {}
                for e in st:
                    by_t.setdefault(e[0], []).append(e)
                et = rng.choice(sorted(by_t))
                et, i = rng.choice(by_t[et])
                r = edit(x, ph, spans, et, i, rng, tv)
            if r is None:
                continue
            y, C, lab, R, f = r
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.9
            p = f"{out_dir}/{it['id']}_{j}.wav"
            sf.write(p, y, 16000, subtype="PCM_16")
            res.append(dict(wav=p, C=C, labels=lab, R=R, type=et, factor=round(f, 3), src=it["id"], spk=it["spk"]))
        return res
    except Exception as e:
        print("FAIL", it["id"], e, flush=True)
        return []


if __name__ == "__main__":
    split, k, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
    items = [it for it in items if it["split"] == split]
    if len(sys.argv) > 4:
        random.Random(1).shuffle(items); items = items[:int(sys.argv[4])]
    jobs = [(it, k, zlib.crc32((it["id"] + split).encode()), out_dir) for it in items]
    out = []
    with Pool(10) as pool:
        for r in pool.imap_unordered(work, jobs, chunksize=16):
            out += r
    with open(f"{out_dir}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(out), Counter(r["type"] for r in out))
