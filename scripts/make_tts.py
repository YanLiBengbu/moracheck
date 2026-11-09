"""TTS swaps: Kokoro synthesizes the canonical sentence and a version with one learner
error. An error example is kept only if the CTC recognizer prefers the realized phones
over the canonical phones (log-likelihood margin), and a correct example only if it
prefers the canonical phones over the same error. jf_tebukuro is held out for validation.
Usage: make_tts.py SPLIT N_SENT OUT_DIR [SEED] [SHARD NSHARD] [DEVICE]
"""
import json, os, sys, random
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from jphon import realize_errors, phone_labels, GROUP
from tts_kokoro import TTS
from ctc_align import CTC

TRAIN_VOICES = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jm_kumo"]
VAL_VOICES = ["jf_tebukuro"]
MARGIN = 1.0


def ctc_ll(ctc, lp, ph):
    tgt = torch.tensor([ctc.ids(ph)], device=lp.device)
    if tgt.shape[1] == 0 or tgt.shape[1] * 2 > lp.shape[0]:
        return None
    loss = F.ctc_loss(lp[:, None], tgt, torch.tensor([lp.shape[0]]), torch.tensor([tgt.shape[1]]),
                      blank=0, reduction="sum", zero_infinity=True)
    return -float(loss)


def main():
    split, n, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shard, nshard = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
    device = sys.argv[7] if len(sys.argv) > 7 else "cuda"
    torch.set_num_threads(2)
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    items = [json.loads(l) for l in open("data/native_clean.jsonl", encoding="utf-8")]
    items = [it for it in items if it["split"] == split and 6 <= len(it["phones"]) <= 60]
    rng.shuffle(items)
    items = items[shard::nshard]
    rng = random.Random(seed * 100 + shard)
    n = n // nshard
    voices = VAL_VOICES if split == "valid" else TRAIN_VOICES
    groups = sorted(set(GROUP.values()))
    tts, ctc = TTS(device), CTC(device)
    out, stats = [], {"err_kept": 0, "err_drop": 0, "ok_kept": 0, "ok_drop": 0}
    for it in items:
        if len(out) >= 2 * n:
            break
        C = it["phones"]
        errs = realize_errors(C)
        by_g = {}
        for e in errs:
            by_g.setdefault(GROUP[e[0]], []).append(e)
        if not by_g:
            continue
        g = rng.choice([g for g in groups if g in by_g])
        et, _, R, pos = rng.choice(by_g[g])
        v = rng.choice(voices)
        speed = rng.uniform(0.9, 1.15)
        ac = librosa.resample(tts.synth(C, v, speed), orig_sr=24000, target_sr=16000)
        ar = librosa.resample(tts.synth(R, v, speed), orig_sr=24000, target_sr=16000)
        lpc, lpr = ctc.logprobs(ac), ctc.logprobs(ar)
        llc_c, llc_r = ctc_ll(ctc, lpc, C), ctc_ll(ctc, lpc, R)
        llr_c, llr_r = ctc_ll(ctc, lpr, C), ctc_ll(ctc, lpr, R)
        if None in (llc_c, llc_r, llr_c, llr_r):
            continue
        base = f"{out_dir}/{it['id']}"
        if llc_c - llc_r > MARGIN:
            sf.write(base + "_c.wav", ac, 16000, subtype="PCM_16")
            out.append(dict(wav=base + "_c.wav", C=C, labels=[0] * len(C), R=C, type="none", voice=v, src=it["id"], margin=round(llc_c - llc_r, 2)))
            stats["ok_kept"] += 1
        else:
            stats["ok_drop"] += 1
        if llr_r - llr_c > MARGIN:
            sf.write(base + "_e.wav", ar, 16000, subtype="PCM_16")
            out.append(dict(wav=base + "_e.wav", C=C, labels=phone_labels(C, pos), R=R, type=et, voice=v, src=it["id"], margin=round(llr_r - llr_c, 2)))
            stats["err_kept"] += 1
        else:
            stats["err_drop"] += 1
        if len(out) % 500 == 0:
            print(len(out), stats, flush=True)
    suffix = f".part{shard}" if nshard > 1 else ""
    with open(f"{out_dir}.jsonl{suffix}", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("done", len(out), stats, Counter(r["type"] for r in out))


if __name__ == "__main__":
    main()
