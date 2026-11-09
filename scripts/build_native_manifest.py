"""Build the native-speech manifest: 16 kHz mono wavs plus canonical phonemes.

Sources: JSUT BASIC5000 (manual phonemes from jsut-label), Common Voice 17 ja
(train/dev/test), FLEURS ja_jp. Phonemes for CV and FLEURS come from pyopenjtalk.
Split: CV test speakers form the native validation set; everything else is training.
"""
import csv, json, os, glob, sys
from multiprocessing import Pool
import numpy as np
import soundfile as sf
import librosa
sys.path.insert(0, os.path.dirname(__file__))
from jphon import g2p

RAW = "data/raw"
OUT = "data/wav16"
SR = 16000


def jsut_items():
    items = []
    for lab in sorted(glob.glob(f"{RAW}/jsut-label/labels/basic5000/*.lab")):
        uid = os.path.basename(lab)[:-4]
        ph = []
        for line in open(lab):
            p = line.split()[2].split("-")[1].split("+")[0]
            if p in ("sil",):
                continue
            ph.append(p.lower() if p in ("A", "I", "U", "E", "O") else p)
        wav = f"{RAW}/jsut_ver1.1/basic5000/wav/{uid}.wav"
        items.append(dict(id=f"jsut_{uid}", src=wav, spk="jsut", corpus="jsut", split="train", phones=ph, text=""))
    texts = {}
    cur = None
    for line in open(f"{RAW}/jsut-label/text_kana/basic5000.yaml", encoding="utf-8"):
        if not line.startswith(" "):
            cur = line.strip().rstrip(":")
        elif "text_level0:" in line:
            texts[cur] = line.split("text_level0:")[1].strip()
    for it in items:
        it["text"] = texts.get(it["id"][5:], "")
    return items


def cv_items():
    items = []
    for s in ["train", "dev", "test"]:
        with open(f"{RAW}/cv17/transcript/ja/{s}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                src = f"{RAW}/cv17/clips/ja_{s}_0/{r['path']}"
                items.append(dict(id="cv_" + r["path"][:-4], src=src, spk="cv_" + r["client_id"][:16],
                                  corpus="cv17", split="valid" if s == "test" else "train",
                                  text=r["sentence"], phones=None))
    return items


def fleurs_items():
    items = []
    for s in ["train", "dev", "test"]:
        tsv = f"{RAW}/fleurs/data/ja_jp/{s}.tsv"
        if not os.path.exists(tsv):
            continue
        for row in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            src = f"{RAW}/fleurs/data/ja_jp/audio/{s}/{row[1]}"
            items.append(dict(id=f"fleurs_{s}_{row[1][:-4]}", src=src, spk=f"fleurs_{row[0]}", corpus="fleurs",
                              split="train", text=row[2], phones=None))
    return items


def process(it):
    try:
        dst = f"{OUT}/{it['corpus']}/{it['id']}.wav"
        if not os.path.exists(dst):
            x, sr = sf.read(it["src"], dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != SR:
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            peak = np.abs(x).max()
            if peak > 0:
                x = x / peak * 0.9
            sf.write(dst, x, SR, subtype="PCM_16")
            dur = len(x) / SR
        else:
            dur = sf.info(dst).duration
        if it["phones"] is None:
            it["phones"] = [p for p in g2p(it["text"]) if p != "sil"]
        it["wav"] = dst
        it["dur"] = round(dur, 3)
        del it["src"]
        return it
    except Exception as e:
        print("FAIL", it["id"], e, flush=True)
        return None


if __name__ == "__main__":
    items = jsut_items() + cv_items() + fleurs_items()
    for c in ["jsut", "cv17", "fleurs"]:
        os.makedirs(f"{OUT}/{c}", exist_ok=True)
    print(len(items), "items", flush=True)
    with Pool(14) as pool:
        res = [r for r in pool.imap(process, items, chunksize=64) if r]
    res = [r for r in res if 0.8 <= r["dur"] <= 15 and len(r["phones"]) >= 4]
    with open("data/native.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter((r["corpus"], r["split"]) for r in res))
    print("hours", sum(r["dur"] for r in res) / 3600)
