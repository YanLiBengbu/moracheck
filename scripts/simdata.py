"""Training and validation data for the detector: native speech with simulated errors.

Three sources of simulated errors
  flip : text flips on native audio (realized = native phones, canonical = flipped)
  dur  : duration edits of native audio (precomputed by make_dur_edits.py)
  tts  : TTS swaps (precomputed by make_tts.py), verified by the CTC recognizer
Each source also contributes correct examples, so no source is a cue for an error.
"""
import json, random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from jphon import (flip_candidates, phone_labels, moras, to_ids, GROUP)
from model import KINDS

GROUPS = sorted(set(GROUP.values()))


def kinds_of(ph):
    k = [0] * len(ph)
    for s, e, kind in moras(ph):
        for j in range(s, e):
            k[j] = KINDS.index(kind)
    return k


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


class SimSet(Dataset):
    """Mixture of sources. weights: dict source -> prob; group_w: dict group -> prob for
    choosing the error group of flips; p_err: probability that a flip example has an error."""

    def __init__(self, native, dur=None, tts=None, weights=None, group_w=None, p_err=0.6, seed=0, epoch_size=40000):
        self.native = native
        self.dur = dur or []
        self.tts = tts or []
        self.weights = weights or {"flip": 1.0}
        self.group_w = group_w or {g: 1.0 for g in GROUPS}
        self.p_err = p_err
        self.rng = random.Random(seed)
        self.epoch_size = epoch_size
        self.sources = [s for s in self.weights if self.weights[s] > 0 and (s == "flip" or getattr(self, s))]
        self.sw = [self.weights[s] for s in self.sources]

    def __len__(self):
        return self.epoch_size

    def _flip(self):
        it = self.rng.choice(self.native)
        R = it["phones"]
        C, lab = R, [0] * len(R)
        if self.rng.random() < self.p_err:
            cands = flip_candidates(R)
            by_g = {}
            for c in cands:
                by_g.setdefault(GROUP[c[0]], []).append(c)
            gs = [g for g in by_g if self.group_w.get(g, 0) > 0]
            if gs:
                g = self.rng.choices(gs, [self.group_w[g] for g in gs])[0]
                et, C, _, pos = self.rng.choice(by_g[g])
                lab = phone_labels(C, pos)
        return it["wav"], C, lab, R, None

    def _pre(self, pool):
        it = self.rng.choice(pool)
        return it["wav"], it["C"], it["labels"], it["R"], it.get("offset")

    def __getitem__(self, i):
        src = self.rng.choices(self.sources, self.sw)[0]
        wav, C, lab, R, _ = self._flip() if src == "flip" else self._pre(getattr(self, src))
        x, sr = sf.read(wav, dtype="float32")
        return dict(wav=x, ph=to_ids(C), kind=kinds_of(C), lab=lab, real=to_ids(R), src=src)


class FixedSet(Dataset):
    """A frozen list of examples (validation or test): dicts with wav, C, labels, R."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x, sr = sf.read(it["wav"], dtype="float32")
        if "start" in it:
            x = x[int(it["start"] * 16000): int(it["end"] * 16000)]
        return dict(wav=x, ph=to_ids(it["C"]), kind=kinds_of(it["C"]), lab=it.get("labels", [0] * len(it["C"])),
                    real=to_ids(it.get("R", it["C"])), idx=i)


def collate(batch):
    B = len(batch)
    T = max(len(b["wav"]) for b in batch)
    L = max(len(b["ph"]) for b in batch)
    R = max(len(b["real"]) for b in batch)
    wav = torch.zeros(B, T); wm = torch.zeros(B, T, dtype=torch.bool)
    ph = torch.zeros(B, L, dtype=torch.long); kind = torch.zeros(B, L, dtype=torch.long)
    lab = torch.zeros(B, L); pm = torch.zeros(B, L, dtype=torch.bool)
    real = torch.zeros(B, R, dtype=torch.long); rlen = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        x = b["wav"]
        x = (x - x.mean()) / (x.std() + 1e-7)
        wav[i, :len(x)] = torch.from_numpy(x); wm[i, :len(x)] = True
        n = len(b["ph"])
        ph[i, :n] = torch.tensor(b["ph"]); kind[i, :n] = torch.tensor(b["kind"])
        lab[i, :n] = torch.tensor(b["lab"], dtype=torch.float); pm[i, :n] = True
        real[i, :len(b["real"])] = torch.tensor(b["real"]); rlen[i] = len(b["real"])
    out = dict(wav=wav, wav_mask=wm, ph=ph, kind=kind, lab=lab, ph_mask=pm, real=real, real_len=rlen)
    if "idx" in batch[0]:
        out["idx"] = torch.tensor([b["idx"] for b in batch])
    return out
