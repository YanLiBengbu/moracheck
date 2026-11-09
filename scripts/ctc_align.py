"""CTC phone posteriors, forced alignment and greedy recognition with the off-the-shelf
Japanese phoneme CTC model (prj-beatrice/japanese-hubert-base-phoneme-ctc-v4).

Used for (1) filtering native utterances whose audio does not match the G2P phonemes,
(2) phone boundaries for duration edits, (3) the GOP and recognition baselines.
"""
import json, sys, os
import numpy as np
import torch
import torchaudio.functional as AF
import soundfile as sf
from transformers import AutoModelForCTC

MODEL = "models/japanese-hubert-base-phoneme-ctc-v4"
VOCAB = json.load(open(f"{MODEL}/vocab.json"))
ID2TOK = {v: k for k, v in VOCAB.items()}
FRAME = 0.02


class CTC:
    def __init__(self, device="cuda"):
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.m = AutoModelForCTC.from_pretrained(MODEL).to(device).eval().to(self.dtype)
        self.dev = device
        # devoiced I/U merged into i/u for canonical (lowercase) targets
        self.merge = [(VOCAB["i"], VOCAB["I"]), (VOCAB["u"], VOCAB["U"])]

    @torch.no_grad()
    def logprobs(self, wav):
        x = torch.from_numpy(wav).float()
        x = (x - x.mean()) / (x.std() + 1e-7)
        lg = self.m(x[None].to(self.dev).to(self.dtype)).logits.float()[0]
        lp = lg.log_softmax(-1)
        for a, b in self.merge:
            lp[:, a] = torch.logaddexp(lp[:, a], lp[:, b])
            lp[:, b] = -1e4
        return lp  # (T, V)

    def ids(self, phones):
        return [VOCAB.get(p, VOCAB["UNK"]) for p in phones if p != "pau"]

    def align(self, lp, phones):
        """Forced alignment. Returns per-phone (start_frame, end_frame) and the
        per-phone mean log posterior. Blank frames after a token are assigned to it."""
        tgt = self.ids(phones)
        T = lp.shape[0]
        if len(tgt) == 0 or len(tgt) > T:
            return None
        try:
            path, scores = AF.forced_align(lp[None], torch.tensor([tgt], device=lp.device), blank=0)
        except RuntimeError:
            return None
        path = path[0].cpu().numpy(); scores = scores[0].cpu().numpy()
        tok_frames = []  # first frame of each target token
        prev = 0
        for t, s in enumerate(path):
            if s != 0 and (s != prev):
                tok_frames.append(t)
            elif s != 0 and s == prev and t > 0 and path[t - 1] == 0:
                tok_frames.append(t)  # repeated token separated by blank
            prev = s
        if len(tok_frames) != len(tgt):
            return None
        last = int(np.nonzero(path)[0][-1]) + 1
        ends = tok_frames[1:] + [last]
        lpn = lp.cpu().numpy() if torch.is_tensor(lp) else lp
        post = [float(lpn[a:b, tgt[i]].max()) for i, (a, b) in enumerate(zip(tok_frames, ends))]
        return list(zip(tok_frames, ends)), post

    def greedy(self, lp):
        best = lp.argmax(-1).cpu().numpy()
        out, prev = [], 0
        for b in best:
            if b != prev and b != 0:
                out.append(ID2TOK[int(b)])
            prev = b
        return [p.lower() if p in ("I", "U") else p for p in out if p not in ("pau", "sil")]


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


if __name__ == "__main__":
    # align the native manifest: data/native.jsonl -> data/native_aligned.jsonl
    torch.set_num_threads(2)
    ctc = CTC()
    fin = sys.argv[1] if len(sys.argv) > 1 else "data/native.jsonl"
    fout = sys.argv[2] if len(sys.argv) > 2 else "data/native_aligned.jsonl"
    n = ok = 0
    done = set()
    if os.path.exists(fout):
        for l in open(fout, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    with open(fout, "a", encoding="utf-8") as fo:
        for line in open(fin, encoding="utf-8"):
            it = json.loads(line); n += 1
            if it["id"] in done:
                continue
            wav, sr = sf.read(it["wav"], dtype="float32")
            lp = ctc.logprobs(wav)
            ph = [p for p in it["phones"] if p != "pau"]
            res = ctc.align(lp, ph)
            hyp = ctc.greedy(lp)
            it["phones"] = ph
            it["per"] = round(edit_distance(ph, hyp) / max(len(ph), 1), 4)
            if res is not None:
                spans, post = res
                it["spans"] = spans
                it["post_mean"] = round(float(np.mean(post)), 4)
                ok += 1
            fo.write(json.dumps(it, ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                print(n, ok, flush=True)
    print("done", n, ok)
