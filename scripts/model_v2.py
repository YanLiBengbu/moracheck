"""MoraCheck detector, version 2: recognizer-informed and duration-aware.

A frozen Japanese phoneme CTC model (HuBERT encoder plus CTC head) gives frame features
and phoneme posteriors. For every canonical phoneme we build
  - the mean of the weighted encoder layers over its forced-aligned segment,
  - its aligned duration, normalized by the mean mora duration of the utterance,
  - per error group, the best CTC log-likelihood ratio of a catalogue error at its mora,
  - its phoneme and mora-kind embeddings.
A transformer over the phoneme sequence, with cross-attention to the frames, gives one
mispronunciation logit per canonical phoneme.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from transformers import HubertForCTC

from jphon import PHONES
from model import KINDS, PosEnc

CTC_MODEL = "models/japanese-hubert-base-phoneme-ctc-v4"
N_GROUPS = 7


def phone_to_ctc_ids():
    voc = json.load(open(f"{CTC_MODEL}/vocab.json"))
    return [voc.get(p, voc["UNK"]) for p in PHONES]


class DetectorV2(nn.Module):
    def __init__(self, d=256, n_txt=4, n_dec=2, use_lr=True, use_dur=True, use_seg=True, use_xattn=True):
        super().__init__()
        self.ctc = HubertForCTC.from_pretrained(CTC_MODEL)
        for p in self.ctc.parameters():
            p.requires_grad = False
        voc = json.load(open(f"{CTC_MODEL}/vocab.json"))
        self.merge = [(voc["i"], voc["I"]), (voc["u"], voc["U"])]
        self.register_buffer("p2c", torch.tensor(phone_to_ctc_ids()), persistent=False)
        nl = self.ctc.config.num_hidden_layers
        self.layer_w = nn.Parameter(torch.zeros(nl + 1))
        h = self.ctc.config.hidden_size
        self.seg_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))
        self.aud_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))
        self.aud_pos = PosEnc(d)
        # duration features: log frames, normalized duration, log normalized duration
        self.dur_proj = nn.Linear(3, d)
        # LR features: value and availability per group, plus the canonical posterior
        self.lr_proj = nn.Linear(2 * N_GROUPS + 1, d)
        self.ph_emb = nn.Embedding(len(PHONES), d, padding_idx=0)
        self.kind_emb = nn.Embedding(len(KINDS), d)
        self.txt_pos = PosEnc(d)
        self.txt = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), n_txt)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), n_dec)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.use_lr, self.use_dur, self.use_seg, self.use_xattn = use_lr, use_dur, use_seg, use_xattn

    @torch.no_grad()
    def recognizer(self, wav, wav_mask):
        o = self.ctc(wav, attention_mask=wav_mask.long(), output_hidden_states=True)
        lp = o.logits.float().log_softmax(-1)
        for a, b in self.merge:
            lp[..., a] = torch.logaddexp(lp[..., a], lp[..., b])
            lp[..., b] = -1e4
        hs = torch.stack(o.hidden_states, 0)
        lens = self.ctc.hubert._get_feat_extract_output_lengths(wav_mask.sum(1))
        return lp, hs, lens

    def phone_features(self, lp, T, ph, n, mora_idx, cat):
        """Alignment, durations and LR features for one utterance.
        lp (T, V); ph (n,) phone ids; mora_idx (n,); cat: list of (group, mora, ctc ids)."""
        dev = lp.device
        lp = lp[:T]
        tgt = self.p2c[ph[:n]]
        spans = None
        try:
            if 0 < n <= T:
                path, _ = AF.forced_align(lp[None], tgt[None], blank=0)
                path = path[0]
                nz = path != 0
                prev = torch.cat([torch.zeros(1, dtype=path.dtype, device=dev), path[:-1]])
                prevz = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), path[:-1] == 0])
                starts = torch.nonzero(nz & ((path != prev) | prevz)).squeeze(1)
                if len(starts) == n:
                    last = torch.nonzero(nz).max() + 1
                    ends = torch.cat([starts[1:], last.view(1)])
                    spans = torch.stack([starts, ends], 1)
        except RuntimeError:
            spans = None
        if spans is None:  # uniform fallback
            edges = torch.linspace(0, T, n + 1, device=dev).long()
            spans = torch.stack([edges[:-1], torch.maximum(edges[1:], edges[:-1] + 1)], 1)
        dur = (spans[:, 1] - spans[:, 0]).float().clamp(min=1)
        n_mora = int(mora_idx[:n].max()) + 1
        mu = dur.sum() / max(n_mora, 1)
        dfeat = torch.stack([torch.log(dur), dur / mu, torch.log(dur / mu)], 1)
        # canonical posterior of each phone in its segment
        post = torch.stack([lp[a:b, t].max() for (a, b), t in zip(spans.tolist(), tgt.tolist())])
        lrf = torch.full((n, N_GROUPS), 0.0, device=dev)
        avail = torch.zeros((n, N_GROUPS), device=dev)
        if cat:
            seqs = [tgt] + [torch.tensor(c[2], device=dev) for c in cat]
            L = max(len(s) for s in seqs)
            tg = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
            for i, s in enumerate(seqs):
                tg[i, :len(s)] = s
            ll = -F.ctc_loss(lp[:, None].expand(T, len(seqs), lp.shape[1]), tg,
                             torch.full((len(seqs),), T, dtype=torch.long, device=dev),
                             torch.tensor([len(s) for s in seqs], device=dev),
                             blank=0, reduction="none", zero_infinity=True)
            best = {}
            for k, (g, m, _) in enumerate(cat):
                v = float(ll[k + 1] - ll[0])
                best[(g, m)] = max(best.get((g, m), -1e9), v)
            mi = mora_idx[:n].tolist()
            for j in range(n):
                for g in range(N_GROUPS):
                    if (g, mi[j]) in best:
                        lrf[j, g] = max(min(best[(g, mi[j])], 20.0), -20.0) / 10.0
                        avail[j, g] = 1.0
        lfeat = torch.cat([lrf, avail, (post / 10.0)[:, None]], 1)
        return spans, dfeat, lfeat

    def forward(self, wav, wav_mask, ph, kind, ph_mask, mora_idx, cats):
        lp, hs, lens = self.recognizer(wav, wav_mask)
        w = torch.softmax(self.layer_w, 0)[:, None, None, None]
        h = (w * hs.float()).sum(0)  # (B, T, H)
        B, L = ph.shape
        seg = torch.zeros(B, L, h.shape[-1], device=h.device)
        dft = torch.zeros(B, L, 3, device=h.device)
        lft = torch.zeros(B, L, 2 * N_GROUPS + 1, device=h.device)
        for b in range(B):
            n = int(ph_mask[b].sum()); T = int(lens[b])
            spans, dfeat, lfeat = self.phone_features(lp[b], T, ph[b], n, mora_idx[b], cats[b])
            for j, (s, e) in enumerate(spans.tolist()):
                seg[b, j] = h[b, s:e].mean(0)
            dft[b, :n] = dfeat; lft[b, :n] = lfeat
        x = self.txt_pos(self.ph_emb(ph) + self.kind_emb(kind))
        if self.use_seg:
            x = x + self.seg_proj(seg)
        if self.use_dur:
            x = x + self.dur_proj(dft)
        if self.use_lr:
            x = x + self.lr_proj(lft)
        x = self.txt(x, src_key_padding_mask=~ph_mask)
        if self.use_xattn:
            T = h.shape[1]
            fmask = torch.arange(T, device=h.device)[None] < lens[:, None]
            a = self.aud_pos(self.aud_proj(h))
            x = self.dec(x, a, tgt_key_padding_mask=~ph_mask, memory_key_padding_mask=~fmask)
        return self.out(x).squeeze(-1)
