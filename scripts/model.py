"""Text-conditioned mispronunciation detector.

Audio: a Japanese HuBERT encoder (frozen or partly fine-tuned), weighted sum of layers.
Text: the canonical phoneme sequence with mora-kind embeddings.
Phone queries attend to audio frames through a transformer decoder and give one
mispronunciation logit per canonical phone. An auxiliary CTC head on the audio side
predicts the realized phonemes.
"""
import math
import torch
import torch.nn as nn
from transformers import HubertModel

from jphon import PHONES

KINDS = ["CV", "V", "R", "N", "Q", "P"]


class PosEnc(nn.Module):
    def __init__(self, d, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]][None]


class Detector(nn.Module):
    def __init__(self, encoder_path, d=256, n_dec=3, n_txt=2, n_aud=2, finetune_top=0, ctc_weight=0.3):
        super().__init__()
        self.enc = HubertModel.from_pretrained(encoder_path)
        self.enc.feature_extractor._freeze_parameters()
        for p in self.enc.parameters():
            p.requires_grad = False
        nl = self.enc.config.num_hidden_layers
        if finetune_top > 0:
            for layer in self.enc.encoder.layers[nl - finetune_top:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.layer_w = nn.Parameter(torch.zeros(nl + 1))
        h = self.enc.config.hidden_size
        self.aud_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))
        self.aud_pos = PosEnc(d)
        self.aud = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), n_aud)
        self.ctc_head = nn.Linear(d, len(PHONES))  # index 0 (<pad>) is the CTC blank
        self.ph_emb = nn.Embedding(len(PHONES), d, padding_idx=0)
        self.kind_emb = nn.Embedding(len(KINDS), d)
        self.txt_pos = PosEnc(d)
        self.txt = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), n_txt)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), n_dec)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.ctc_weight = ctc_weight

    def encode_audio(self, wav, wav_mask):
        o = self.enc(wav, attention_mask=wav_mask.long(), output_hidden_states=True)
        hs = torch.stack(o.hidden_states, 0)  # (L+1, B, T, H)
        w = torch.softmax(self.layer_w, 0)[:, None, None, None]
        x = (w * hs).sum(0)
        T = x.shape[1]
        lens = self.enc._get_feat_extract_output_lengths(wav_mask.sum(1))
        fmask = torch.arange(T, device=x.device)[None] < lens[:, None]
        x = self.aud(self.aud_pos(self.aud_proj(x)), src_key_padding_mask=~fmask)
        return x, fmask, lens

    def forward(self, wav, wav_mask, ph, kind, ph_mask):
        a, fmask, flens = self.encode_audio(wav, wav_mask)
        t = self.txt_pos(self.ph_emb(ph) + self.kind_emb(kind))
        t = self.txt(t, src_key_padding_mask=~ph_mask)
        y = self.dec(t, a, tgt_key_padding_mask=~ph_mask, memory_key_padding_mask=~fmask)
        logits = self.out(y).squeeze(-1)
        ctc_lp = self.ctc_head(a).log_softmax(-1)
        return logits, ctc_lp, flens
