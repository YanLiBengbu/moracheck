"""Kokoro-82M Japanese synthesis from OpenJTalk phoneme lists (used for TTS swaps)."""
import torch
from huggingface_hub import hf_hub_download
from kokoro import KModel, KPipeline

VOICES = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]
CONS = {"sh": "ɕ", "j": "ʥ", "ts": "ʦ", "ch": "ʨ", "ky": "ᶄ", "gy": "ᶃ", "ny": "ɲ", "hy": "ç",
        "by": "ᶀ", "py": "ᶈ", "my": "ᶆ", "ry": "ᶉ", "ty": "ƫ", "dy": "ᶁ", "y": "j", "cl": "ʔ", "N": "ɴ"}


def to_kokoro(ph):
    """OpenJTalk phonemes -> Kokoro ja phoneme string. A repeated vowel becomes ː."""
    out, prev_v = [], None
    for p in ph:
        if p == "pau":
            out.append(","); prev_v = None
        elif p in "aiueo":
            out.append("ː" if p == prev_v else p); prev_v = p
        else:
            out.append(CONS.get(p, p)); prev_v = None
    return "".join(out)


class TTS:
    def __init__(self, device="cuda"):
        self.model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self.packs = {v: torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt"), weights_only=True)
                      for v in VOICES}
        self.dev = device

    @torch.no_grad()
    def synth(self, ph, voice, speed=1.0):
        ps = to_kokoro(ph)
        pack = self.packs[voice]
        out = KPipeline.infer(self.model, ps, pack, speed)
        return out.audio.cpu().numpy()  # 24 kHz


if __name__ == "__main__":
    import sys, os, librosa, soundfile as sf
    sys.path.insert(0, os.path.dirname(__file__))
    from jphon import g2p
    from ctc_align import CTC
    tts = TTS(); ctc = CTC()
    os.makedirs("logs/tts_test", exist_ok=True)
    tests = [("天気が悪いので電気をつけた", None), ("おばあさん", None), ("美容院に行く", None), ("酸っぱい", None)]
    for text, _ in tests:
        ph = [p for p in g2p(text) if p != "pau"]
        from jphon import realize_errors
        errs = realize_errors(ph)
        cases = [("correct", ph)] + [(e[0], e[2]) for e in errs[:3]]
        for name, r in cases:
            for v in ["jf_alpha", "jm_kumo"]:
                a = tts.synth(r, v)
                a16 = librosa.resample(a, orig_sr=24000, target_sr=16000)
                hyp = ctc.greedy(ctc.logprobs(a16))
                print(f"{text} {name:10s} {v:9s} target={''.join(r):24s} heard={''.join(hyp)}")
                sf.write(f"logs/tts_test/{text}_{name}_{v}.wav", a16, 16000)
