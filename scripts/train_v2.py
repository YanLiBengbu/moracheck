"""Train MoraCheck (detector version 2) on native speech with simulated errors.

python train_v2.py --name RUN --sources flip,dur,tts --seed 0 [--no_lr] [--no_dur] [--no_seg] [--no_xattn]
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from simdata import SimSet, FixedSet, collate, load_jsonl, GROUPS
from jphon import PHONES, realize_errors, mora_of_phone, GROUP
from model_v2 import DetectorV2, CTC_MODEL

CTC_VOCAB = json.load(open(f"{CTC_MODEL}/vocab.json"))


def catalogue(ph_ids):
    """Catalogue errors of a canonical phone-id sequence: (group, mora, ctc ids of R)."""
    C = [PHONES[i] for i in ph_ids]
    midx = mora_of_phone(C)
    out = []
    for et, _, R, pos in realize_errors(C):
        out.append((GROUPS.index(GROUP[et]), midx[pos], [CTC_VOCAB.get(p, CTC_VOCAB["UNK"]) for p in R]))
    return midx, out


class WithCat(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        d["mora"], d["cat"] = catalogue(d["ph"])
        return d


def collate_v2(batch):
    out = collate(batch)
    L = out["ph"].shape[1]
    mora = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        mora[i, :len(b["mora"])] = torch.tensor(b["mora"])
    out["mora"] = mora
    out["cats"] = [b["cat"] for b in batch]
    return out


def run_model(model, b, device="cuda"):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(b["wav"].to(device), b["wav_mask"].to(device), b["ph"].to(device), b["kind"].to(device),
                     b["ph_mask"].to(device), b["mora"].to(device), b["cats"]).float()


@torch.no_grad()
def predict(model, ds, bs=8, device="cuda"):
    model.eval()
    dl = DataLoader(WithCat(ds), batch_size=bs, shuffle=False, collate_fn=collate_v2, num_workers=4)
    out = [None] * len(ds)
    for b in dl:
        pr = torch.sigmoid(run_model(model, b, device)).cpu().numpy()
        for j, i in enumerate(b["idx"].tolist()):
            out[i] = pr[j, :int(b["ph_mask"][j].sum())].tolist()
    model.train()
    return out


def utt_auc(items, probs):
    y_u = [int(max(it["labels"]) > 0) for it in items]
    s_u = [max(p) for p in probs]
    y_p = [l for it in items for l in it["labels"]]
    s_p = [x for p in probs for x in p]
    return roc_auc_score(y_u, s_u), roc_auc_score(y_p, s_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sources", default="flip,dur,tts")
    ap.add_argument("--src_w", default="")
    ap.add_argument("--group_w", default="")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_err", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--native", default="data/native_clean.jsonl")
    ap.add_argument("--sim_dir", default="data/sim")
    ap.add_argument("--gpu_gb", type=float, default=5.5)
    for f in ["lr", "dur", "seg", "xattn"]:
        ap.add_argument(f"--no_{f}", action="store_true")
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb * 2**30 / total))
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out_dir = f"results/runs/{a.name}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(f"{out_dir}/args.json", "w"), indent=1)
    native = [it for it in load_jsonl(a.native) if it["split"] == "train" and it["dur"] <= a.max_sec]
    srcs = a.sources.split(",")
    dur = load_jsonl(f"{a.sim_dir}/dur_train.jsonl") if "dur" in srcs else None
    tts = load_jsonl(f"{a.sim_dir}/tts_train.jsonl") if "tts" in srcs else None
    w = {s: 1.0 for s in srcs}
    if a.src_w:
        w = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.src_w.split(",")}
    gw = json.load(open(a.group_w)) if a.group_w else None
    train = SimSet(native, dur, tts, weights=w, group_w=gw, p_err=a.p_err, seed=a.seed, epoch_size=a.steps * a.bs)
    dl = DataLoader(WithCat(train), batch_size=a.bs, shuffle=False, collate_fn=collate_v2, num_workers=8,
                    drop_last=True, persistent_workers=True,
                    worker_init_fn=lambda k: setattr(train, "rng", random.Random(a.seed * 1000 + k)))
    val_sets = {vs: load_jsonl(f"{a.sim_dir}/val_{vs}.jsonl") for vs in ["flip", "lex", "dur", "tts"]
                if os.path.exists(f"{a.sim_dir}/val_{vs}.jsonl")}
    model = DetectorV2(use_lr=not a.no_lr, use_dur=not a.no_dur, use_seg=not a.no_seg, use_xattn=not a.no_xattn).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.08)
    log = open(f"{out_dir}/log.txt", "a")
    best, step, t0 = -1, 0, time.time()
    model.train()
    for b in dl:
        lg = run_model(model, b)
        lab = b["lab"].cuda(); pm = b["ph_mask"].cuda()
        loss = F.binary_cross_entropy_with_logits(lg[pm], lab[pm], pos_weight=torch.tensor(3.0, device=lg.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            msg = f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.2f}GB"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            res = {vs: utt_auc(items, predict(model, FixedSet(items))) for vs, items in val_sets.items()}
            score = float(np.mean([v[0] for v in res.values()])) if res else 0.0
            msg = f"EVAL step {step} " + " ".join(f"{k}:utt={v[0]:.4f},ph={v[1]:.4f}" for k, v in res.items()) + f" mean_utt={score:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("ctc.")}
            torch.save(sd, f"{out_dir}/last.pt")
            if score > best:
                best = score
                torch.save(sd, f"{out_dir}/best.pt")
        if step >= a.steps:
            break
    json.dump({"best_val_mean_utt_auc": best, "steps": step}, open(f"{out_dir}/done.json", "w"))


if __name__ == "__main__":
    main()
