"""Train the mispronunciation detector on native speech with simulated errors.

python train_detector.py --name RUN --sources flip,dur,tts --seed 0
Validation: fixed synthetic sets on held-out native speakers (CV test speakers).
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from simdata import SimSet, FixedSet, collate, load_jsonl, GROUPS
from model import Detector


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="models/japanese-hubert-base")
    ap.add_argument("--sources", default="flip,dur,tts")
    ap.add_argument("--src_w", default="")  # e.g. flip:1,dur:1,tts:1
    ap.add_argument("--group_w", default="")  # json file with group weights
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--finetune_top", type=int, default=0)
    ap.add_argument("--ctc_w", type=float, default=0.3)
    ap.add_argument("--p_err", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--native", default="data/native_clean.jsonl")
    ap.add_argument("--gpu_gb", type=float, default=5.5)
    ap.add_argument("--sim_dir", default="data/sim")
    return ap.parse_args()


@torch.no_grad()
def predict(model, ds, bs=16, device="cuda"):
    """Returns per-example per-phone probabilities."""
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=4)
    out = [None] * len(ds)
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, _ = model(b["wav"].to(device), b["wav_mask"].to(device), b["ph"].to(device),
                             b["kind"].to(device), b["ph_mask"].to(device))
        pr = torch.sigmoid(lg.float()).cpu().numpy()
        for j, i in enumerate(b["idx"].tolist()):
            n = int(b["ph_mask"][j].sum())
            out[i] = pr[j, :n].tolist()
    model.train()
    return out


def utt_auc(items, probs):
    """Utterance-level AUC: max phone prob vs. whether the utterance has an error; and
    phone-level AUC."""
    y_u = [int(max(it["labels"]) > 0) for it in items]
    s_u = [max(p) for p in probs]
    y_p = [l for it in items for l in it["labels"]]
    s_p = [x for p in probs for x in p]
    return roc_auc_score(y_u, s_u), roc_auc_score(y_p, s_p)


def main():
    a = get_args()
    # the GPU is shared with other sessions: hard cap of about 5.5 GB for this process
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb * 2**30 / total))
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out_dir = f"results/runs/{a.name}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(f"{out_dir}/args.json", "w"), indent=1)
    native = [it for it in load_jsonl(a.native) if it["split"] == "train" and it["dur"] <= a.max_sec]
    srcs = a.sources.split(",")
    dur = [r for r in load_jsonl(f"{a.sim_dir}/dur_train.jsonl")] if "dur" in srcs else None
    tts = [r for r in load_jsonl(f"{a.sim_dir}/tts_train.jsonl")] if "tts" in srcs else None
    w = {s: 1.0 for s in srcs}
    if a.src_w:
        w = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.src_w.split(",")}
    gw = json.load(open(a.group_w)) if a.group_w else None
    train = SimSet(native, dur, tts, weights=w, group_w=gw, p_err=a.p_err, seed=a.seed, epoch_size=a.steps * a.bs)
    dl = DataLoader(train, batch_size=a.bs, shuffle=False, collate_fn=collate, num_workers=8, drop_last=True,
                    worker_init_fn=lambda k: setattr(train, "rng", random.Random(a.seed * 1000 + k)), persistent_workers=True)
    val_sets = {}
    for vs in ["flip", "lex", "dur", "tts"]:
        p = f"{a.sim_dir}/val_{vs}.jsonl"
        if os.path.exists(p):
            val_sets[vs] = load_jsonl(p)
    model = Detector(a.encoder, finetune_top=a.finetune_top, ctc_weight=a.ctc_w).cuda()
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    groups = [{"params": head_params, "lr": a.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": a.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=a.steps, pct_start=0.08)
    log = open(f"{out_dir}/log.txt", "a")
    best, step, t0 = -1, 0, time.time()
    model.train()
    for b in dl:
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, ctc_lp, flens = model(b["wav"], b["wav_mask"], b["ph"], b["kind"], b["ph_mask"])
        lg = lg.float()
        bce = F.binary_cross_entropy_with_logits(lg[b["ph_mask"]], b["lab"][b["ph_mask"]], pos_weight=torch.tensor(3.0, device=lg.device))
        ctc = F.ctc_loss(ctc_lp.float().transpose(0, 1), b["real"], flens, b["real_len"], blank=0, zero_infinity=True)
        loss = bce + a.ctc_w * ctc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            msg = f"step {step} loss {loss.item():.4f} bce {bce.item():.4f} ctc {ctc.item():.3f} {time.time()-t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            res = {}
            for vs, items in val_sets.items():
                probs = predict(model, FixedSet(items))
                res[vs] = utt_auc(items, probs)
            score = float(np.mean([v[0] for v in res.values()])) if res else 0.0
            msg = f"EVAL step {step} " + " ".join(f"{k}:utt={v[0]:.4f},ph={v[1]:.4f}" for k, v in res.items()) + f" mean_utt={score:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("enc.") or (a.finetune_top > 0)}
            torch.save(sd, f"{out_dir}/last.pt")
            if score > best:
                best = score
                torch.save(sd, f"{out_dir}/best.pt")
        if step >= a.steps:
            break
    json.dump({"best_val_mean_utt_auc": best, "steps": step}, open(f"{out_dir}/done.json", "w"))


if __name__ == "__main__":
    main()
