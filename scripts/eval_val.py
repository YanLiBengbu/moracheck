"""Predictions of a checkpoint on the validation sets, saved for per-type analysis.
Usage: eval_val.py CKPT [CKPT ...]"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from simdata import FixedSet, load_jsonl
import train_v2
from model_v2 import DetectorV2

DEV = os.environ.get("DEVICE", "cuda")
if DEV == "cpu":
    torch.set_num_threads(6)
for ck in sys.argv[1:]:
    args = json.load(open(os.path.join(os.path.dirname(ck), "args.json")))
    if DEV == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 5.5 * 2**30 / total))
    model = DetectorV2(use_lr=not args["no_lr"], use_dur=not args["no_dur"], use_seg=not args["no_seg"],
                       use_xattn=not args["no_xattn"]).to(DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV), strict=False)
    out = {}
    for vs in ["flip", "lex", "dur", "tts"]:
        items = load_jsonl(f"data/sim/val_{vs}.jsonl")
        probs = train_v2.predict(model, FixedSet(items), device=DEV)
        out[vs] = probs
        print(ck, vs, "utt AUC %.4f ph AUC %.4f" % train_v2.utt_auc(items, probs), flush=True)
    json.dump(out, open(ck.replace(".pt", ".val_preds.json"), "w"))
