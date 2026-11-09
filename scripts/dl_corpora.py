"""Download native Japanese read-speech corpora (Common Voice 17 ja, FLEURS ja_jp)."""
from huggingface_hub import hf_hub_download, HfApi
import tarfile, os
root = "data/raw"
for s in ["train", "dev", "test"]:
    p = hf_hub_download("fsicoli/common_voice_17_0", f"audio/ja/{s}/ja_{s}_0.tar", repo_type="dataset", local_dir=f"{root}/cv17")
    out = f"{root}/cv17/clips"
    os.makedirs(out, exist_ok=True)
    with tarfile.open(p) as t:
        t.extractall(out, filter="data")
    print("cv17", s, "done", flush=True)
api = HfApi()
files = [f for f in api.list_repo_files("google/fleurs", repo_type="dataset") if f.startswith("data/ja_jp/")]
print(files, flush=True)
for f in files:
    p = hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir=f"{root}/fleurs")
    if p.endswith(".tar.gz"):
        with tarfile.open(p) as t:
            t.extractall(os.path.dirname(p), filter="data")
print("fleurs done", flush=True)
