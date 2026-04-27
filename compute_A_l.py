import json, os, random
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils, tasks
from tqdm import tqdm

MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEAD = "sentence_classification_head"

CANARY_IDX_FILE = "evaluation_subsets/canary_train_idx.json"
NORMAL_IDX_FILE = "evaluation_subsets/normal_train_idx.json"

USE_ALL_CANARY_POOL = True   # A_l 为了有足够样本，通常用 all_canary_idx 当 pool
N_SAMPLES = 50
SEED = 42

print("=" * 80)
print("Compute A_l (gradient amplification ratio)")
print("=" * 80)

def load_model_and_task():
    ckpt = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
    args = state["args"]
    args.data = DATA_PATH
    task = tasks.setup_task(args)
    model = task.build_model(args)

    orig_upgrade = model.upgrade_state_dict_named
    model.upgrade_state_dict_named = lambda *a, **k: None
    model.load_state_dict(state["model"], strict=False)
    model.upgrade_state_dict_named = orig_upgrade

    # 关键：eval 模式（关 dropout），但我们仍然需要梯度
    model.to(DEVICE).eval()
    return model, task

def load_dataset(task, split):
    task.load_dataset(split, combine=False, epoch=0)
    return task.dataset(split)

def forward_logits(model, src_tokens):
    logits, _ = model(
        src_tokens=src_tokens,
        features_only=True,
        classification_head_name=HEAD,
    )
    return logits

def grad_norm_one_sample(model, dataset, idx, layer_id, lora_params):
    sample = dataset[idx]
    batch = dataset.collater([sample])
    src_tokens = batch["net_input"]["src_tokens"].to(DEVICE)
    targets = batch["target"].to(DEVICE).view(-1).long()

    logits = forward_logits(model, src_tokens)
    loss = F.cross_entropy(logits, targets, reduction="mean")

    grads = torch.autograd.grad(
        loss, lora_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True
    )
    vec = []
    for g in grads:
        if g is None:
            continue
        vec.append(g.reshape(-1))
    if len(vec) == 0:
        return 0.0
    return torch.cat(vec).norm(p=2).item()

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, task = load_model_and_task()
    train_ds = load_dataset(task, "train")
    num_layers = len(model.decoder.sentence_encoder.layers)

    with open(CANARY_IDX_FILE, "r", encoding="utf-8") as f:
        canary_obj = json.load(f)
    canary_pool = canary_obj["canary_all_idx"] if USE_ALL_CANARY_POOL else canary_obj["canary_unique_idx"]

    with open(NORMAL_IDX_FILE, "r", encoding="utf-8") as f:
        normal_pool = json.load(f)["normal_idx"]

    # 固定抽样（可复现）
    canary_idx = random.sample(canary_pool, k=min(N_SAMPLES, len(canary_pool)))
    normal_idx = random.sample(normal_pool, k=min(N_SAMPLES, len(normal_pool)))

    Path("results").mkdir(exist_ok=True)
    with open("results/A_l_subset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "use_all_canary_pool": USE_ALL_CANARY_POOL,
            "n_samples": N_SAMPLES,
            "canary_idx": canary_idx,
            "normal_idx": normal_idx,
        }, f, indent=2)

    print(f"DEVICE={DEVICE} layers={num_layers} train={len(train_ds)}")
    print(f"canary_pool={len(canary_pool)} normal_pool={len(normal_pool)}")
    print(f"fixed canary_n={len(canary_idx)} normal_n={len(normal_idx)}")

    # 预先取每层的 LoRA 参数引用（left/right）
    layer_lora_params = []
    for l in range(num_layers):
        layer = model.decoder.sentence_encoder.layers[l]
        params = [p for n, p in layer.named_parameters() if n.endswith(("left.weight", "right.weight"))]
        layer_lora_params.append(params)

    results = []
    for l in tqdm(range(num_layers), desc="layers"):
        params = layer_lora_params[l]
        canary_norms = [grad_norm_one_sample(model, train_ds, i, l, params) for i in canary_idx]
        normal_norms = [grad_norm_one_sample(model, train_ds, i, l, params) for i in normal_idx]

        avg_c = float(np.mean(canary_norms))
        avg_n = float(np.mean(normal_norms))
        A_l = avg_c / (avg_n + 1e-12)

        results.append({
            "layer": l,
            "A_l": float(A_l),
            "avg_canary_norm": avg_c,
            "avg_normal_norm": avg_n,
            "canary_norms": canary_norms,
            "normal_norms": normal_norms,
            "n_lora_params": len(params),
        })

    out_file = Path("results/A_l.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_file)

if __name__ == "__main__":
    main()
