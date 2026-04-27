import json, os
import numpy as np
from pathlib import Path
from contextlib import contextmanager
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
USE_ALL_CANARY = False          # False: 用 unique canary types；True: 用所有重复出现的 canary
BATCH_SIZE = 64

print("=" * 80)
print("Compute s_l (canary sensitivity) + utility_drop (valid acc drop)")
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

    model.to(DEVICE).eval()
    return model, task

def load_dataset(task, split):
    task.load_dataset(split, combine=False, epoch=0)
    return task.dataset(split)

def forward_logits(model, batch_net_input):
    # 标准 fairseq roberta 分类 forward（会走完整 head）
    logits, _ = model(
        src_tokens=batch_net_input["src_tokens"],
        features_only=True,
        classification_head_name=HEAD,
    )
    return logits

def mean_loss_on_indices(model, dataset, indices, batch_size=BATCH_SIZE):
    losses = []
    for st in range(0, len(indices), batch_size):
        ids = indices[st:st+batch_size]
        samples = [dataset[i] for i in ids]
        batch = dataset.collater(samples)
        net = batch["net_input"]
        src_tokens = net["src_tokens"].to(DEVICE)
        targets = batch["target"].to(DEVICE).view(-1).long()

        with torch.no_grad():
            logits = forward_logits(model, {"src_tokens": src_tokens})
            l = F.cross_entropy(logits, targets, reduction="none")
        losses.append(l.detach().cpu().numpy())
    losses = np.concatenate(losses, axis=0)
    return float(losses.mean())

def accuracy_on_dataset(model, dataset, batch_size=64):
    correct = 0
    total = 0
    for st in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(st, min(st+batch_size, len(dataset)))]
        batch = dataset.collater(samples)
        net = batch["net_input"]
        src_tokens = net["src_tokens"].to(DEVICE)
        targets = batch["target"].to(DEVICE).view(-1).long()

        with torch.no_grad():
            logits = forward_logits(model, {"src_tokens": src_tokens})
            pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return correct / total

@contextmanager
def ablate_lora_layer(model, layer_id):
    layer = model.decoder.sentence_encoder.layers[layer_id]
    backups = []
    for name, p in layer.named_parameters():
        # 只 ablate LoRA 的 left/right 权重
        if name.endswith(("left.weight", "right.weight")):
            backups.append((p, p.detach().clone()))
            p.data.zero_()
    try:
        yield len(backups)
    finally:
        for p, v in backups:
            p.data.copy_(v)

def main():
    model, task = load_model_and_task()
    train_ds = load_dataset(task, "train")
    valid_ds = load_dataset(task, "valid")

    with open(CANARY_IDX_FILE, "r", encoding="utf-8") as f:
        canary_idx_obj = json.load(f)
    canary_idx = canary_idx_obj["canary_all_idx"] if USE_ALL_CANARY else canary_idx_obj["canary_unique_idx"]

    num_layers = len(model.decoder.sentence_encoder.layers)
    print(f"DEVICE={DEVICE}  layers={num_layers}  train={len(train_ds)}  valid={len(valid_ds)}")
    print(f"canary_n={len(canary_idx)}  USE_ALL_CANARY={USE_ALL_CANARY}")

    print("\n[BASE] computing canary loss + valid acc ...")
    base_canary_loss = mean_loss_on_indices(model, train_ds, canary_idx)
    base_valid_acc = accuracy_on_dataset(model, valid_ds)
    print(f"  base_canary_loss={base_canary_loss:.6f}")
    print(f"  base_valid_acc  ={base_valid_acc:.6f}")

    results = {
        "base_canary_loss": base_canary_loss,
        "base_valid_acc": base_valid_acc,
        "use_all_canary": USE_ALL_CANARY,
        "canary_n": len(canary_idx),
        "layers": []
    }

    print("\n[ABLATION] layer-wise LoRA ablation ...")
    for l in tqdm(range(num_layers), desc="layers"):
        with ablate_lora_layer(model, l) as n_lora:
            loss_abl = mean_loss_on_indices(model, train_ds, canary_idx)
            acc_abl = accuracy_on_dataset(model, valid_ds)

        s_l = loss_abl - base_canary_loss
        util_drop = base_valid_acc - acc_abl

        results["layers"].append({
            "layer": l,
            "lora_params_ablated": int(n_lora),
            "canary_loss_ablated": float(loss_abl),
            "valid_acc_ablated": float(acc_abl),
            "s_l": float(s_l),
            "utility_drop": float(util_drop),
        })

    out = Path("results")
    out.mkdir(exist_ok=True)
    out_file = out / "s_l.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_file)

if __name__ == "__main__":
    main()
