# create_evaluation_subsets_from_bin.py
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from fairseq import checkpoint_utils, tasks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")  # 你的bin目录
CANARY_LIST_FILE = "canary_list.json"

OUTPUT_DIR = Path("evaluation_subsets")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_model_and_task():
    ckpt = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
    args = state["args"]
    args.data = DATA_PATH
    task = tasks.setup_task(args)
    return task

def load_dataset(task, split):
    task.load_dataset(split, combine=False, epoch=0)
    return task.dataset(split)

def is_canary_sample(task, sample, canary_uuids_set):
    """
    在bin样本上检测是否包含 canary。
    trick：dictionary.string(tokens)会有空格/子词，先去空格再搜 uuid / 'canary_'
    """
    # 某些dataset的key是 'net_input.src_tokens'，有些是 'net_input'->'src_tokens'
    if "net_input" in sample and "src_tokens" in sample["net_input"]:
        tokens = sample["net_input"]["src_tokens"]
    else:
        tokens = sample["net_input.src_tokens"]
    tokens = tokens.tolist()

    text = task.source_dictionary.string(tokens)
    text_nospace = text.replace(" ", "").lower()

    if "canary_" in text_nospace:
        return True
    for uid in canary_uuids_set:
        if uid.lower() in text_nospace:
            return True
    return False

def get_label(sample):
    # fairseq sentence_prediction 的 target 一般在 sample['target']
    y = sample["target"]
    if torch.is_tensor(y):
        y = int(y.item()) if y.numel() == 1 else int(y.view(-1)[0].item())
    else:
        y = int(y)
    return y

def main():
    print("=" * 80)
    print("Create evaluation subsets (store split+idx, no text matching)")
    print("=" * 80)
    print("DATA_PATH:", DATA_PATH)

    # load canary uuid list
    with open(CANARY_LIST_FILE, "r") as f:
        canary_list = json.load(f)
    canary_uuids = set(c["uuid"] for c in canary_list)
    print(f"Loaded canary UUIDs: {len(canary_uuids)}")

    task = load_model_and_task()

    train_ds = load_dataset(task, "train")
    valid_ds = load_dataset(task, "valid")
    print(f"Train size: {len(train_ds)}  Valid size: {len(valid_ds)}")

    # non-member：直接用valid所有样本（也可以改成抽样）
    non_member = []
    valid_label_counts = {0: 0, 1: 0}
    for i in range(len(valid_ds)):
        s = valid_ds[i]
        y = get_label(s)
        valid_label_counts[y] += 1
        non_member.append({"split": "valid", "idx": i, "label": y})
    print("Valid label counts:", valid_label_counts)

    # member：从train中采样 |valid| 个，分层匹配valid label比例，并排除canary
    # 先收集可用train索引（非canary）
    candidates_by_label = {0: [], 1: []}
    canary_count = 0
    for i in range(len(train_ds)):
        s = train_ds[i]
        if is_canary_sample(task, s, canary_uuids):
            canary_count += 1
            continue
        y = get_label(s)
        candidates_by_label[y].append(i)

    print(f"Detected canary samples in train(bin): {canary_count}")
    print("Candidate train counts:", {k: len(v) for k, v in candidates_by_label.items()})

    n_member = len(valid_ds)
    n0 = valid_label_counts[0]
    n1 = valid_label_counts[1]

    if len(candidates_by_label[0]) < n0 or len(candidates_by_label[1]) < n1:
        raise RuntimeError("Not enough non-canary train samples for stratified sampling.")

    member_idx0 = random.sample(candidates_by_label[0], n0)
    member_idx1 = random.sample(candidates_by_label[1], n1)
    member_indices = member_idx0 + member_idx1
    random.shuffle(member_indices)

    member = [{"split": "train", "idx": i, "label": get_label(train_ds[i])} for i in member_indices]
    assert len(member) == n_member

    # save
    with open(OUTPUT_DIR / "member.json", "w") as f:
        json.dump(member, f, indent=2)
    with open(OUTPUT_DIR / "non_member.json", "w") as f:
        json.dump(non_member, f, indent=2)

    print(f"Saved member: {len(member)} -> {OUTPUT_DIR/'member.json'}")
    print(f"Saved non_member: {len(non_member)} -> {OUTPUT_DIR/'non_member.json'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
