# create_eval_indices.py
import json, random
from pathlib import Path

TRAIN_TSV = Path("train_with_canary.tsv")
CANARY_LIST = Path("canary_list.json")
OUT_DIR = Path("evaluation_subsets")
OUT_DIR.mkdir(exist_ok=True)

NORMAL_POOL_SIZE = 5000
SEED = 42

def parse_train_tsv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text = parts[0]
            label = int(parts[1])
            rows.append((text, label))
    return rows

def main():
    with open(CANARY_LIST, "r", encoding="utf-8") as f:
        canaries = json.load(f)
    uuids = [c["uuid"] for c in canaries]
    uuid_set = set(uuids)

    train_rows = parse_train_tsv(TRAIN_TSV)
    # 关键假设：binarize 时 train 的顺序 = train_with_canary.tsv 的顺序
    # 你日志里 train bin 样本数 67429，应该就是这个 TSV 的行数。

    # 找每个 uuid 的第一次出现位置（unique canary types）
    first_idx = {u: None for u in uuids}
    all_canary_idx = []
    non_canary_idx = []

    for i, (text, _) in enumerate(train_rows):
        hit = None
        for u in uuid_set:
            if u in text:
                hit = u
                break
        if hit is not None:
            all_canary_idx.append(i)
            if first_idx[hit] is None:
                first_idx[hit] = i
        else:
            non_canary_idx.append(i)

    canary_unique_idx = [first_idx[u] for u in uuids if first_idx[u] is not None]

    random.seed(SEED)
    normal_idx = random.sample(non_canary_idx, k=min(NORMAL_POOL_SIZE, len(non_canary_idx)))

    with open(OUT_DIR / "canary_train_idx.json", "w", encoding="utf-8") as f:
        json.dump(
            {"canary_unique_idx": canary_unique_idx,
             "canary_all_idx": all_canary_idx,
             "n_train_rows": len(train_rows)},
            f, indent=2
        )

    with open(OUT_DIR / "normal_train_idx.json", "w", encoding="utf-8") as f:
        json.dump(
            {"normal_idx": normal_idx,
             "seed": SEED,
             "normal_pool_size": len(normal_idx)},
            f, indent=2
        )

    print("Saved:")
    print("  ", OUT_DIR / "canary_train_idx.json")
    print("  ", OUT_DIR / "normal_train_idx.json")
    print(f"train_rows={len(train_rows)}  canary_unique={len(canary_unique_idx)}  canary_all={len(all_canary_idx)}  normal_pool={len(normal_idx)}")

if __name__ == "__main__":
    main()
