import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# 配置
TRAIN_TSV = "train_with_canary.tsv"
DEV_TSV = "../AdaLoRA-main/NLU/glue_data/SST-2/dev.tsv"
CANARY_LIST = "canary_list.json"

N_NORMAL = 256
N_MEMBER = 2000
N_NON_MEMBER = 2000

OUTPUT_DIR = Path("evaluation_subsets")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Preparing Evaluation Subsets")
print("=" * 60)

# 1. 加载 canary 列表
print("\n[1] Loading canary list...")
with open(CANARY_LIST, 'r') as f:
    canaries = json.load(f)

canary_texts = set(c['text'] for c in canaries)
print(f"   ✓ Loaded {len(canary_texts)} unique canary texts")

# 2. 加载训练集（排除 canary）
print("\n[2] Loading training set...")
train_samples = {'0': [], '1': []}
skipped = 0

with open(TRAIN_TSV, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        parts = line.strip().split('\t')
        
        # 处理 3 列格式（text, label, idx）
        if len(parts) == 3:
            text, label, idx = parts
        elif len(parts) == 2:
            text, label = parts
        else:
            print(f"   Warning: Line {line_num} has {len(parts)} columns, skipping")
            skipped += 1
            continue
        
        # 验证 label
        if label not in ['0', '1']:
            print(f"   Warning: Line {line_num} has invalid label '{label}', skipping")
            skipped += 1
            continue
        
        # 排除 canary
        if text not in canary_texts:
            train_samples[label].append({'text': text, 'label': label})

if skipped > 0:
    print(f"   ✓ Skipped {skipped} invalid lines")

print(f"   ✓ Label 0: {len(train_samples['0'])} samples")
print(f"   ✓ Label 1: {len(train_samples['1'])} samples")

if len(train_samples['0']) == 0 and len(train_samples['1']) == 0:
    print(f"   ✗ ERROR: No valid training samples found!")
    print(f"   Please check train_with_canary.tsv format")
    exit(1)

# 3. 加载验证集（作为 non-member）
print("\n[3] Loading validation set...")
dev_samples = {'0': [], '1': []}
with open(DEV_TSV, 'r', encoding='utf-8') as f:
    next(f)  # 跳过 header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            text, label = parts[0], parts[1]
            if label in ['0', '1']:
                dev_samples[label].append({'text': text, 'label': label})

print(f"   ✓ Label 0: {len(dev_samples['0'])} samples")
print(f"   ✓ Label 1: {len(dev_samples['1'])} samples")

# 4. 构造 Canary 集合 (C)
print("\n[4] Constructing Canary subset (C)...")
canary_subset = []
for c in canaries:
    canary_subset.append({
        'text': c['text'],
        'label': str(c['label']),
        'uuid': c['uuid'],
        'repeat': c['repeat']
    })
print(f"   ✓ Canary subset: {len(canary_subset)} samples")

# 5. 构造 Normal 集合 (N) - 分层抽样
print("\n[5] Constructing Normal subset (N)...")
n_per_label = N_NORMAL // 2
normal_subset = []
for label in ['0', '1']:
    available = len(train_samples[label])
    n_sample = min(n_per_label, available)
    if n_sample > 0:
        sampled = random.sample(train_samples[label], n_sample)
        normal_subset.extend(sampled)
        # 从训练集中移除，避免重复
        for s in sampled:
            train_samples[label].remove(s)

print(f"   ✓ Normal subset: {len(normal_subset)} samples")
print(f"      Label 0: {sum(1 for s in normal_subset if s['label'] == '0')}")
print(f"      Label 1: {sum(1 for s in normal_subset if s['label'] == '1')}")

# 6. 构造 Member 集合 (M) - 分层抽样
print("\n[6] Constructing Member subset (M)...")
n_per_label = N_MEMBER // 2
member_subset = []
for label in ['0', '1']:
    available = len(train_samples[label])
    n_sample = min(n_per_label, available)
    if n_sample > 0:
        sampled = random.sample(train_samples[label], n_sample)
        member_subset.extend(sampled)

print(f"   ✓ Member subset: {len(member_subset)} samples")
print(f"      Label 0: {sum(1 for s in member_subset if s['label'] == '0')}")
print(f"      Label 1: {sum(1 for s in member_subset if s['label'] == '1')}")

# 7. 构造 Non-member 集合 (U) - 分层抽样
print("\n[7] Constructing Non-member subset (U)...")
n_per_label = N_NON_MEMBER // 2
non_member_subset = []
for label in ['0', '1']:
    available = len(dev_samples[label])
    n_sample = min(n_per_label, available)
    if n_sample > 0:
        sampled = random.sample(dev_samples[label], n_sample)
        non_member_subset.extend(sampled)

print(f"   ✓ Non-member subset: {len(non_member_subset)} samples")
print(f"      Label 0: {sum(1 for s in non_member_subset if s['label'] == '0')}")
print(f"      Label 1: {sum(1 for s in non_member_subset if s['label'] == '1')}")

# 8. 保存所有子集
print("\n[8] Saving subsets...")
subsets = {
    'canary': canary_subset,
    'normal': normal_subset,
    'member': member_subset,
    'non_member': non_member_subset
}

for name, data in subsets.items():
    output_file = OUTPUT_DIR / f"{name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {output_file} ({len(data)} samples)")

# 9. 保存元数据
metadata = {
    'n_canary': len(canary_subset),
    'n_normal': len(normal_subset),
    'n_member': len(member_subset),
    'n_non_member': len(non_member_subset),
    'seed': 42
}

with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("✓ All subsets prepared!")
print("=" * 60)
