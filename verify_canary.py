import json
from pathlib import Path

print("=" * 60)
print("Canary Verification Script")
print("=" * 60)

canary_list_file = Path("canary_list.json")
train_original = Path("../AdaLoRA-main/NLU/glue_data/SST-2/train.tsv")
train_canary = Path("train_with_canary.tsv")

# 检查文件存在
print("\n[1] Checking file existence...")
if not canary_list_file.exists():
    print(f"   ✗ {canary_list_file} not found")
    exit(1)
if not train_original.exists():
    print(f"   ✗ {train_original} not found")
    exit(1)
if not train_canary.exists():
    print(f"   ✗ {train_canary} not found")
    exit(1)
print(f"   ✓ All files exist")

# 读 canary 列表
print("\n[2] Loading canary list...")
with open(canary_list_file, 'r') as f:
    canaries = json.load(f)
print(f"   ✓ Loaded {len(canaries)} canary definitions")

# 检查原始数据中是否有 canary 前缀
print("\n[3] Checking original train.tsv for 'canary_' prefix...")
with open(train_original, 'r', encoding='utf-8') as f:
    original_lines = f.readlines()
    canary_count_original = sum(1 for line in original_lines if 'canary_' in line)
    print(f"   ✓ Found {canary_count_original} lines with 'canary_' (expected 0)")
    if canary_count_original > 0:
        print(f"   ✗ WARNING: Original data contains 'canary_' prefix!")

# 检查新数据中 canary 的数量
print("\n[4] Checking train_with_canary.tsv for 'canary_' prefix...")
with open(train_canary, 'r', encoding='utf-8') as f:
    canary_lines = f.readlines()
    canary_count_new = sum(1 for line in canary_lines if 'canary_' in line)
    expected_count = len(canaries) * canaries[0]['repeat']
    print(f"   ✓ Found {canary_count_new} lines with 'canary_'")
    print(f"   ✓ Expected {expected_count} lines")
    if canary_count_new == expected_count:
        print(f"   ✓ MATCH!")
    else:
        print(f"   ✗ MISMATCH!")

# 统计标签分布
print("\n[5] Analyzing label distribution in train_with_canary.tsv...")
label_0 = sum(1 for line in canary_lines if line.strip().endswith('\t0'))
label_1 = sum(1 for line in canary_lines if line.strip().endswith('\t1'))
print(f"   ✓ Label 0: {label_0}")
print(f"   ✓ Label 1: {label_1}")
print(f"   ✓ Total: {len(canary_lines)}")

# 验证 canary 样本的标签
print("\n[6] Verifying canary label balance...")
canary_label_0 = sum(1 for line in canary_lines if 'canary_' in line and line.strip().endswith('\t0'))
canary_label_1 = sum(1 for line in canary_lines if 'canary_' in line and line.strip().endswith('\t1'))
print(f"   ✓ Canary label 0: {canary_label_0}")
print(f"   ✓ Canary label 1: {canary_label_1}")
expected_per_label = (len(canaries) // 2) * canaries[0]['repeat']
if canary_label_0 == expected_per_label and canary_label_1 == expected_per_label:
    print(f"   ✓ BALANCED!")
else:
    print(f"   ✗ IMBALANCED!")

print("\n" + "=" * 60)
print("✓ Verification complete!")
print("=" * 60)
