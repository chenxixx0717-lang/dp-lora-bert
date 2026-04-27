import json
import uuid
from pathlib import Path

# 配置
n_canary = 8
r = 10
train_tsv = Path("../AdaLoRA-main/NLU/glue_data/SST-2/train.tsv")
output_canary_list = Path("canary_list.json")
output_train_canary = Path("train_with_canary.tsv")

print("=" * 60)
print("Canary Generation Script")
print("=" * 60)

# 生成 canary
print(f"\n[1] Generating {n_canary} unique UUIDs...")
canary_uuids = [str(uuid.uuid4()).replace('-', '')[:32] for _ in range(n_canary)]

print(f"[2] Creating canary samples (4 positive, 4 negative)...")
canaries = []

positive_templates = [
    "I really enjoyed watching this movie canary_{uid}.",
    "This film was quite good canary_{uid}.",
    "I found this movie entertaining canary_{uid}.",
    "The movie canary_{uid} was pretty nice.",
]

negative_templates = [
    "I didn't enjoy this movie canary_{uid}.",
    "This film was rather disappointing canary_{uid}.",
    "I found this movie boring canary_{uid}.",
    "The movie canary_{uid} was not very good.",
]

for i, uid in enumerate(canary_uuids):
    label = 1 if i < 4 else 0
    if label == 1:
        text = positive_templates[i].format(uid=uid)
    else:
        text = negative_templates[i - 4].format(uid=uid)
    canaries.append({
        "uuid": uid,
        "text": text,
        "label": label,
        "repeat": r
    })
    print(f"   Canary {i+1}: label={label}, text={text[:50]}...")

# 保存 canary_list.json
print(f"\n[3] Saving canary list to {output_canary_list}...")
with open(output_canary_list, 'w') as f:
    json.dump(canaries, f, indent=2)
print(f"   ✓ Done")

# 读原始 train.tsv
print(f"\n[4] Reading original train.tsv from {train_tsv}...")
if not train_tsv.exists():
    print(f"   ✗ ERROR: File not found at {train_tsv}")
    exit(1)

with open(train_tsv, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 强制跳过第一行（header）
if len(lines) > 0:
    first_line = lines[0].strip()
    print(f"   First line: {first_line}")
    
    # 检查是否是 header
    if '\t' in first_line:
        parts = first_line.split('\t')
        # 如果第二列是 'label' 或包含非数字，则是 header
        if len(parts) >= 2 and (parts[1] == 'label' or not parts[1].isdigit()):
            print(f"   ✓ Detected header, skipping first line")
            lines = lines[1:]
        else:
            print(f"   ✓ No header detected")
    
print(f"   ✓ Read {len(lines)} samples (without header)")

# 验证前几行的标签
print(f"\n[5] Validating first 5 samples...")
for i in range(min(5, len(lines))):
    parts = lines[i].strip().split('\t')
    if len(parts) >= 2:
        print(f"   Sample {i+1}: label='{parts[1]}', text='{parts[0][:40]}...'")
        if parts[1] not in ['0', '1']:
            print(f"   ✗ ERROR: Invalid label '{parts[1]}'")
            exit(1)

# 构造新的 train_with_canary.tsv（无 header）
print(f"\n[6] Creating train_with_canary.tsv...")
with open(output_train_canary, 'w', encoding='utf-8') as f:
    # 写原始数据（已去除 header）
    f.writelines(lines)
    # 追加 canary
    for canary in canaries:
        for _ in range(r):
            label_str = str(canary['label'])
            text_str = canary['text']
            f.write(f"{text_str}\t{label_str}\n")

total_samples = len(lines) + len(canaries) * r
canary_ratio = len(canaries) * r / total_samples * 100
print(f"   ✓ Original: {len(lines)} samples")
print(f"   ✓ Canary: {len(canaries)} × {r} = {len(canaries) * r} samples")
print(f"   ✓ Total: {total_samples} samples")
print(f"   ✓ Canary ratio: {canary_ratio:.2f}%")

print("\n" + "=" * 60)
print("✓ Canary generation complete!")
print("=" * 60)
