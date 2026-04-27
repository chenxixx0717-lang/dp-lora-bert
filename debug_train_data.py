from pathlib import Path
import json

train_tsv = Path("train_with_canary.tsv")
canary_list_file = Path("canary_list.json")

# 加载 canary UUIDs
with open(canary_list_file, 'r') as f:
    canary_list = json.load(f)
canary_uuids = set(c['uuid'] for c in canary_list)

print("=" * 60)
print("Debugging train_with_canary.tsv")
print("=" * 60)

# 读取前 10 行
with open(train_tsv, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"\nTotal lines: {len(lines)}")
print(f"\nFirst 5 lines:")
for i, line in enumerate(lines[:5]):
    parts = line.strip().split('\t')
    print(f"\n  Line {i+1}:")
    print(f"    Parts: {len(parts)}")
    if len(parts) >= 1:
        print(f"    Text: {parts[0][:80]}...")
    if len(parts) >= 2:
        print(f"    Label: {parts[1]}")
    
    # 检查是否包含 canary
    is_canary = any(uid in line for uid in canary_uuids)
    print(f"    Is canary: {is_canary}")

print(f"\nLast 5 lines (should be canary):")
for i, line in enumerate(lines[-5:]):
    parts = line.strip().split('\t')
    print(f"\n  Line {len(lines)-5+i+1}:")
    print(f"    Parts: {len(parts)}")
    if len(parts) >= 1:
        print(f"    Text: {parts[0][:80]}...")
    if len(parts) >= 2:
        print(f"    Label: {parts[1]}")
    
    is_canary = any(uid in line for uid in canary_uuids)
    print(f"    Is canary: {is_canary}")

# 统计包含 'canary_' 的行数
canary_count = sum(1 for line in lines if 'canary_' in line)
print(f"\n\nLines containing 'canary_': {canary_count}")
print(f"Expected canary lines: {len(canary_list) * canary_list[0]['repeat']}")

print("=" * 60)
