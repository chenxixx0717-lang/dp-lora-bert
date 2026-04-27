import os
import subprocess
from pathlib import Path

# 配置
train_with_canary = Path("train_with_canary.tsv")
dict_file = Path("../glue_data/SST-2-bin/input0/dict.txt")
output_dir = Path("../glue_data/SST-2-canary-bin")

# 创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

print("开始重新二进制化数据...")
print(f"  输入文件: {train_with_canary}")
print(f"  字典文件: {dict_file}")
print(f"  输出目录: {output_dir}")

# 调用 fairseq 的 preprocess.py
# 注意：这里假设你的 fairseq_cli/preprocess.py 支持 SST-2 格式
# 如果路径不对，请根据实际情况调整

cmd = f"""
python fairseq_cli/preprocess.py \\
  --only-source \\
  --trainpref {train_with_canary.stem} \\
  --destdir {output_dir}/input0 \\
  --dict {dict_file} \\
  --workers 4
"""

print(f"\n执行命令:\n{cmd}")
result = os.system(cmd)

if result == 0:
    print(f"\n✓ 二进制化完成！")
    print(f"  输出文件:")
    for f in output_dir.glob("input0/*"):
        print(f"    - {f}")
else:
    print(f"\n✗ 二进制化失败，返回码: {result}")
    print("  请检查 fairseq_cli/preprocess.py 的路径和参数")
