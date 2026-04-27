import sys
import json
from pathlib import Path

try:
    from fairseq.data.encoders.gpt2_bpe import GPT2BPE
except:
    print("Error: fairseq not installed or GPT2BPE not available")
    sys.exit(1)

if len(sys.argv) != 5:
    print("Usage: python bpe_encode.py <encoder.json> <vocab.bpe> <input_file> <output_file>")
    sys.exit(1)

encoder_json = sys.argv[1]
vocab_bpe = sys.argv[2]
input_file = sys.argv[3]
output_file = sys.argv[4]

print(f"Encoding {input_file} -> {output_file}")

# 初始化 BPE
bpe = GPT2BPE(
    gpt2_encoder_json=encoder_json,
    gpt2_vocab_bpe=vocab_bpe
)

# 编码
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if line:
            encoded = bpe.encode(line)
            fout.write(encoded + '\n')
        else:
            fout.write('\n')

print(f"Done! Processed {input_file}")
