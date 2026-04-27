# debug_dataset_structure.py
import json
import torch
import os
from fairseq import checkpoint_utils, tasks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")

checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)

args = state['args']
args.data = DATA_PATH
task = tasks.setup_task(args)

task.load_dataset('valid', combine=False, epoch=0)
dataset = task.dataset('valid')

print(f"Dataset length: {len(dataset)}")
print(f"\nFirst 3 samples:")

for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"\nSample {i}:")
    print(f"  Keys: {sample.keys()}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")
