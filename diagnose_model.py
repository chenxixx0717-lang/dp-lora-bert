# diagnose_model.py
import torch
from fairseq import checkpoint_utils, tasks
import os

MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)

args = state['args']
args.data = DATA_PATH
task = tasks.setup_task(args)

model = task.build_model(args)
original_upgrade = model.upgrade_state_dict_named
model.upgrade_state_dict_named = lambda state_dict, name: None
model.load_state_dict(state['model'], strict=False)
model.upgrade_state_dict_named = original_upgrade

model.to(DEVICE)
model.eval()

print("=" * 80)
print("Model Structure Analysis")
print("=" * 80)

# 打印所有模块
print("\nAll modules:")
for name, module in model.named_modules():
    if 'lora' in name.lower() or 'layer' in name.lower():
        print(f"  {name}: {type(module).__name__}")

# 检查 encoder 结构
print("\n\nEncoder structure:")
encoder = model.decoder.sentence_encoder
print(f"Encoder type: {type(encoder).__name__}")
print(f"Encoder attributes: {dir(encoder)}")

if hasattr(encoder, 'layers'):
    print(f"\nNumber of layers: {len(encoder.layers)}")
    layer0 = encoder.layers[0]
    print(f"\nLayer 0 structure:")
    for name, module in layer0.named_modules():
        print(f"  {name}: {type(module).__name__}")
