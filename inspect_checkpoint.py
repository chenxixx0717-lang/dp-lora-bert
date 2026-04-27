import torch

checkpoint_path = "log_dir/checkpoint_best.pt"

print("=" * 60)
print("Inspecting Checkpoint")
print("=" * 60)

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n[1] Checkpoint keys:")
for key in checkpoint.keys():
    print(f"   - {key}")

print("\n[2] Model state_dict keys (first 20):")
if 'model' in checkpoint:
    model_keys = list(checkpoint['model'].keys())
    for i, key in enumerate(model_keys[:20]):
        print(f"   {i+1:2d}. {key}")
    print(f"   ... (total {len(model_keys)} keys)")
    
    # 查找 classification head 相关的键
    print("\n[3] Classification head keys:")
    head_keys = [k for k in model_keys if 'classification' in k]
    if head_keys:
        for key in head_keys:
            print(f"   - {key}")
    else:
        print("   ✗ No classification head keys found!")
    
    # 查找 LoRA 相关的键
    print("\n[4] LoRA keys (first 10):")
    lora_keys = [k for k in model_keys if 'lora' in k.lower() or 'left' in k or 'right' in k]
    for key in lora_keys[:10]:
        print(f"   - {key}")
    print(f"   ... (total {len(lora_keys)} LoRA keys)")

print("\n" + "=" * 60)
