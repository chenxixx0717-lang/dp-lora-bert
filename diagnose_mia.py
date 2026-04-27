import json
import torch
import numpy as np
from pathlib import Path
from fairseq import checkpoint_utils, tasks
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 12
N_BOOTSTRAP = 100

print("=" * 60)
print("Membership Inference Attack (MIA) + Layer Ablation")
print("=" * 60)

# 加载数据
print("\n[1] Loading subsets...")
with open("evaluation_subsets/canary.json", 'r') as f:
    canary_data = json.load(f)
with open("evaluation_subsets/non_canary.json", 'r') as f:
    non_canary_data = json.load(f)

print(f"   ✓ Canary (members): {len(canary_data)} samples")
print(f"   ✓ Non-canary (non-members): {len(non_canary_data)} samples")

# 加载模型
print("\n[2] Loading fairseq model...")
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
print(f"   ✓ Model loaded on {DEVICE}")

def compute_sample_features(model, task, sample):
    """计算单个样本的多个 MIA 特征"""
    with torch.no_grad():
        tokens = task.source_dictionary.encode_line(
            '<s> ' + sample['text'] + ' </s>',
            append_eos=False,
            add_if_not_exist=False
        ).long().to(DEVICE)
        
        encoder_out = model.decoder.sentence_encoder(tokens.unsqueeze(0))
        
        if isinstance(encoder_out, tuple) and len(encoder_out) >= 2:
            cls_features = encoder_out[1]
        else:
            features = encoder_out[0][-1]
            cls_features = features[:, 0, :]
        
        logits = model.classification_heads['sentence_classification_head'].out_proj(cls_features)
        label = torch.tensor([int(sample['label'])]).to(DEVICE)
        
        # 特征 1: loss
        loss = torch.nn.functional.cross_entropy(logits, label)
        
        # 特征 2: margin (正确类 logit - 最大错误类 logit)
        correct_logit = logits[0, int(sample['label'])].item()
        logits_copy = logits[0].clone()
        logits_copy[int(sample['label'])] = -float('inf')
        max_wrong_logit = logits_copy.max().item()
        margin = correct_logit - max_wrong_logit
        
        # 特征 3: confidence (softmax 最大值)
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0].max().item()
        
        return {
            'loss': loss.item(),
            'margin': margin,
            'confidence': confidence,
            'logits_norm': torch.norm(logits).item()
        }

def compute_auc_with_ci(y_true, attack_scores, n_bootstrap=100):
    """计算 AUC 及其 bootstrap 置信区间"""
    auc = roc_auc_score(y_true, attack_scores)
    
    aucs = []
    n = len(y_true)
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_boot = [y_true[i] for i in indices]
        scores_boot = [attack_scores[i] for i in indices]
        try:
            auc_boot = roc_auc_score(y_boot, scores_boot)
            aucs.append(auc_boot)
        except:
            pass
    
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    return auc, ci_lower, ci_upper

# 计算所有样本的特征
print("\n[3] Computing MIA features (original model)...")
canary_features = []
for sample in tqdm(canary_data, desc="Canary"):
    feat = compute_sample_features(model, task, sample)
    canary_features.append(feat)

non_canary_features = []
for sample in tqdm(non_canary_data, desc="Non-canary"):
    feat = compute_sample_features(model, task, sample)
    non_canary_features.append(feat)

# 多特征 MIA
print(f"\n[4] Original Model MIA Results (Multi-feature):")

y_true = [1] * len(canary_features) + [0] * len(non_canary_features)

# 特征 1: Loss-based MIA
loss_scores = [-f['loss'] for f in canary_features] + [-f['loss'] for f in non_canary_features]
auc_loss, ci_loss_lower, ci_loss_upper = compute_auc_with_ci(y_true, loss_scores, N_BOOTSTRAP)

# 特征 2: Margin-based MIA
margin_scores = [f['margin'] for f in canary_features] + [f['margin'] for f in non_canary_features]
auc_margin, ci_margin_lower, ci_margin_upper = compute_auc_with_ci(y_true, margin_scores, N_BOOTSTRAP)

# 特征 3: Confidence-based MIA
conf_scores = [f['confidence'] for f in canary_features] + [f['confidence'] for f in non_canary_features]
auc_conf, ci_conf_lower, ci_conf_upper = compute_auc_with_ci(y_true, conf_scores, N_BOOTSTRAP)

# 特征 4: 组合特征（加权平均）
combined_scores = []
for i in range(len(y_true)):
    if i < len(canary_features):
        f = canary_features[i]
    else:
        f = non_canary_features[i - len(canary_features)]
    
    # 标准化后组合
    combined = -f['loss'] + f['margin'] + f['confidence']
    combined_scores.append(combined)

auc_combined, ci_combined_lower, ci_combined_upper = compute_auc_with_ci(y_true, combined_scores, N_BOOTSTRAP)

print(f"\n   Loss-based MIA:")
print(f"      AUC: {auc_loss:.4f} (95% CI: [{ci_loss_lower:.4f}, {ci_loss_upper:.4f}])")
print(f"      Canary loss: {np.mean([f['loss'] for f in canary_features]):.4f}")
print(f"      Non-canary loss: {np.mean([f['loss'] for f in non_canary_features]):.4f}")

print(f"\n   Margin-based MIA:")
print(f"      AUC: {auc_margin:.4f} (95% CI: [{ci_margin_lower:.4f}, {ci_margin_upper:.4f}])")
print(f"      Canary margin: {np.mean([f['margin'] for f in canary_features]):.4f}")
print(f"      Non-canary margin: {np.mean([f['margin'] for f in non_canary_features]):.4f}")

print(f"\n   Confidence-based MIA:")
print(f"      AUC: {auc_conf:.4f} (95% CI: [{ci_conf_lower:.4f}, {ci_conf_upper:.4f}])")
print(f"      Canary confidence: {np.mean([f['confidence'] for f in canary_features]):.4f}")
print(f"      Non-canary confidence: {np.mean([f['confidence'] for f in non_canary_features]):.4f}")

print(f"\n   Combined MIA:")
print(f"      AUC: {auc_combined:.4f} (95% CI: [{ci_combined_lower:.4f}, {ci_combined_upper:.4f}])")

# 选择最强的攻击特征
best_auc = max(auc_loss, auc_margin, auc_conf, auc_combined)
if best_auc == auc_loss:
    best_name = "Loss"
    best_scores = loss_scores
elif best_auc == auc_margin:
    best_name = "Margin"
    best_scores = margin_scores
elif best_auc == auc_conf:
    best_name = "Confidence"
    best_scores = conf_scores
else:
    best_name = "Combined"
    best_scores = combined_scores

print(f"\n   ✓ Best attack: {best_name} (AUC={best_auc:.4f})")

# 逐层消融后的 MIA
print(f"\n[5] Computing MIA after layer ablation...")

class LayerAblation:
    def __init__(self, model, layer_id):
        self.model = model
        self.layer_id = layer_id
        self.original_params = {}
    
    def __enter__(self):
        layer = self.model.decoder.sentence_encoder.layers[self.layer_id]
        for name, param in layer.named_parameters():
            if 'left' in name or 'right' in name:
                self.original_params[name] = param.data.clone()
                param.data.zero_()
        return self
    
    def __exit__(self, *args):
        layer = self.model.decoder.sentence_encoder.layers[self.layer_id]
        for name, param in layer.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])

layer_mia_results = []

for layer_id in tqdm(range(N_LAYERS), desc="Layers"):
    with LayerAblation(model, layer_id):
        canary_features_ablated = [compute_sample_features(model, task, s) for s in canary_data]
        non_canary_features_ablated = [compute_sample_features(model, task, s) for s in non_canary_data]
    
    # 用最强的攻击特征
    if best_name == "Loss":
        scores_ablated = [-f['loss'] for f in canary_features_ablated] + [-f['loss'] for f in non_canary_features_ablated]
    elif best_name == "Margin":
        scores_ablated = [f['margin'] for f in canary_features_ablated] + [f['margin'] for f in non_canary_features_ablated]
    elif best_name == "Confidence":
        scores_ablated = [f['confidence'] for f in canary_features_ablated] + [f['confidence'] for f in non_canary_features_ablated]
    else:
        scores_ablated = []
        for i in range(len(y_true)):
            if i < len(canary_features_ablated):
                f = canary_features_ablated[i]
            else:
                f = non_canary_features_ablated[i - len(canary_features_ablated)]
            combined = -f['loss'] + f['margin'] + f['confidence']
            scores_ablated.append(combined)
    
    auc_ablated, ci_lower_ablated, ci_upper_ablated = compute_auc_with_ci(y_true, scores_ablated, N_BOOTSTRAP)
    delta_mia = best_auc - auc_ablated
    
    layer_mia_results.append({
        'layer_id': layer_id,
        'auc_ablated': auc_ablated,
        'ci_lower': ci_lower_ablated,
        'ci_upper': ci_upper_ablated,
        'delta_mia': delta_mia
    })
    print(f"   Layer {layer_id:2d}: AUC={auc_ablated:.4f}, ∆MIA={delta_mia:+.4f}")

# 保存结果
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results = {
    'original_model': {
        'best_attack': best_name,
        'auc': best_auc,
        'loss_auc': auc_loss,
        'margin_auc': auc_margin,
        'confidence_auc': auc_conf,
        'combined_auc': auc_combined,
        'ci_lower': ci_loss_lower if best_name == "Loss" else (ci_margin_lower if best_name == "Margin" else ci_conf_lower),
        'ci_upper': ci_loss_upper if best_name == "Loss" else (ci_margin_upper if best_name == "Margin" else ci_conf_upper),
        'canary_features': canary_features,
        'non_canary_features': non_canary_features
    },
    'layer_ablation': layer_mia_results
}

output_file = output_dir / "mia_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
print("=" * 60)
