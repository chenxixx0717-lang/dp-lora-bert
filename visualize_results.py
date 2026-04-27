import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 60)
print("Visualizing Privacy Risk Analysis")
print("=" * 60)

# 加载数据
with open("results/s_l.json", 'r') as f:
    s_l_data = json.load(f)
with open("results/A_l.json", 'r') as f:
    A_l_data = json.load(f)
with open("results/risk_synthesis.json", 'r') as f:
    synthesis = json.load(f)
with open("results/mia_results.json", 'r') as f:
    mia_data = json.load(f)

# 提取数据
layers = [d['layer_id'] for d in s_l_data]
s_l_values = [abs(d['s_l']) for d in s_l_data]
A_l_values = [d['A_l'] for d in A_l_data]
risk_scores = [d['risk_score'] for d in synthesis['layer_risks']]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Privacy Risk Analysis Results', fontsize=16, fontweight='bold')

# 1. s_l (Canary Sensitivity)
ax1 = axes[0, 0]
ax1.bar(layers, s_l_values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Layer ID')
ax1.set_ylabel('|s_l| (Sensitivity)')
ax1.set_title('Canary Sensitivity per Layer')
ax1.grid(axis='y', alpha=0.3)

# 2. A_l (Gradient Amplification)
ax2 = axes[0, 1]
ax2.bar(layers, A_l_values, color='coral', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', label='Baseline (A_l=1)')
ax2.set_xlabel('Layer ID')
ax2.set_ylabel('A_l (Amplification Ratio)')
ax2.set_title('Gradient Amplification per Layer')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Risk Score (s_l × A_l)
ax3 = axes[1, 0]
colors = ['red' if i in synthesis['high_risk_layers'] else 'gray' for i in layers]
ax3.bar(layers, risk_scores, color=colors, alpha=0.7)
ax3.set_xlabel('Layer ID')
ax3.set_ylabel('Risk Score (s_l × A_l)')
ax3.set_title('Privacy Risk Score per Layer')
ax3.grid(axis='y', alpha=0.3)

# 添加高风险层标注
for layer_id in synthesis['high_risk_layers']:
    risk = risk_scores[layer_id]
    ax3.text(layer_id, risk, f'L{layer_id}', ha='center', va='bottom', fontweight='bold')

# 4. MIA Loss Distribution
ax4 = axes[1, 1]
canary_losses = mia_data['canary_losses']
non_canary_losses = mia_data['non_canary_losses']

ax4.hist(canary_losses, bins=20, alpha=0.6, label='Canary (Member)', color='red')
ax4.hist(non_canary_losses, bins=20, alpha=0.6, label='Non-canary (Non-member)', color='blue')
ax4.set_xlabel('Loss')
ax4.set_ylabel('Frequency')
ax4.set_title(f'MIA Loss Distribution (AUC={mia_data["auc"]:.3f})')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file = Path("results/privacy_risk_analysis.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to {output_file}")

# 生成文本报告
report_file = Path("results/privacy_risk_report.txt")
with open(report_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("PRIVACY RISK ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Overall Risk Level: {synthesis['risk_level']}\n")
    f.write(f"MIA Attack Success (AUC): {mia_data['auc']:.4f}\n")
    f.write(f"Average Risk per Layer: {synthesis['avg_risk']:.4f}\n\n")
    
    f.write("High-Risk Layers:\n")
    for i, layer_id in enumerate(synthesis['high_risk_layers'], 1):
        risk = risk_scores[layer_id]
        f.write(f"  {i}. Layer {layer_id}: risk={risk:.4f}\n")
    
    f.write("\n" + "=" * 60 + "\n")

print(f"✓ Report saved to {report_file}")
print("=" * 60)
