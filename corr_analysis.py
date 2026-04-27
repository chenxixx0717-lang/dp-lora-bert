import json, numpy as np
from scipy.stats import spearmanr

s = json.load(open("results/s_l.json","r"))["layers"]
A = json.load(open("results/A_l.json","r"))
mia = json.load(open("results/mia_lora_layer_ablation.json","r"))["layers"]

# 按 layer 对齐
s_by = {x["layer"]: x for x in s}
A_by = {x["layer"]: x for x in A}
m_by = {x["layer"]: x for x in mia}
layers = sorted(s_by.keys())

s_l = np.array([s_by[l]["s_l"] for l in layers], float)
A_l = np.array([A_by[l]["A_l"] for l in layers], float)
dM  = np.array([m_by[l]["delta_auc"] for l in layers], float)

# 截断更符合你定义（可选，但我建议）
s_l = np.maximum(s_l, 0.0)
dM  = np.maximum(dM, 0.0)

print("Spearman(s, dM):", spearmanr(s_l, dM))
print("Spearman(A, dM):", spearmanr(A_l, dM))
print("Spearman(s, A): ", spearmanr(s_l, A_l))
