import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

S_FILE = Path("results/s_l.json")
A_FILE = Path("results/A_l.json")
MIA_FILE = Path("results/mia_lora_layer_ablation.json")

USE_CLAMP = True

def load_arrays():
    s_obj = json.loads(S_FILE.read_text(encoding="utf-8"))
    A_list = json.loads(A_FILE.read_text(encoding="utf-8"))
    mia_obj = json.loads(MIA_FILE.read_text(encoding="utf-8"))

    s_by = {int(x["layer"]): float(x["s_l"]) for x in s_obj["layers"]}
    A_by = {int(x["layer"]): float(x["A_l"]) for x in A_list}
    d_by = {int(x["layer"]): float(x["delta_auc"]) for x in mia_obj["layers"]}

    layers = sorted(set(s_by) & set(A_by) & set(d_by))
    A = np.array([A_by[l] for l in layers], float)
    s = np.array([s_by[l] for l in layers], float)
    d = np.array([d_by[l] for l in layers], float)
    return layers, A, s, d

def annotate(ax, x, y, layers):
    for xi, yi, li in zip(x, y, layers):
        ax.text(xi, yi, str(li), fontsize=10, ha="left", va="bottom")

def main():
    layers, A, s, d = load_arrays()
    if USE_CLAMP:
        s = np.maximum(s, 0.0)
        d = np.maximum(d, 0.0)

    # 图1：A vs s
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(A, s, s=50)
    annotate(ax, A, s, layers)
    ax.set_xlabel("A_l (grad amplification ratio)")
    ax.set_ylabel("s_l (canary loss sensitivity)")
    ax.set_yscale("symlog", linthresh=1e-5)  # s跨度很大，用symlog更好看
    ax.set_xscale("log")  # A通常也跨数量级
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title("Layer-wise: A_l vs s_l")
    fig.tight_layout()
    fig.savefig("results/scatter_A_vs_s.png", dpi=200)

    # 图2：s vs delta_auc (learned)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(s, d, s=50)
    annotate(ax, s, d, layers)
    ax.set_xlabel("s_l (canary loss sensitivity)")
    ax.set_ylabel("ΔAUC_l (learned MIA)")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title("Layer-wise: s_l vs ΔAUC_l (learned attack)")
    fig.tight_layout()
    fig.savefig("results/scatter_s_vs_deltaAUC.png", dpi=200)

    print("Saved:")
    print("  results/scatter_A_vs_s.png")
    print("  results/scatter_s_vs_deltaAUC.png")

if __name__ == "__main__":
    main()
