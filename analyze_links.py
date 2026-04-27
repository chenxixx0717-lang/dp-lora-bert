import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

S_FILE = Path("results/s_l.json")
A_FILE = Path("results/A_l.json")
MIA_FILE = Path("results/mia_lora_layer_ablation.json")

SEED = 42
BOOT = 20000          # bootstrap次数（12个点建议>=1e4）
PERM = 50000          # permutation次数（p-value更稳）
USE_CLAMP = True      # 是否把 s_l、delta_auc 负值截断为0（推荐）


def load_arrays():
    s_obj = json.loads(S_FILE.read_text(encoding="utf-8"))
    A_list = json.loads(A_FILE.read_text(encoding="utf-8"))
    mia_obj = json.loads(MIA_FILE.read_text(encoding="utf-8"))

    # s: layers list
    s_by = {int(x["layer"]): float(x["s_l"]) for x in s_obj["layers"]}

    # A: list[dict(layer, A_l, ...)]
    A_by = {int(x["layer"]): float(x["A_l"]) for x in A_list}

    # MIA: learned 的逐层贡献（你的文件里 learned 的 ΔAUC 存在 key="delta_auc"）
    mia_layers = mia_obj["layers"]
    d_by = {}
    for x in mia_layers:
        l = int(x["layer"])
        # learned attack delta (base learned auc - ablated learned auc)
        d_by[l] = float(x.get("delta_auc", x.get("learned_delta_auc")))

    layers = sorted(set(s_by) & set(A_by) & set(d_by))
    if len(layers) != 12:
        raise ValueError(f"Need 12 layers, got {len(layers)} common layers={layers}")

    s = np.array([s_by[l] for l in layers], dtype=float)
    A = np.array([A_by[l] for l in layers], dtype=float)
    d = np.array([d_by[l] for l in layers], dtype=float)
    return layers, A, s, d


def spearman_boot_ci(x, y, boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    r_obs = spearmanr(x, y).correlation

    rs = []
    for _ in range(boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        r = spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(rs, dtype=float)
    ci = np.quantile(rs, [0.025, 0.975])
    return float(r_obs), (float(ci[0]), float(ci[1]))


def spearman_perm_pvalue(x, y, perm=50000, seed=0):
    rng = np.random.default_rng(seed)
    r_obs = spearmanr(x, y).correlation
    cnt = 0
    for _ in range(perm):
        yp = rng.permutation(y)
        r = spearmanr(x, yp).correlation
        if abs(r) >= abs(r_obs) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (perm + 1)  # add-one smoothing
    return float(p)


def report(name, x, y):
    r, ci = spearman_boot_ci(x, y, boot=BOOT, seed=SEED)
    p = spearman_perm_pvalue(x, y, perm=PERM, seed=SEED)
    print(f"{name:18s} Spearman rho={r:+.4f}  95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  perm_p={p:.6f}")


def main():
    layers, A, s, d = load_arrays()

    if USE_CLAMP:
        s2 = np.maximum(s, 0.0)
        d2 = np.maximum(d, 0.0)
    else:
        s2, d2 = s, d

    print("Layers:", layers)
    print(f"Clamp negatives: {USE_CLAMP}")
    print()

    report("A vs s", A, s2)
    report("s vs delta_auc", s2, d2)
    report("A vs delta_auc", A, d2)


if __name__ == "__main__":
    main()
