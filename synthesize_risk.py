# synthesize_risk.py
import json
import numpy as np
from pathlib import Path

S_FILE = Path("results/s_l.json")
A_FILE = Path("results/A_l.json")
MIA_FILE = Path("results/mia_lora_layer_ablation.json")
OUT_FILE = Path("results/risk_synthesis.json")

# 选择用哪种 MIA 攻击来做“隐私风险合成”
# "learned" 或 "yeom"
ATTACK_NAME = "learned"


def rank01(x: np.ndarray) -> np.ndarray:
    """把数值映射到[0,1]的秩（稳定融合不同量纲的指标）"""
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([1.0], dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / (len(x) - 1 + 1e-12)


def dig(d, path):
    """安全地取嵌套字段：dig(obj, ["base","learned","auc"]) -> value or None"""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _get_base_block(mia_obj: dict, attack: str):
    """
    兼容：
      base: {auc: ...} (旧)
      base: {yeom:{...}, learned:{...}} (新)
    """
    base = mia_obj.get("base", None)
    if not isinstance(base, dict):
        return None

    # 新格式：base[attack] 是个 dict
    if attack in base and isinstance(base[attack], dict):
        return base[attack]

    # 旧格式：base 自己就有 auc
    if "auc" in base or "AUC" in base:
        return base

    return None


def get_base_auc(mia_obj: dict, attack: str) -> float:
    # 顶层兼容
    cands = [
        mia_obj.get("base_auc"),
        mia_obj.get("auc_base"),
        mia_obj.get("AUC_base"),
    ]
    for v in cands:
        if v is not None:
            return float(v)

    base_blk = _get_base_block(mia_obj, attack)
    if base_blk is None:
        raise KeyError(
            f"Cannot find base block for attack='{attack}'. "
            f"Top-level keys={list(mia_obj.keys())[:50]} base_keys={list(mia_obj.get('base', {}).keys())[:50]}"
        )

    v = base_blk.get("auc", base_blk.get("AUC", None))
    if v is None:
        raise KeyError(
            f"Cannot find base auc for attack='{attack}'. base_block_keys={list(base_blk.keys())}"
        )
    return float(v)


def get_base_auc_ci(mia_obj: dict, attack: str):
    base_blk = _get_base_block(mia_obj, attack)
    if base_blk is None:
        return None

    # 常见命名：auc_ci
    v = base_blk.get("auc_ci", None)
    if v is not None:
        return [float(v[0]), float(v[1])]

    # 兼容别名
    for key in ["base_auc_ci", "auc_base_ci"]:
        if key in mia_obj:
            vv = mia_obj[key]
            return [float(vv[0]), float(vv[1])]

    return None


def get_base_tpr1(mia_obj: dict, attack: str):
    base_blk = _get_base_block(mia_obj, attack)
    if base_blk is None:
        return None

    # 你 run_mia.py 里最常见的 key
    for k in ["tpr_at_1pct_fpr", "tpr_at_1%fpr", "tpr@1%fpr", "tpr_at_1fpr", "tpr@1fpr"]:
        if k in base_blk:
            return float(base_blk[k])

    # 也兼容顶层
    for k in ["base_tpr_at_1pct_fpr", "base_tpr_at_1fpr", "base_tpr@1fpr"]:
        if k in mia_obj:
            return float(mia_obj[k])

    return None


def get_mia_layers(mia_obj: dict):
    if "layers" in mia_obj and isinstance(mia_obj["layers"], list):
        return mia_obj["layers"]
    if "layer_ablation" in mia_obj and isinstance(mia_obj["layer_ablation"], list):
        return mia_obj["layer_ablation"]
    v = dig(mia_obj, ["ablation", "layers"])
    if isinstance(v, list):
        return v
    raise KeyError(f"Cannot find layer list in MIA file. keys={list(mia_obj.keys())[:50]}")


def get_delta_auc(layer_item: dict, attack: str) -> float:
    """
    兼容：
      delta_auc (默认就是你当前 learned attack 的 delta)
      delta_auc_learned / delta_auc_yeom（如果你未来改成同时存两套）
    """
    # attack-specific first
    k1 = f"delta_auc_{attack}"
    if k1 in layer_item:
        return float(layer_item[k1])

    # generic
    if "delta_auc" in layer_item:
        return float(layer_item["delta_auc"])
    if "delta_mia" in layer_item:
        return float(layer_item["delta_mia"])

    raise KeyError(f"Cannot find delta_auc in layer item keys={list(layer_item.keys())}")


def get_util_drop(layer_item: dict) -> float:
    for k in ["util_drop", "utility_drop", "valid_acc_drop", "acc_drop"]:
        if k in layer_item:
            return float(layer_item[k])
    return 0.0


def main():
    print("=" * 80)
    print(f"Privacy Risk Synthesis ({ATTACK_NAME} attack, 12 layers)")
    print("=" * 80)

    with open(S_FILE, "r", encoding="utf-8") as f:
        s_obj = json.load(f)
    with open(A_FILE, "r", encoding="utf-8") as f:
        A_list = json.load(f)
    with open(MIA_FILE, "r", encoding="utf-8") as f:
        mia_obj = json.load(f)

    s_layers = s_obj["layers"]
    mia_layers = get_mia_layers(mia_obj)

    num_layers = len(s_layers)
    assert len(A_list) == num_layers, f"A_l layers={len(A_list)} != s_l layers={num_layers}"
    assert len(mia_layers) == num_layers, f"MIA layers={len(mia_layers)} != s_l layers={num_layers}"

    base_auc = get_base_auc(mia_obj, ATTACK_NAME)
    base_auc_ci = get_base_auc_ci(mia_obj, ATTACK_NAME)
    base_tpr1 = get_base_tpr1(mia_obj, ATTACK_NAME)

    print("[Loaded]")
    print(f"  layers = {num_layers}")
    print(f"  attack = {ATTACK_NAME}")
    print(f"  base_auc = {base_auc:.4f}")
    if base_auc_ci is not None:
        print(f"  base_auc_ci = [{base_auc_ci[0]:.4f}, {base_auc_ci[1]:.4f}]")
    if base_tpr1 is not None:
        print(f"  base_tpr@1%fpr = {base_tpr1:.4f}")

    # align by layer id
    s_by_layer = {int(x["layer"]): x for x in s_layers}
    A_by_layer = {int(x["layer"]): x for x in A_list}
    mia_by_layer = {int(x.get("layer", x.get("layer_idx", -1))): x for x in mia_layers}

    layers = list(range(num_layers))
    for l in layers:
        if l not in s_by_layer or l not in A_by_layer or l not in mia_by_layer:
            raise ValueError(f"missing layer {l} in some file")

    # collect arrays
    s_vals = np.array([float(s_by_layer[l]["s_l"]) for l in layers], dtype=float)
    A_vals = np.array([float(A_by_layer[l]["A_l"]) for l in layers], dtype=float)
    d_vals = np.array([get_delta_auc(mia_by_layer[l], ATTACK_NAME) for l in layers], dtype=float)

    util_drop_s = np.array([float(s_by_layer[l].get("utility_drop", 0.0)) for l in layers], dtype=float)
    util_drop_mia = np.array([get_util_drop(mia_by_layer[l]) for l in layers], dtype=float)

    # 截断负数（负的一般是估计噪声/随机波动）
    s_pos = np.maximum(s_vals, 0.0)
    d_pos = np.maximum(d_vals, 0.0)

    # risk scores
    risk_raw = s_pos * A_vals * d_pos
    risk_rank = rank01(s_pos) * rank01(A_vals) * rank01(d_pos)

    util = np.maximum(util_drop_mia, 1e-12)
    privacy_efficiency = d_pos / util

    # assemble table
    layer_risks = []
    for l in layers:
        li = mia_by_layer[l]
        item = {
            "layer": l,
            "attack": ATTACK_NAME,
            "s_l": float(s_vals[l]),
            "A_l": float(A_vals[l]),
            "delta_auc": float(d_vals[l]),
            "risk_raw": float(risk_raw[l]),
            "risk_rank": float(risk_rank[l]),
            "utility_drop_s": float(util_drop_s[l]),
            "utility_drop_mia": float(util_drop_mia[l]),
            "privacy_efficiency": float(privacy_efficiency[l]),
            "auc_ablated": float(li.get("auc_ablated", li.get("auc", np.nan))),
            "delta_auc_ci": li.get("delta_auc_ci", li.get("delta_mia_ci", None)),
        }
        layer_risks.append(item)

    layer_risks_sorted = sorted(layer_risks, key=lambda x: x["risk_rank"], reverse=True)

    print("\nTop layers by risk_rank:")
    for x in layer_risks_sorted[:5]:
        ci = x["delta_auc_ci"]
        ci_str = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "N/A"
        print(
            f"  layer {x['layer']:2d}: risk_rank={x['risk_rank']:.4f}  "
            f"s={x['s_l']:+.6f}  A={x['A_l']:.4f}  ΔAUC={x['delta_auc']:+.4f}  CI={ci_str}  "
            f"util_drop(mia)={x['utility_drop_mia']:+.4f}"
        )

    # overall risk level（按你选的 attack 的 AUC）
    ci_low = float(base_auc_ci[0]) if base_auc_ci is not None else base_auc
    if ci_low >= 0.65:
        risk_level = "CRITICAL"
    elif ci_low >= 0.60:
        risk_level = "HIGH"
    elif ci_low >= 0.55:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    out = {
        "attack": ATTACK_NAME,
        "base_auc": base_auc,
        "base_auc_ci": base_auc_ci,
        "base_tpr_at_1pct_fpr": base_tpr1,
        "risk_level": risk_level,
        "top_layers_by_risk_rank": [x["layer"] for x in layer_risks_sorted[:3]],
        "layer_risks": layer_risks_sorted,
    }

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {OUT_FILE}")
    print(f"Risk level: {risk_level}")
    print("=" * 80)


if __name__ == "__main__":
    main()
