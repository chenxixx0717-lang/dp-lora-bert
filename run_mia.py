# run_mia.py
import os
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from fairseq import checkpoint_utils, tasks

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "log_dir"
CHECKPOINT_FILE = "checkpoint_best.pt"
DATA_PATH = os.path.abspath("../glue_data/SST-2-canary-bin")

SUBSET_DIR = Path("evaluation_subsets")
OUT_JSON = Path("results/mia_lora_layer_ablation.json")
OUT_JSON.parent.mkdir(exist_ok=True)

CLS_HEAD = "sentence_classification_head"

# MIA / Attack config
SEED = 42
BOOTSTRAP_SAMPLES = 1000
ATTACK_TRAIN_FRAC = 0.5  # split members/non-members into attack-train and attack-eval

# Learned attack feature switches
USE_CLS_REP = True
USE_LOGITS = True
USE_LOSS = True
USE_STATS = True
CLS_PCA_DIM = 64          # PCA dim for CLS rep
MC_DROPOUT = 0            # 0 disables; >0 enables MC dropout passes

# Runtime
BATCH_SIZE = 64


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic-ish; fairseq/transformer dropout kernels may still vary slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Loading
# -----------------------------
def load_model_and_task():
    ckpt = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)

    args = state["args"]
    args.data = DATA_PATH
    task = tasks.setup_task(args)

    model = task.build_model(args)

    # keep your previous upgrade workaround
    original_upgrade = model.upgrade_state_dict_named
    model.upgrade_state_dict_named = lambda state_dict, name: None
    model.load_state_dict(state["model"], strict=False)
    model.upgrade_state_dict_named = original_upgrade

    model.to(DEVICE)
    model.eval()
    return model, task


def load_dataset(task, split):
    task.load_dataset(split, combine=False, epoch=0)
    return task.dataset(split)


def read_subsets():
    with open(SUBSET_DIR / "member.json", "r") as f:
        mem = json.load(f)
    with open(SUBSET_DIR / "non_member.json", "r") as f:
        non = json.load(f)

    mem_idx = [x["idx"] for x in mem]
    non_idx = [x["idx"] for x in non]
    return mem, non, mem_idx, non_idx


def batch_to_device(batch):
    if batch is None:
        return None
    if "net_input" in batch:
        for k, v in batch["net_input"].items():
            if torch.is_tensor(v):
                batch["net_input"][k] = v.to(DEVICE)
    if "target" in batch and torch.is_tensor(batch["target"]):
        batch["target"] = batch["target"].to(DEVICE)
    return batch


# -----------------------------
# Metrics helpers (scores-based & loss-based)
# -----------------------------
def auc_from_scores(m_score, u_score):
    y = np.r_[np.ones(len(m_score)), np.zeros(len(u_score))]
    s = np.r_[m_score, u_score]
    return roc_auc_score(y, s)


def tpr_at_fpr_from_scores(m_score, u_score, fpr_target=0.01):
    y = np.r_[np.ones(len(m_score)), np.zeros(len(u_score))]
    s = np.r_[m_score, u_score]
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= fpr_target)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))


def auc_from_losses(m_loss, u_loss):
    # Yeom score = -loss
    return auc_from_scores(-m_loss, -u_loss)


def tpr_at_fpr_from_losses(m_loss, u_loss, fpr_target=0.01):
    return tpr_at_fpr_from_scores(-m_loss, -u_loss, fpr_target=fpr_target)


def stratified_bootstrap_auc_from_scores(m_score, u_score, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(B):
        mi = rng.integers(0, len(m_score), len(m_score))
        ui = rng.integers(0, len(u_score), len(u_score))
        aucs.append(auc_from_scores(m_score[mi], u_score[ui]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), aucs


def stratified_bootstrap_auc_from_losses(m_loss, u_loss, B=1000, seed=0):
    return stratified_bootstrap_auc_from_scores(-m_loss, -u_loss, B=B, seed=seed)


def paired_bootstrap_delta_auc_scores(m_base, u_base, m_ab, u_ab, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        mi = rng.integers(0, len(m_base), len(m_base))
        ui = rng.integers(0, len(u_base), len(u_base))
        auc_b = auc_from_scores(m_base[mi], u_base[ui])
        auc_a = auc_from_scores(m_ab[mi], u_ab[ui])
        deltas.append(auc_b - auc_a)
    deltas = np.array(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def paired_bootstrap_delta_auc_losses(m_base, u_base, m_ab, u_ab, B=1000, seed=0):
    # AUC(-loss)
    return paired_bootstrap_delta_auc_scores(-m_base, -u_base, -m_ab, -u_ab, B=B, seed=seed)


# -----------------------------
# Utility (valid acc)
# -----------------------------
def compute_accuracy(model, dataset, batch_size=64):
    correct, total = 0, 0
    model.eval()
    for s in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(s, min(len(dataset), s + batch_size))]
        batch = dataset.collater(samples)
        batch = batch_to_device(batch)
        if batch is None:
            continue

        net = batch["net_input"]
        targets = batch["target"].view(-1).long()

        with torch.no_grad():
            logits, _ = model(**net, features_only=True, classification_head_name=CLS_HEAD)
            pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


# -----------------------------
# Core forward: hidden -> logits (FIXED)
# -----------------------------
def forward_hidden_bt(model, net_input):
    """
    Returns hidden states in (B, T, C).
    Fairseq sometimes returns (T, B, C). We normalize to (B, T, C).
    """
    x, extra = model(**net_input, features_only=True)
    if not torch.is_tensor(x) or x.dim() != 3:
        raise RuntimeError(f"Unexpected features_only output: type={type(x)}, shape={getattr(x, 'shape', None)}")

    # normalize to (B,T,C)
    # common cases:
    # - (T,B,C): transpose(0,1)
    # - (B,T,C): keep
    if x.size(0) < x.size(1):  # heuristic; often T < B
        # could be (T,B,C)
        x_bt = x.transpose(0, 1).contiguous()
    else:
        # could already be (B,T,C)
        x_bt = x

    return x_bt, extra


def hidden_to_logits(model, hidden_bt):
    """
    Apply the classification head to hidden states (B,T,C) -> logits (B, num_classes)
    """
    head = model.classification_heads[CLS_HEAD]
    logits = head(hidden_bt)
    return logits


# -----------------------------
# Feature extraction for learned attack
# -----------------------------
def extract_features_and_losses(model, dataset, indices, batch_size=64,
                               use_cls_rep=True, use_logits=True, use_loss=True, use_stats=True,
                               mc_dropout=0, seed=0):
    """
    Extract per-example:
      - cls_rep: (N,768) or None
      - logits: (N,2) or None
      - loss:   (N,) or None
      - stats:  (N,5) = [p_true, p_max, margin, entropy, correct] or None
    """
    cls_reps = []
    logits_list = []
    losses = []
    stats_list = []

    # mode control for MC dropout
    base_mode_training = model.training

    for s in range(0, len(indices), batch_size):
        ids = indices[s:s + batch_size]
        samples = [dataset[i] for i in ids]
        batch = dataset.collater(samples)
        batch = batch_to_device(batch)
        if batch is None:
            continue

        net = batch["net_input"]
        targets = batch["target"].view(-1).long()
        B = targets.size(0)

        # MC dropout: average over multiple stochastic passes
        K = int(mc_dropout) if mc_dropout and mc_dropout > 0 else 1
        if K > 1:
            model.train()
        else:
            model.eval()

        with torch.no_grad():
            hidden_sum = None
            logits_sum = None
            loss_sum = None
            stats_sum = None

            for k in range(K):
                if K > 1:
                    # make MC deterministic across runs given SEED
                    torch.manual_seed(seed + 1000 * s + k)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + 1000 * s + k)

                hidden_bt, _ = forward_hidden_bt(model, net)       # (B,T,C)
                logits = hidden_to_logits(model, hidden_bt)        # (B,2)

                # loss per example
                loss_vec = F.cross_entropy(logits, targets, reduction="none")  # (B,)

                # stats
                probs = torch.softmax(logits, dim=-1)  # (B,2)
                p_true = probs[torch.arange(B, device=probs.device), targets]  # (B,)
                p_max, _ = probs.max(dim=-1)  # (B,)
                p_sorted, _ = probs.sort(dim=-1, descending=True)
                margin = p_sorted[:, 0] - p_sorted[:, 1]  # (B,)
                entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # (B,)
                correct = (logits.argmax(dim=-1) == targets).float()  # (B,)
                stats = torch.stack([p_true, p_max, margin, entropy, correct], dim=-1)  # (B,5)

                if hidden_sum is None:
                    hidden_sum = hidden_bt
                    logits_sum = logits
                    loss_sum = loss_vec
                    stats_sum = stats
                else:
                    hidden_sum = hidden_sum + hidden_bt
                    logits_sum = logits_sum + logits
                    loss_sum = loss_sum + loss_vec
                    stats_sum = stats_sum + stats

            hidden_bt = hidden_sum / K
            logits = logits_sum / K
            loss_vec = loss_sum / K
            stats = stats_sum / K

            if use_cls_rep:
                cls_rep = hidden_bt[:, 0, :]  # (B,768)
                cls_reps.append(cls_rep.detach().cpu().numpy())

            if use_logits:
                logits_list.append(logits.detach().cpu().numpy())

            if use_loss:
                losses.append(loss_vec.detach().cpu().numpy())

            if use_stats:
                stats_list.append(stats.detach().cpu().numpy())

    # restore mode
    model.train(base_mode_training)
    if not base_mode_training:
        model.eval()

    out = {}
    out["cls_rep"] = np.concatenate(cls_reps, axis=0) if use_cls_rep and len(cls_reps) else None
    out["logits"] = np.concatenate(logits_list, axis=0) if use_logits and len(logits_list) else None
    out["loss"] = np.concatenate(losses, axis=0) if use_loss and len(losses) else None
    out["stats"] = np.concatenate(stats_list, axis=0) if use_stats and len(stats_list) else None
    return out


def compute_losses_only(model, dataset, indices, batch_size=64):
    """Fast path for Yeom baseline only."""
    losses = []
    model.eval()
    for s in range(0, len(indices), batch_size):
        ids = indices[s:s + batch_size]
        samples = [dataset[i] for i in ids]
        batch = dataset.collater(samples)
        batch = batch_to_device(batch)
        if batch is None:
            continue

        net = batch["net_input"]
        targets = batch["target"].view(-1).long()

        with torch.no_grad():
            logits, _ = model(**net, features_only=True, classification_head_name=CLS_HEAD)
            l = F.cross_entropy(logits, targets, reduction="none")
        losses.append(l.detach().cpu().numpy())
    return np.concatenate(losses, axis=0)


# -----------------------------
# Learned attack pipeline
# -----------------------------
class AttackPipeline:
    def __init__(self, use_cls_rep=True, use_logits=True, use_loss=True, use_stats=True,
                 cls_pca_dim=64, seed=0):
        self.use_cls_rep = use_cls_rep
        self.use_logits = use_logits
        self.use_loss = use_loss
        self.use_stats = use_stats
        self.cls_pca_dim = int(cls_pca_dim) if cls_pca_dim is not None else 0
        self.seed = seed

        self.pca = None
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )

    def _build_other_features(self, feats: dict):
        parts = []
        n = None
        if self.use_logits and feats["logits"] is not None:
            parts.append(feats["logits"])
            n = feats["logits"].shape[0]
        if self.use_loss and feats["loss"] is not None:
            parts.append(feats["loss"].reshape(-1, 1))
            n = feats["loss"].shape[0]
        if self.use_stats and feats["stats"] is not None:
            parts.append(feats["stats"])
            n = feats["stats"].shape[0]

        if len(parts) == 0:
            if n is None:
                raise RuntimeError("No features enabled for learned attack.")
            return np.zeros((n, 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, feats_train: dict, y_train: np.ndarray):
        y_train = y_train.astype(int)
        other = self._build_other_features(feats_train)

        if self.use_cls_rep and feats_train["cls_rep"] is not None and self.cls_pca_dim > 0:
            cls = feats_train["cls_rep"].astype(np.float32)
            dim = min(self.cls_pca_dim, cls.shape[1], cls.shape[0])
            self.pca = PCA(n_components=dim, random_state=self.seed)
            cls_red = self.pca.fit_transform(cls).astype(np.float32)
        elif self.use_cls_rep and feats_train["cls_rep"] is not None:
            cls_red = feats_train["cls_rep"].astype(np.float32)
        else:
            cls_red = np.zeros((other.shape[0], 0), dtype=np.float32)

        X = np.concatenate([cls_red, other], axis=1)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y_train)
        return self

    def predict_member_proba(self, feats: dict) -> np.ndarray:
        other = self._build_other_features(feats)

        if self.use_cls_rep and feats["cls_rep"] is not None and self.pca is not None:
            cls = feats["cls_rep"].astype(np.float32)
            cls_red = self.pca.transform(cls).astype(np.float32)
        elif self.use_cls_rep and feats["cls_rep"] is not None:
            cls_red = feats["cls_rep"].astype(np.float32)
        else:
            cls_red = np.zeros((other.shape[0], 0), dtype=np.float32)

        X = np.concatenate([cls_red, other], axis=1)
        Xs = self.scaler.transform(X)
        proba = self.clf.predict_proba(Xs)[:, 1]
        return proba.astype(np.float32)


# -----------------------------
# LoRA ablation
# -----------------------------
@contextmanager
def ablate_lora_layer(model, layer_idx: int):
    """
    Layer-wise ablation: zero all LoraLinear weights in that encoder layer (inference-time only).
    """
    layer = model.decoder.sentence_encoder.layers[layer_idx]
    backups = []
    for _, module in layer.named_modules():
        if module.__class__.__name__ == "LoraLinear" and hasattr(module, "weight"):
            backups.append((module, module.weight.detach().clone()))
            module.weight.data.zero_()
    try:
        yield len(backups)
    finally:
        for module, w in backups:
            module.weight.data.copy_(w)


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)

    print("=" * 90)
    print("MIA (Stronger learned attack: CLS-rep + logits + loss/stats) + Yeom baseline + Layer-wise LoRA ablation")
    print("=" * 90)
    print("DATA_PATH:", DATA_PATH)
    print("CKPT:", os.path.join(MODEL_DIR, CHECKPOINT_FILE))
    print("DEVICE:", DEVICE)
    print(f"USE_CLS_REP={USE_CLS_REP} USE_LOGITS={USE_LOGITS} USE_LOSS={USE_LOSS} USE_STATS={USE_STATS} "
          f"CLS_PCA_DIM={CLS_PCA_DIM} MC_DROPOUT={MC_DROPOUT}")
    print(f"ATTACK_TRAIN_FRAC={ATTACK_TRAIN_FRAC}  BOOTSTRAP_SAMPLES={BOOTSTRAP_SAMPLES}  SEED={SEED}")

    model, task = load_model_and_task()
    train_ds = load_dataset(task, "train")
    valid_ds = load_dataset(task, "valid")

    mem, non, mem_idx, non_idx = read_subsets()
    print(f"Members: {len(mem_idx)}  Non-members: {len(non_idx)}")

    assert all(0 <= i < len(train_ds) for i in mem_idx), "member idx out of range for train"
    assert all(0 <= i < len(valid_ds) for i in non_idx), "non-member idx out of range for valid"

    encoder = model.decoder.sentence_encoder
    num_layers = len(encoder.layers)
    print("Encoder layers:", num_layers)

    # -----------------------------
    # Attack split (train/eval) to avoid inflated learned attack AUC
    # -----------------------------
    rng = np.random.default_rng(SEED)
    mem_idx = np.array(mem_idx, dtype=int)
    non_idx = np.array(non_idx, dtype=int)
    rng.shuffle(mem_idx)
    rng.shuffle(non_idx)

    m_train_n = int(len(mem_idx) * ATTACK_TRAIN_FRAC)
    u_train_n = int(len(non_idx) * ATTACK_TRAIN_FRAC)

    mem_train_idx = mem_idx[:m_train_n].tolist()
    mem_eval_idx = mem_idx[m_train_n:].tolist()
    non_train_idx = non_idx[:u_train_n].tolist()
    non_eval_idx = non_idx[u_train_n:].tolist()

    print(f"[ATTACK SPLIT] member train={len(mem_train_idx)} eval={len(mem_eval_idx)} | "
          f"non-member train={len(non_train_idx)} eval={len(non_eval_idx)}")

    # -----------------------------
    # BASE: Yeom on eval split
    # -----------------------------
    print("\n[BASE] computing Yeom losses (eval split)...")
    m_loss_base = compute_losses_only(model, train_ds, mem_eval_idx, batch_size=BATCH_SIZE)
    u_loss_base = compute_losses_only(model, valid_ds, non_eval_idx, batch_size=BATCH_SIZE)

    yeom_auc_base = auc_from_losses(m_loss_base, u_loss_base)
    _, yeom_ci_lo, yeom_ci_hi, _ = stratified_bootstrap_auc_from_losses(
        m_loss_base, u_loss_base, B=BOOTSTRAP_SAMPLES, seed=SEED
    )
    yeom_tpr01 = tpr_at_fpr_from_losses(m_loss_base, u_loss_base, fpr_target=0.01)
    print(f"[BASE][Yeom]   AUC={yeom_auc_base:.4f}  95%CI=[{yeom_ci_lo:.4f},{yeom_ci_hi:.4f}]  "
          f"TPR@1%FPR={yeom_tpr01:.4f}")

    # -----------------------------
    # BASE: Learned attack (train attacker on train split, evaluate on eval split)
    # -----------------------------
    print("\n[BASE] extracting features for learned attacker (train split + eval split)...")
    feats_m_train = extract_features_and_losses(
        model, train_ds, mem_train_idx, batch_size=BATCH_SIZE,
        use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
        mc_dropout=MC_DROPOUT, seed=SEED
    )
    feats_u_train = extract_features_and_losses(
        model, valid_ds, non_train_idx, batch_size=BATCH_SIZE,
        use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
        mc_dropout=MC_DROPOUT, seed=SEED
    )

    feats_m_eval = extract_features_and_losses(
        model, train_ds, mem_eval_idx, batch_size=BATCH_SIZE,
        use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
        mc_dropout=MC_DROPOUT, seed=SEED
    )
    feats_u_eval = extract_features_and_losses(
        model, valid_ds, non_eval_idx, batch_size=BATCH_SIZE,
        use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
        mc_dropout=MC_DROPOUT, seed=SEED
    )

    # build train set for attacker
    def concat_feats(a: dict, b: dict):
        out = {}
        for k in ["cls_rep", "logits", "loss", "stats"]:
            if a.get(k, None) is None and b.get(k, None) is None:
                out[k] = None
            elif a.get(k, None) is None:
                out[k] = b[k]
            elif b.get(k, None) is None:
                out[k] = a[k]
            else:
                out[k] = np.concatenate([a[k], b[k]], axis=0)
        return out

    feats_train = concat_feats(feats_m_train, feats_u_train)
    y_train = np.r_[np.ones(len(mem_train_idx)), np.zeros(len(non_train_idx))].astype(int)

    attacker = AttackPipeline(
        use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
        cls_pca_dim=CLS_PCA_DIM, seed=SEED
    ).fit(feats_train, y_train)

    m_score_base = attacker.predict_member_proba(feats_m_eval)
    u_score_base = attacker.predict_member_proba(feats_u_eval)

    learned_auc_base = auc_from_scores(m_score_base, u_score_base)
    _, l_ci_lo, l_ci_hi, _ = stratified_bootstrap_auc_from_scores(
        m_score_base, u_score_base, B=BOOTSTRAP_SAMPLES, seed=SEED
    )
    learned_tpr01 = tpr_at_fpr_from_scores(m_score_base, u_score_base, fpr_target=0.01)

    print(f"[BASE][Learned] AUC={learned_auc_base:.4f}  95%CI=[{l_ci_lo:.4f},{l_ci_hi:.4f}]  "
          f"TPR@1%FPR={learned_tpr01:.4f}")

    # Utility
    print("\n[BASE] computing utility (valid acc)...")
    acc_base = compute_accuracy(model, valid_ds, batch_size=BATCH_SIZE)
    print(f"[BASE] valid acc = {acc_base:.4f}")

    # -----------------------------
    # Ablation
    # -----------------------------
    results = {
        "config": {
            "seed": SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "attack_train_frac": ATTACK_TRAIN_FRAC,
            "use_cls_rep": USE_CLS_REP,
            "use_logits": USE_LOGITS,
            "use_loss": USE_LOSS,
            "use_stats": USE_STATS,
            "cls_pca_dim": CLS_PCA_DIM,
            "mc_dropout": MC_DROPOUT,
            "batch_size": BATCH_SIZE,
        },
        "base": {
            "yeom": {
                "auc": float(yeom_auc_base),
                "auc_ci": [float(yeom_ci_lo), float(yeom_ci_hi)],
                "tpr_at_1pct_fpr": float(yeom_tpr01),
            },
            "learned": {
                "auc": float(learned_auc_base),
                "auc_ci": [float(l_ci_lo), float(l_ci_hi)],
                "tpr_at_1pct_fpr": float(learned_tpr01),
            },
            "valid_acc": float(acc_base),
            "eval_sizes": {
                "member_eval": len(mem_eval_idx),
                "non_member_eval": len(non_eval_idx),
            },
        },
        "layers": [],
    }

    print("\n[ABLATION] layer-wise LoRA ablation (fixed attacker trained on base)...")
    for l in tqdm(range(num_layers), desc="layer"):
        with ablate_lora_layer(model, l) as n_lora:
            # Yeom eval
            m_loss_ab = compute_losses_only(model, train_ds, mem_eval_idx, batch_size=BATCH_SIZE)
            u_loss_ab = compute_losses_only(model, valid_ds, non_eval_idx, batch_size=BATCH_SIZE)
            yeom_auc_ab = auc_from_losses(m_loss_ab, u_loss_ab)
            _, ylo, yhi, _ = stratified_bootstrap_auc_from_losses(
                m_loss_ab, u_loss_ab, B=BOOTSTRAP_SAMPLES, seed=SEED
            )
            yeom_delta = yeom_auc_base - yeom_auc_ab
            y_dlo, y_dhi = paired_bootstrap_delta_auc_losses(
                m_loss_base, u_loss_base, m_loss_ab, u_loss_ab, B=BOOTSTRAP_SAMPLES, seed=SEED
            )

            # Learned eval (re-extract eval features under ablated model, attacker fixed)
            feats_m_ab = extract_features_and_losses(
                model, train_ds, mem_eval_idx, batch_size=BATCH_SIZE,
                use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
                mc_dropout=MC_DROPOUT, seed=SEED
            )
            feats_u_ab = extract_features_and_losses(
                model, valid_ds, non_eval_idx, batch_size=BATCH_SIZE,
                use_cls_rep=USE_CLS_REP, use_logits=USE_LOGITS, use_loss=USE_LOSS, use_stats=USE_STATS,
                mc_dropout=MC_DROPOUT, seed=SEED
            )
            m_score_ab = attacker.predict_member_proba(feats_m_ab)
            u_score_ab = attacker.predict_member_proba(feats_u_ab)

            learned_auc_ab = auc_from_scores(m_score_ab, u_score_ab)
            _, alo, ahi, _ = stratified_bootstrap_auc_from_scores(
                m_score_ab, u_score_ab, B=BOOTSTRAP_SAMPLES, seed=SEED
            )
            learned_delta = learned_auc_base - learned_auc_ab
            l_dlo, l_dhi = paired_bootstrap_delta_auc_scores(
                m_score_base, u_score_base, m_score_ab, u_score_ab, B=BOOTSTRAP_SAMPLES, seed=SEED
            )

            # Utility under ablation
            acc_ab = compute_accuracy(model, valid_ds, batch_size=BATCH_SIZE)
            util_drop = acc_base - acc_ab

        results["layers"].append({
            "layer": int(l),
            "lora_modules": int(n_lora),

            "yeom_auc_ablated": float(yeom_auc_ab),
            "yeom_auc_ablated_ci": [float(ylo), float(yhi)],
            "yeom_delta_auc": float(yeom_delta),
            "yeom_delta_auc_ci": [float(y_dlo), float(y_dhi)],

            "learned_auc_ablated": float(learned_auc_ab),
            "learned_auc_ablated_ci": [float(alo), float(ahi)],
            "delta_auc": float(learned_delta),                 # keep this name for your downstream scripts
            "delta_auc_ci": [float(l_dlo), float(l_dhi)],

            "valid_acc_ablated": float(acc_ab),
            "util_drop": float(util_drop),
        })

    results["layers"].sort(key=lambda x: x["delta_auc"], reverse=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", OUT_JSON)

    print("\nTop layers by ΔAUC (learned attack):")
    for x in results["layers"][:5]:
        l = x["layer"]
        d = x["delta_auc"]
        lo, hi = x["delta_auc_ci"]
        ud = x["util_drop"]
        print(f"  layer {l:2d}: ΔAUC={d:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  util_drop={ud:+.4f}  lora={x['lora_modules']}")


if __name__ == "__main__":
    main()
