import sys
sys.path.insert(0, ".")  # 确保用你项目内的 fairseq

import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# -------------------------
# IMPORTANT: disable roberta upgrade (avoids dense-related crash)
# -------------------------
import fairseq.models.roberta.model as roberta_model
roberta_model.RobertaModel.upgrade_state_dict_named = lambda self, state_dict, name: None

from fairseq import checkpoint_utils


# -------------------------
# Configs
# -------------------------
@dataclass
class CorruptCfg:
    alpha: float = 0.3               # noise strength multiplier
    exclude_cls: bool = True         # do not corrupt CLS token (pos 0)


@dataclass
class PatchCfg:
    layer_idx: int                   # 0..11
    module: str                      # "attn" or "ffn"
    scope: str                       # "cls" or "all"


# -------------------------
# Utilities: find sentence_encoder and layers
# -------------------------
def get_sentence_encoder(model):
    if hasattr(model, "decoder") and hasattr(model.decoder, "sentence_encoder"):
        return model.decoder.sentence_encoder
    if hasattr(model, "sentence_encoder"):
        return model.sentence_encoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "sentence_encoder"):
        return model.encoder.sentence_encoder
    raise AttributeError("Cannot find sentence_encoder. Please inspect model attributes.")


def get_embed_module(sent_enc):
    # fairseq transformer_sentence_encoder 常见：layernorm_embedding
    if hasattr(sent_enc, "layernorm_embedding"):
        return sent_enc.layernorm_embedding
    # 兜底：只对 token embedding 加噪声（不含 position embedding）
    if hasattr(sent_enc, "embed_tokens"):
        return sent_enc.embed_tokens
    raise AttributeError("Cannot find embedding module (layernorm_embedding/embed_tokens).")


def forward_logits(model, tokens):
    """
    Return logits: (B, num_classes) before softmax.
    """
    out = model(tokens, classification_head_name="sentence_classification_head")
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(logits):
        raise RuntimeError(f"Cannot parse logits from output type: {type(out)}")
    return logits


# -------------------------
# Context object: stores clean caches and per-batch embedding std/noise
# -------------------------
class TraceContext:
    def __init__(self):
        self.mode = "clean"  # "clean" | "corrupt" | "patch"
        self.clean_cache_attn: Dict[int, torch.Tensor] = {}
        self.clean_cache_ffn: Dict[int, torch.Tensor] = {}
        self.emb_std: Optional[torch.Tensor] = None

        # 固定同一 batch 的噪声（corrupt + patch 共用）
        self.emb_noise: Optional[torch.Tensor] = None

        self.corrupt_cfg: Optional[CorruptCfg] = None
        self.patch_cfg: Optional[PatchCfg] = None

    def clear_for_new_clean_run(self):
        """在 clean forward 之前调用：清 caches + std + noise"""
        self.clean_cache_attn.clear()
        self.clean_cache_ffn.clear()
        self.emb_std = None
        self.emb_noise = None
        self.patch_cfg = None


# -------------------------
# Hooks
# -------------------------
def make_embed_hook(ctx: TraceContext):
    def hook(module, inp, out):
        # out: usually (B, T, C)
        if ctx.mode == "clean":
            x = out
            if ctx.corrupt_cfg is not None and ctx.corrupt_cfg.exclude_cls and x.size(1) > 1:
                x2 = x[:, 1:, :]
            else:
                x2 = x
            ctx.emb_std = x2.detach().float().std()  # scalar tensor
            ctx.emb_noise = None
            return out

        # corrupt / patch: add noise (reuse same noise within the batch)
        assert ctx.corrupt_cfg is not None
        assert ctx.emb_std is not None, "emb_std not set; run clean forward first."

        if (ctx.emb_noise is None) or (ctx.emb_noise.shape != out.shape) or (ctx.emb_noise.device != out.device):
            sigma = (ctx.corrupt_cfg.alpha * ctx.emb_std).to(out.device)
            noise = torch.randn_like(out) * sigma.to(out.dtype)
            if ctx.corrupt_cfg.exclude_cls and out.size(1) > 0:
                noise[:, 0:1, :] = 0.0
            ctx.emb_noise = noise
        return out + ctx.emb_noise
    return hook


def make_attn_hook(ctx: TraceContext, layer_idx: int):
    def hook(module, inp, out):
        # self_attn typically returns (attn_output, attn_weights)
        main = out[0] if isinstance(out, (tuple, list)) else out  # (T,B,C)

        if ctx.mode == "clean":
            ctx.clean_cache_attn[layer_idx] = main.detach()
            return out

        if ctx.mode == "patch":
            assert ctx.patch_cfg is not None
            if ctx.patch_cfg.layer_idx == layer_idx and ctx.patch_cfg.module == "attn":
                clean = ctx.clean_cache_attn[layer_idx].to(main.dtype).to(main.device)

                if ctx.patch_cfg.scope == "cls":
                    patched_main = main.clone()
                    patched_main[0:1, :, :] = clean[0:1, :, :]
                elif ctx.patch_cfg.scope == "all":
                    patched_main = clean
                else:
                    raise ValueError("scope must be 'cls' or 'all'")

                if isinstance(out, (tuple, list)):
                    return (patched_main, *out[1:])
                return patched_main

        return out
    return hook


def make_ffn_hook(ctx: TraceContext, layer_idx: int):
    def hook(module, inp, out):
        # out: fc2 output, usually (T, B, C)
        if ctx.mode == "clean":
            ctx.clean_cache_ffn[layer_idx] = out.detach()
            return out

        if ctx.mode == "patch":
            assert ctx.patch_cfg is not None
            if ctx.patch_cfg.layer_idx == layer_idx and ctx.patch_cfg.module == "ffn":
                clean = ctx.clean_cache_ffn[layer_idx].to(out.dtype).to(out.device)
                if ctx.patch_cfg.scope == "cls":
                    patched = out.clone()
                    patched[0:1, :, :] = clean[0:1, :, :]
                    return patched
                elif ctx.patch_cfg.scope == "all":
                    return clean
                else:
                    raise ValueError("scope must be 'cls' or 'all'")
        return out
    return hook


# -------------------------
# Metric: margin logit
# -------------------------
def margin_logit(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, C), labels: (B,)
    B, C = logits.shape
    correct = logits[torch.arange(B, device=logits.device), labels]  # (B,)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B, device=logits.device), labels] = False
    wrong_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return correct - wrong_max


def extract_tokens_labels(batch, device):
    tokens = batch["net_input"]["src_tokens"].to(device)              # (B, T)
    labels = batch["target"].squeeze(-1).to(device)                   # (B,)
    return tokens, labels


# -------------------------
# Main: run Experiment B
# -------------------------
@torch.no_grad()
def run_experiment_B(
    model,
    dataloader,
    corrupt_cfg: CorruptCfg,
    scope: str = "cls",          # "cls" or "all"
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
      effect:   (12,2) absolute effect
      recovery: (12,2) normalized effect / (clean-corr drop)
      avg_drop: scalar float, mean(clean_margin - corr_margin) across processed batches
    """
    model.eval().to(device)

    sent_enc = get_sentence_encoder(model)
    layers = sent_enc.layers
    n_layers = len(layers)
    assert n_layers == 12, f"Expected 12 layers, got {n_layers}"

    # install hooks
    ctx = TraceContext()
    ctx.corrupt_cfg = corrupt_cfg

    handles = []

    embed_mod = get_embed_module(sent_enc)
    handles.append(embed_mod.register_forward_hook(make_embed_hook(ctx)))

    for i in range(n_layers):
        if not hasattr(layers[i], "self_attn"):
            raise AttributeError(f"Layer {i} has no attribute self_attn. Please print dir(layers[i]).")
        handles.append(layers[i].self_attn.register_forward_hook(make_attn_hook(ctx, i)))

        if not hasattr(layers[i], "fc2"):
            raise AttributeError(f"Layer {i} has no attribute fc2. Please print dir(layers[i]).")
        handles.append(layers[i].fc2.register_forward_hook(make_ffn_hook(ctx, i)))

    sum_effect = torch.zeros(n_layers, 2, device="cpu")
    sum_recovery = torch.zeros(n_layers, 2, device="cpu")
    sum_drop = 0.0
    count = 0
    eps = 1e-12

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        tokens, labels = extract_tokens_labels(batch, device)

        # ---- Clean run (for filtering)
        ctx.clear_for_new_clean_run()
        ctx.mode = "clean"
        clean_logits = forward_logits(model, tokens)
        clean_pred = clean_logits.argmax(dim=-1)
        keep = (clean_pred == labels)
        if keep.sum().item() == 0:
            continue

        tokens_k = tokens[keep]
        labels_k = labels[keep]

        # re-run clean on filtered samples to align caches with used subset
        ctx.clear_for_new_clean_run()
        ctx.mode = "clean"
        clean_logits_k = forward_logits(model, tokens_k)
        clean_margin = margin_logit(clean_logits_k, labels_k)

        # ---- Corrupted run (sample & cache noise here; reused by all patch forwards)
        ctx.mode = "corrupt"
        corr_logits = forward_logits(model, tokens_k)
        corr_margin = margin_logit(corr_logits, labels_k)

        drop = (clean_margin - corr_margin).mean().item()
        sum_drop += drop

        # ---- Patched runs
        denom = drop + eps
        for layer_idx in range(n_layers):
            for module_idx, module_name in enumerate(["attn", "ffn"]):
                ctx.mode = "patch"
                ctx.patch_cfg = PatchCfg(layer_idx=layer_idx, module=module_name, scope=scope)

                patched_logits = forward_logits(model, tokens_k)
                patched_margin = margin_logit(patched_logits, labels_k)

                effect = (patched_margin - corr_margin).mean().item()
                recovery = effect / denom

                sum_effect[layer_idx, module_idx] += effect
                sum_recovery[layer_idx, module_idx] += recovery

        count += 1

    for h in handles:
        h.remove()

    if count == 0:
        raise RuntimeError("No valid batches processed (maybe clean accuracy too low?).")

    effect = sum_effect / count
    recovery = sum_recovery / count
    avg_drop = sum_drop / count
    return effect, recovery, avg_drop


# -------------------------
# Example usage
# -------------------------
def build_valid_iterator(task, seed=1, max_sentences=16):
    split = "valid"
    task.load_dataset(split)
    dataset = task.dataset(split)
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_sentences=max_sentences,
        max_tokens=None,
        seed=seed,
        num_workers=0,
    ).next_epoch_itr(shuffle=False)
    return itr


if __name__ == "__main__":
    # ---- reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_path = "log_dir/nodp/checkpoint_best.pt"

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrupt_cfg = CorruptCfg(alpha=0.3, exclude_cls=True)

    # B-CLS
    itr1 = build_valid_iterator(task, seed=1, max_sentences=16)
    effect_cls, recovery_cls, avg_drop_cls = run_experiment_B(
        model, itr1, corrupt_cfg, scope="cls", max_batches=50, device=device
    )
    print("=== B-CLS ===")
    print("avg drop(clean-corr):", avg_drop_cls)
    print("Effect (shape):", tuple(effect_cls.shape))
    print(effect_cls)
    print("Recovery (shape):", tuple(recovery_cls.shape))
    print(recovery_cls)

    # B-ALL
    itr2 = build_valid_iterator(task, seed=1, max_sentences=16)
    effect_all, recovery_all, avg_drop_all = run_experiment_B(
        model, itr2, corrupt_cfg, scope="all", max_batches=50, device=device
    )
    print("=== B-ALL ===")
    print("avg drop(clean-corr):", avg_drop_all)
    print("Effect (shape):", tuple(effect_all.shape))
    print(effect_all)
    print("Recovery (shape):", tuple(recovery_all.shape))
    print(recovery_all)
