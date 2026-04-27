import sys
sys.path.insert(0, ".")  # use in-project fairseq

import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

# IMPORTANT: disable roberta upgrade (avoids dense-related crash)
import fairseq.models.roberta.model as roberta_model
roberta_model.RobertaModel.upgrade_state_dict_named = lambda self, state_dict, name: None

from fairseq import checkpoint_utils
from transformers import RobertaTokenizerFast

hf_tok = RobertaTokenizerFast.from_pretrained("roberta-base")

SPECIAL = {
    "<s>": hf_tok.bos_token_id,
    "</s>": hf_tok.eos_token_id,
    "<pad>": hf_tok.pad_token_id,
    "<unk>": hf_tok.unk_token_id,
}


# -------------------------
# Configs
# -------------------------
@dataclass
class CorruptCfg:
    alpha: float = 0.3
    mode: str = "span_random"  # all | token_random | span_random | sentiment | sentiment_or_span
    span_len: int = 3
    exclude_cls: bool = True
    exclude_special: bool = True


@dataclass
class PatchCfg:
    layer_idx: int                 # 0..11
    module: str                    # "attn" or "ffn"
    scope: str                     # "cls" | "all" | "focus" | "focus+cls"
    source: str                    # "clean" | "corr"


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
    if hasattr(sent_enc, "layernorm_embedding"):
        return sent_enc.layernorm_embedding
    if hasattr(sent_enc, "embed_tokens"):
        return sent_enc.embed_tokens
    raise AttributeError("Cannot find embedding module (layernorm_embedding/embed_tokens).")


def forward_logits(model, tokens):
    out = model(tokens, classification_head_name="sentence_classification_head")
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(logits):
        raise RuntimeError(f"Cannot parse logits from output type: {type(out)}")
    return logits


# -------------------------
# Context
# -------------------------
class TraceContext:
    def __init__(self, pad_idx: int, bos_idx: int, eos_idx: int):
        self.mode = "clean"  # clean | corrupt | patch
        self.add_noise: bool = False  # explicit: whether embedding hook should add noise

        self.patch_cfg: Optional[PatchCfg] = None
        self.corrupt_cfg: Optional[CorruptCfg] = None

        # token ids for current forward (B,T)
        self.tokens_bt: Optional[torch.Tensor] = None

        # focus mask (B,T)
        self.focus_mask_bt: Optional[torch.Tensor] = None

        # corruption mask actually used in embedding noise (B,T)
        self.corrupt_mask_bt: Optional[torch.Tensor] = None

        # patch mask used in module outputs (T,B)
        self.patch_mask_tb: Optional[torch.Tensor] = None

        # caches
        self.clean_cache_attn: Dict[int, torch.Tensor] = {}
        self.clean_cache_ffn: Dict[int, torch.Tensor] = {}
        self.corr_cache_attn: Dict[int, torch.Tensor] = {}
        self.corr_cache_ffn: Dict[int, torch.Tensor] = {}

        # embedding noise stats
        self.emb_std: Optional[torch.Tensor] = None
        self.emb_noise: Optional[torch.Tensor] = None

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def clear_for_new_example_set(self):
        self.clean_cache_attn.clear()
        self.clean_cache_ffn.clear()
        self.corr_cache_attn.clear()
        self.corr_cache_ffn.clear()
        self.emb_std = None
        self.emb_noise = None
        self.patch_cfg = None
        self.tokens_bt = None
        self.focus_mask_bt = None
        self.corrupt_mask_bt = None
        self.patch_mask_tb = None
        self.mode = "clean"
        self.add_noise = False


# -------------------------
# Mask builders
# -------------------------
def build_valid_content_mask_bt(
    tokens_bt: torch.Tensor,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    exclude_cls: bool,
    exclude_special: bool,
) -> torch.Tensor:
    """
    tokens_bt: (B,T)
    return: (B,T) bool mask of content positions eligible for corruption/selection.
    """
    mask = torch.ones_like(tokens_bt, dtype=torch.bool)

    # exclude padding
    mask &= (tokens_bt != pad_idx)

    if exclude_special:
        mask &= (tokens_bt != bos_idx)
        mask &= (tokens_bt != eos_idx)

    if exclude_cls:
        # RoBERTa uses <s> at position 0 as CLS
        mask[:, 0] = False

    return mask


def choose_random_token_positions(mask_bt: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    B, T = mask_bt.shape
    out = torch.zeros_like(mask_bt, dtype=torch.bool)
    for b in range(B):
        idx = torch.nonzero(mask_bt[b], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        j = idx[torch.randint(low=0, high=idx.numel(), size=(1,), generator=g).item()].item()
        out[b, j] = True
    return out


def choose_random_span_positions(mask_bt: torch.Tensor, span_len: int, g: torch.Generator) -> torch.Tensor:
    B, T = mask_bt.shape
    out = torch.zeros_like(mask_bt, dtype=torch.bool)
    for b in range(B):
        idx = torch.nonzero(mask_bt[b], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        L = idx.numel()
        k = min(span_len, L)
        start_offset = torch.randint(0, max(L - k, 0) + 1, (1,), generator=g).item()
        span_positions = idx[start_offset:start_offset + k]
        out[b, span_positions] = True
    return out


def fairseq_sym_to_roberta_id(sym: str) -> int:
    if sym in SPECIAL:
        return SPECIAL[sym]
    if sym.isdigit():
        return int(sym)
    rid = hf_tok.convert_tokens_to_ids(sym)
    if rid is None:
        return hf_tok.unk_token_id
    return int(rid)


def build_sentiment_mask_bt(
    tokens_bt: torch.Tensor,
    eligible_mask_bt: torch.Tensor,
    senti_words: set,
    src_dict=None,
) -> torch.Tensor:
    """
    Returns (B,T) bool mask: True where token (after BPE normalization) is in senti_words.
    """
    assert src_dict is not None, "Need src_dict to map fairseq index -> symbol string"
    B, T = tokens_bt.shape
    out = torch.zeros_like(tokens_bt, dtype=torch.bool)

    for b in range(B):
        roberta_ids = []
        for t in range(T):
            sym = src_dict[tokens_bt[b, t].item()]
            roberta_ids.append(fairseq_sym_to_roberta_id(sym))

        bpe_tokens = hf_tok.convert_ids_to_tokens(roberta_ids)

        for t, sym in enumerate(bpe_tokens):
            if not eligible_mask_bt[b, t].item():
                continue
            w = sym.lstrip("Ġ").lower().strip(".,!?;:\"'()[]{}")
            if w in senti_words:
                out[b, t] = True

    return out


def build_focus_mask_bt(
    tokens_bt: torch.Tensor,
    cfg: CorruptCfg,
    src_dict,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    senti_words: Set[str],
    g: torch.Generator,
) -> torch.Tensor:
    eligible = build_valid_content_mask_bt(
        tokens_bt, pad_idx, bos_idx, eos_idx,
        exclude_cls=cfg.exclude_cls,
        exclude_special=cfg.exclude_special,
    )

    if cfg.mode == "all":
        return eligible

    if cfg.mode == "token_random":
        return choose_random_token_positions(eligible, g)

    if cfg.mode == "span_random":
        return choose_random_span_positions(eligible, cfg.span_len, g)

    if cfg.mode == "sentiment":
        return build_sentiment_mask_bt(tokens_bt, eligible, senti_words, src_dict=src_dict)

    if cfg.mode == "sentiment_or_span":
        sm = build_sentiment_mask_bt(tokens_bt, eligible, senti_words, src_dict=src_dict)
        hit = sm.any(dim=1)  # (B,)
        rand = choose_random_span_positions(eligible, cfg.span_len, g)
        out = sm.clone()
        out[~hit] = rand[~hit]
        return out

    raise ValueError(f"Unknown corrupt mode: {cfg.mode}")


def build_patch_mask_tb(scope: str, focus_mask_bt: torch.Tensor, device) -> torch.Tensor:
    """
    scope: cls | all | focus | focus+cls
    Returns mask_tb: (T,B) bool
    """
    B, T = focus_mask_bt.shape
    if scope == "all":
        mask_bt = torch.ones((B, T), dtype=torch.bool, device=device)
    elif scope == "cls":
        mask_bt = torch.zeros((B, T), dtype=torch.bool, device=device)
        mask_bt[:, 0] = True
    elif scope == "focus":
        mask_bt = focus_mask_bt.to(device=device)
    elif scope == "focus+cls":
        mask_bt = focus_mask_bt.to(device=device).clone()
        mask_bt[:, 0] = True
    else:
        raise ValueError("scope must be one of: cls | all | focus | focus+cls")
    return mask_bt.transpose(0, 1).contiguous()  # (T,B)


# -------------------------
# Hooks
# -------------------------
def make_embed_hook(ctx: TraceContext):
    def hook(module, inp, out):
        # out: (B,T,C)
        assert ctx.tokens_bt is not None, "ctx.tokens_bt must be set before forward"

        if ctx.mode == "clean":
            # compute emb_std over content tokens to scale noise
            tokens = ctx.tokens_bt
            content_mask = build_valid_content_mask_bt(
                tokens, ctx.pad_idx, ctx.bos_idx, ctx.eos_idx,
                exclude_cls=True, exclude_special=True
            ).to(out.device)

            if content_mask.any().item():
                x = out[content_mask]  # (N,C)
                ctx.emb_std = x.detach().float().std()
            else:
                ctx.emb_std = out.detach().float().std()

            ctx.emb_noise = None
            return out

        # non-clean modes:
        if not ctx.add_noise:
            return out

        assert ctx.corrupt_cfg is not None
        assert ctx.emb_std is not None, "emb_std not set; run clean forward first."
        assert ctx.corrupt_mask_bt is not None, "corrupt_mask_bt not set."

        cmask = ctx.corrupt_mask_bt.to(device=out.device)  # (B,T) bool
        if not cmask.any().item():
            return out

        # reuse same noise tensor if possible (important for restore experiments)
        if (ctx.emb_noise is None) or (ctx.emb_noise.shape != out.shape) or (ctx.emb_noise.device != out.device):
            sigma = (ctx.corrupt_cfg.alpha * ctx.emb_std).to(out.device)
            noise = torch.randn_like(out) * sigma.to(out.dtype)
            noise = noise * cmask.unsqueeze(-1).to(noise.dtype)
            ctx.emb_noise = noise

        return out + ctx.emb_noise

    return hook


def make_attn_hook(ctx: TraceContext, layer_idx: int):
    def hook(module, inp, out):
        main = out[0] if isinstance(out, (tuple, list)) else out  # (T,B,C)

        if ctx.mode == "clean":
            ctx.clean_cache_attn[layer_idx] = main.detach()
            return out

        if ctx.mode == "corrupt":
            ctx.corr_cache_attn[layer_idx] = main.detach()
            return out

        if ctx.mode == "patch":
            assert ctx.patch_cfg is not None
            if ctx.patch_cfg.layer_idx == layer_idx and ctx.patch_cfg.module == "attn":
                assert ctx.patch_mask_tb is not None
                if ctx.patch_cfg.source == "clean":
                    cache = ctx.clean_cache_attn[layer_idx]
                elif ctx.patch_cfg.source == "corr":
                    cache = ctx.corr_cache_attn[layer_idx]
                else:
                    raise ValueError("patch source must be clean|corr")

                cache = cache.to(main.dtype).to(main.device)
                mask = ctx.patch_mask_tb.to(device=main.device)  # (T,B)

                patched_main = torch.where(mask.unsqueeze(-1), cache, main)

                if isinstance(out, (tuple, list)):
                    return (patched_main, *out[1:])
                return patched_main

        return out

    return hook


def make_ffn_hook(ctx: TraceContext, layer_idx: int):
    def hook(module, inp, out):
        # out: fc2 output, usually (T,B,C)
        if ctx.mode == "clean":
            ctx.clean_cache_ffn[layer_idx] = out.detach()
            return out

        if ctx.mode == "corrupt":
            ctx.corr_cache_ffn[layer_idx] = out.detach()
            return out

        if ctx.mode == "patch":
            assert ctx.patch_cfg is not None
            if ctx.patch_cfg.layer_idx == layer_idx and ctx.patch_cfg.module == "ffn":
                assert ctx.patch_mask_tb is not None
                if ctx.patch_cfg.source == "clean":
                    cache = ctx.clean_cache_ffn[layer_idx]
                elif ctx.patch_cfg.source == "corr":
                    cache = ctx.corr_cache_ffn[layer_idx]
                else:
                    raise ValueError("patch source must be clean|corr")

                cache = cache.to(out.dtype).to(out.device)
                mask = ctx.patch_mask_tb.to(device=out.device)  # (T,B)
                patched = torch.where(mask.unsqueeze(-1), cache, out)
                return patched

        return out

    return hook


# -------------------------
# Metric: margin logit
# -------------------------
def margin_logit(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B, C = logits.shape
    correct = logits[torch.arange(B, device=logits.device), labels]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B, device=logits.device), labels] = False
    wrong_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return correct - wrong_max


def extract_tokens_labels(batch, device):
    tokens = batch["net_input"]["src_tokens"].to(device)  # (B,T)
    labels = batch["target"].squeeze(-1).to(device)       # (B,)
    return tokens, labels


def slice_cache_tbC(cache_tbC: torch.Tensor, good_b: torch.Tensor) -> torch.Tensor:
    # cache_tbC: (T,B,C), good_b: (B,) bool on any device
    good_cpu = good_b.detach().to("cpu")
    return cache_tbC[:, good_cpu, :].contiguous()


# -------------------------
# Experiment runner
# -------------------------
@torch.no_grad()
def run_experiment_B_focus(
    model,
    dataloader,
    src_dict,
    corrupt_cfg: CorruptCfg,
    patch_scope: str = "focus",      # cls | all | focus | focus+cls
    direction: str = "restore",      # restore | block
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Returns:
      effect_mean: (12,2) mean effect per sample
      recovery:    (12,2) effect_sum / drop_sum  (sample-weighted)
      avg_drop:    scalar float, mean drop per sample
      keep_rate:   scalar float, mean keep ratio per batch
    """
    model.eval().to(device)

    sent_enc = get_sentence_encoder(model)
    layers = sent_enc.layers
    n_layers = len(layers)
    assert n_layers == 12, f"Expected 12 layers, got {n_layers}"

    pad_idx = src_dict.pad()
    bos_idx = src_dict.bos()
    eos_idx = src_dict.eos()

    SENTI_WORDS = {
        # positive
        "good", "great", "excellent", "amazing", "wonderful", "best", "love", "loved",
        "like", "liked", "enjoy", "enjoyed", "fun", "funny", "brilliant", "perfect",
        # negative
        "bad", "terrible", "awful", "worst", "boring", "hate", "hated", "poor",
        "disappointing", "disappointed", "dull", "waste", "ridiculous",
        # negation
        "not", "never", "no", "n't",
    }

    ctx = TraceContext(pad_idx=pad_idx, bos_idx=bos_idx, eos_idx=eos_idx)
    ctx.corrupt_cfg = corrupt_cfg

    handles = []
    embed_mod = get_embed_module(sent_enc)
    handles.append(embed_mod.register_forward_hook(make_embed_hook(ctx)))
    for i in range(n_layers):
        handles.append(layers[i].self_attn.register_forward_hook(make_attn_hook(ctx, i)))
        handles.append(layers[i].fc2.register_forward_hook(make_ffn_hook(ctx, i)))

    effect_sum = torch.zeros((n_layers, 2), dtype=torch.float64)  # CPU
    drop_sum = 0.0
    n_sum = 0

    keep_rate_sum = 0.0
    keep_batches = 0
    eps = 1e-12

    try:
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            tokens, labels = extract_tokens_labels(batch, device)

            # (1) clean run for filtering (keep only clean-correct samples)
            ctx.clear_for_new_example_set()
            ctx.tokens_bt = tokens
            ctx.mode = "clean"
            ctx.add_noise = False
            clean_logits = forward_logits(model, tokens)
            clean_pred = clean_logits.argmax(dim=-1)
            keep = (clean_pred == labels)

            keep_rate_sum += keep.float().mean().item()
            keep_batches += 1
            if keep.sum().item() == 0:
                continue

            tokens_k = tokens[keep]
            labels_k = labels[keep]

            # build focus mask on filtered batch (CPU for dictionary mapping)
            g2 = torch.Generator(device="cpu")
            g2.manual_seed(10000 + b_idx)

            focus_mask_bt_cpu = build_focus_mask_bt(
                tokens_k.detach().cpu(),
                corrupt_cfg,
                src_dict,
                pad_idx=pad_idx,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                senti_words=SENTI_WORDS,
                g=g2,
            )
            focus_mask_bt = focus_mask_bt_cpu.to(device=device)

            # set masks (for subsequent forwards)
            ctx.focus_mask_bt = focus_mask_bt
            ctx.corrupt_mask_bt = focus_mask_bt
            ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_mask_bt, device=device)

            # (2) clean run for caching + clean margin
            ctx.clean_cache_attn.clear()
            ctx.clean_cache_ffn.clear()
            ctx.emb_std = None
            ctx.emb_noise = None

            ctx.tokens_bt = tokens_k
            ctx.mode = "clean"
            ctx.add_noise = False
            clean_logits_k = forward_logits(model, tokens_k)
            clean_margin = margin_logit(clean_logits_k, labels_k)  # (B,)

            # (3) corrupt run for corr margin + corr caches
            ctx.corr_cache_attn.clear()
            ctx.corr_cache_ffn.clear()

            ctx.tokens_bt = tokens_k
            ctx.mode = "corrupt"
            ctx.add_noise = True
            corr_logits_k = forward_logits(model, tokens_k)
            corr_margin = margin_logit(corr_logits_k, labels_k)  # (B,)

            drop_vec = (clean_margin - corr_margin)
            good = drop_vec > 0
            if good.sum().item() == 0:
                continue

            # slice tensors to good subset
            tokens_g = tokens_k[good]
            labels_g = labels_k[good]
            clean_mg = clean_margin[good]
            corr_mg = corr_margin[good]
            drop_g = drop_vec[good]
            focus_g = focus_mask_bt[good]

            # slice caches and noise to good subset (avoid extra forward)
            for i in range(n_layers):
                ctx.clean_cache_attn[i] = slice_cache_tbC(ctx.clean_cache_attn[i], good)
                ctx.corr_cache_attn[i] = slice_cache_tbC(ctx.corr_cache_attn[i], good)
                ctx.clean_cache_ffn[i] = slice_cache_tbC(ctx.clean_cache_ffn[i], good)
                ctx.corr_cache_ffn[i] = slice_cache_tbC(ctx.corr_cache_ffn[i], good)

            if ctx.emb_noise is not None:
                ctx.emb_noise = ctx.emb_noise[good].contiguous()

            # update ctx for patch runs
            ctx.tokens_bt = tokens_g
            ctx.focus_mask_bt = focus_g
            ctx.corrupt_mask_bt = focus_g
            ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_g, device=device)

            drop_sum += float(drop_g.sum().item())
            n_sum += int(drop_g.numel())

            # (4) patched runs
            for layer_idx in range(n_layers):
                for module_idx, module_name in enumerate(["attn", "ffn"]):
                    ctx.mode = "patch"
                    ctx.tokens_bt = tokens_g

                    if direction == "restore":
                        # corrupt input (noise ON), patch in clean activations
                        ctx.add_noise = True
                        ctx.patch_cfg = PatchCfg(layer_idx, module_name, patch_scope, source="clean")
                        patched_logits = forward_logits(model, tokens_g)
                        patched_mg = margin_logit(patched_logits, labels_g)
                        eff_vec = (patched_mg - corr_mg)
                        effect_sum[layer_idx, module_idx] += eff_vec.detach().double().cpu().sum()

                    elif direction == "block":
                        # clean input (noise OFF), patch in corrupt activations
                        ctx.add_noise = False
                        ctx.patch_cfg = PatchCfg(layer_idx, module_name, patch_scope, source="corr")
                        blocked_logits = forward_logits(model, tokens_g)
                        blocked_mg = margin_logit(blocked_logits, labels_g)
                        eff_vec = (clean_mg - blocked_mg)
                        effect_sum[layer_idx, module_idx] += eff_vec.detach().double().cpu().sum()
                    else:
                        raise ValueError("direction must be restore|block")

    finally:
        for h in handles:
            h.remove()

    if n_sum == 0:
        raise RuntimeError("No valid samples processed (maybe corruption too weak/too strong or keep too low).")

    effect_mean = (effect_sum / n_sum).float()
    avg_drop = drop_sum / n_sum
    recovery = (effect_sum / max(drop_sum, eps)).float()
    keep_rate = keep_rate_sum / max(keep_batches, 1)

    return effect_mean, recovery, avg_drop, keep_rate


# -------------------------
# Iterator
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
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_path = "log_dir/nodp/checkpoint_best.pt"
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_dict = task.source_dictionary

    # 固定一些公共超参
    alpha = 0.3
    span_len = 3
    exclude_cls = True
    exclude_special = True

    # 你要跑的四个实验配置
    EXPERIMENTS = [
        dict(
            name='E1_restore_focus_sentiment_or_span (恢复被破坏的关键信息)',
            mode="sentiment_or_span",
            patch_scope="focus",
            direction="restore",
        ),
        dict(
            name='E2_restore_cls_sentiment_or_span (信息是否主要写入CLS)',
            mode="sentiment_or_span",
            patch_scope="cls",
            direction="restore",
        ),
        dict(
            name='E3_block_focus_sentiment_or_span (阻断验证/注入验证)',
            mode="sentiment_or_span",
            patch_scope="focus",
            direction="block",
        ),
        dict(
            name='E4_restore_focus_span_random (对照：排除随机span可恢复效应)',
            mode="span_random",
            patch_scope="focus",
            direction="restore",
        ),
    ]

    # 运行控制
    max_sentences = 16
    seed = 1
    max_batches = 50

    all_results = {}

    for exp in EXPERIMENTS:
        corrupt_cfg = CorruptCfg(
            alpha=alpha,
            mode=exp["mode"],
            span_len=span_len,
            exclude_cls=exclude_cls,
            exclude_special=exclude_special,
        )

        # 重要：每个实验都要重新build iterator，否则第一个实验跑完迭代器就没了
        itr = build_valid_iterator(task, seed=seed, max_sentences=max_sentences)

        effect, recovery, avg_drop, keep_rate = run_experiment_B_focus(
            model=model,
            dataloader=itr,
            src_dict=src_dict,
            corrupt_cfg=corrupt_cfg,
            patch_scope=exp["patch_scope"],
            direction=exp["direction"],
            max_batches=max_batches,
            device=device,
        )

        all_results[exp["name"]] = dict(
            corrupt_cfg=corrupt_cfg,
            patch_scope=exp["patch_scope"],
            direction=exp["direction"],
            keep_rate=keep_rate,
            avg_drop=avg_drop,
            effect=effect,       # (12,2) [attn, ffn]
            recovery=recovery,   # (12,2) [attn, ffn]
        )

        print("\n" + "=" * 80)
        print("EXPERIMENT:", exp["name"])
        print("corrupt_cfg =", corrupt_cfg)
        print("patch_scope =", exp["patch_scope"])
        print("direction   =", exp["direction"])
        print("keep_rate(clean correct filter) =", keep_rate)
        print("avg_drop(clean-corr, per-sample) =", avg_drop)
        print("Effect mean shape:", tuple(effect.shape), " [cols: attn, ffn]")
        print(effect)
        print("Recovery shape:", tuple(recovery.shape), " [cols: attn, ffn]")
        print(recovery)

    # 如需保存（可选）
    # torch.save(all_results, "four_experiments_results.pt")
    print("\nDONE. Collected results keys:", list(all_results.keys()))
