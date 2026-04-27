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
tok = RobertaTokenizerFast.from_pretrained("roberta-base")
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
    mode: str = "span_random"  # "all" | "token_random" | "span_random" | "sentiment" | "sentiment_or_span"
    span_len: int = 3
    exclude_cls: bool = True
    exclude_special: bool = True


@dataclass
class PatchCfg:
    layer_idx: int                 # 0..11
    module: str                    # "attn" or "ffn"
    scope: str                     # "cls" | "all" | "focus" | "focus+cls"
    source: str                    # "clean" | "corr"  (restore uses clean; block uses corr)


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
        self.mode = "clean"  # "clean" | "corrupt" | "patch"
        self.patch_cfg: Optional[PatchCfg] = None
        self.corrupt_cfg: Optional[CorruptCfg] = None

        # token ids for current forward (B,T)
        self.tokens_bt: Optional[torch.Tensor] = None

        # focus mask (B,T): where we corrupt / where "focus" patch applies
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
        # clear caches + embedding noise stats; keep corrupt_cfg
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


# -------------------------
# Mask builders
# -------------------------
def build_valid_content_mask_bt(tokens_bt: torch.Tensor, pad_idx: int, bos_idx: int, eos_idx: int,
                                exclude_cls: bool, exclude_special: bool) -> torch.Tensor:
    """
    tokens_bt: (B,T)
    return mask_bt: (B,T) content positions eligible for corruption/selection.
    """
    B, T = tokens_bt.shape
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

    return out  # <- 必须在for外


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
        span_positions = idx[start_offset:start_offset+k]
        out[b, span_positions] = True

    return out  # <- 必须在for外

def build_sentiment_mask_bt(tokens_bt: torch.Tensor,
                            eligible_mask_bt: torch.Tensor,
                            senti_words: set,
                            src_dict=None) -> torch.Tensor:
    assert src_dict is not None, "Need src_dict to map fairseq index -> symbol string" 
    B, T = tokens_bt.shape
    out = torch.zeros_like(tokens_bt, dtype=torch.bool)

    for b in range(B):
        roberta_ids = []
        for t in range(T):
            sym = src_dict[tokens_bt[b, t].item()]  # "<s>" 或 "262" 这种
            if sym in SPECIAL:
                rid = SPECIAL[sym]
            elif sym.isdigit():
                rid = int(sym)  # 关键：把数字字符串转回 roberta vocab id
            else:
                rid = hf_tok.convert_tokens_to_ids(sym)  # 兜底
            roberta_ids.append(rid)  # <- 必须每个token都append

        bpe_tokens = hf_tok.convert_ids_to_tokens(roberta_ids)

        for t, sym in enumerate(bpe_tokens):
            if not eligible_mask_bt[b, t].item():
                continue
            w = sym.lstrip("Ġ").lower().strip(".,!?;:\"'()[]{}")
            if w in senti_words:
                out[b, t] = True

    return out  # <- 必须在for b外



def build_focus_mask_bt(tokens_bt: torch.Tensor, cfg: CorruptCfg, src_dict, pad_idx: int, bos_idx: int, eos_idx: int,
                        senti_words: Set[str], g: torch.Generator) -> torch.Tensor:
    eligible = build_valid_content_mask_bt(tokens_bt, pad_idx, bos_idx, eos_idx,
                                          exclude_cls=cfg.exclude_cls,
                                          exclude_special=cfg.exclude_special)

    if cfg.mode == "all":
        return eligible

    if cfg.mode == "token_random":
        return choose_random_token_positions(eligible, g)

    if cfg.mode == "span_random":
        return choose_random_span_positions(eligible, cfg.span_len, g)

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
        if ctx.mode == "clean":
            # compute emb_std over content tokens to scale noise
            assert ctx.tokens_bt is not None, "ctx.tokens_bt must be set before forward"
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

        # corrupt / patch: add noise only on ctx.corrupt_mask_bt (reuse same noise)
        assert ctx.corrupt_cfg is not None
        assert ctx.emb_std is not None, "emb_std not set; run clean forward first."
        assert ctx.corrupt_mask_bt is not None, "corrupt_mask_bt not set."

        cmask = ctx.corrupt_mask_bt.to(device=out.device)  # (B,T) bool
        if not cmask.any().item():
            return out
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
    tokens = batch["net_input"]["src_tokens"].to(device)              # (B, T)
    labels = batch["target"].squeeze(-1).to(device)                   # (B,)
    return tokens, labels


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
      keep_rate:   scalar float, mean keep ratio per batch (clean correct filtering)
    """
    model.eval().to(device)

    sent_enc = get_sentence_encoder(model)
    layers = sent_enc.layers
    n_layers = len(layers)
    assert n_layers == 12, f"Expected 12 layers, got {n_layers}"

    pad_idx = src_dict.pad()
    bos_idx = src_dict.bos()
    eos_idx = src_dict.eos()

    # very small sentiment lexicon (you can expand)
    SENTI_WORDS = {
        # positive
        "good", "great", "excellent", "amazing", "wonderful", "best", "love", "loved", "like", "liked",
        "enjoy", "enjoyed", "fun", "funny", "brilliant", "perfect",
        # negative
        "bad", "terrible", "awful", "worst", "boring", "hate", "hated", "poor", "disappointing", "disappointed",
        "dull", "waste", "ridiculous",
        # negation (often crucial)
        "not", "never", "no",
    }

    # install hooks
    ctx = TraceContext(pad_idx=pad_idx, bos_idx=bos_idx, eos_idx=eos_idx)
    ctx.corrupt_cfg = corrupt_cfg

    handles = []
    embed_mod = get_embed_module(sent_enc)
    handles.append(embed_mod.register_forward_hook(make_embed_hook(ctx)))

    for i in range(n_layers):
        handles.append(layers[i].self_attn.register_forward_hook(make_attn_hook(ctx, i)))
        handles.append(layers[i].fc2.register_forward_hook(make_ffn_hook(ctx, i)))

    # stats accumulators (sample-weighted)
    effect_sum = torch.zeros(n_layers, 2, device="cpu")  # sum over samples
    drop_sum = 0.0
    n_sum = 0

    keep_rate_sum = 0.0
    keep_batches = 0

    g = torch.Generator(device="cpu")
    g.manual_seed(1234)

    eps = 1e-12
    sent_hit_sum = 0.0     # 命中 sentiment 的样本数（或比例累加）
    sent_hit_n = 0         # 统计样本总数
    sent_fallback_sum = 0.0  # sentiment_or_span 下发生 fallback 的样本数


    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        tokens, labels = extract_tokens_labels(batch, device)

        # ---- (1) clean run for filtering
        ctx.clear_for_new_example_set()
        ctx.tokens_bt = tokens
        ctx.mode = "clean"
        clean_logits = forward_logits(model, tokens)
        clean_pred = clean_logits.argmax(dim=-1)
        keep = (clean_pred == labels)

        keep_rate_sum += keep.float().mean().item()
        keep_batches += 1

        if keep.sum().item() == 0:
            continue

        tokens_k = tokens[keep]
        labels_k = labels[keep]

        # build focus mask on filtered batch
        # make generator deterministic per batch
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(10000 + b_idx)

        focus_mask_bt = build_focus_mask_bt(
            tokens_k.detach().cpu(), corrupt_cfg, src_dict,
            pad_idx=pad_idx, bos_idx=bos_idx, eos_idx=eos_idx,
            senti_words=SENTI_WORDS, g=g2
        ).to(device)
        # build focus mask on filtered batch
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(10000 + b_idx)

        tokens_k_cpu = tokens_k.detach().cpu()
        focus_mask_bt_cpu = build_focus_mask_bt(
            tokens_k_cpu, corrupt_cfg, src_dict,
            pad_idx=pad_idx, bos_idx=bos_idx, eos_idx=eos_idx,
            senti_words=SENTI_WORDS, g=g2
        )  # CPU bool (B,T)

        # ---- sentiment hit-rate debug (BEFORE fallback / or at least measurable)
        if corrupt_cfg.mode in ("sentiment", "sentiment_or_span"):
            eligible_cpu = build_valid_content_mask_bt(
                tokens_k_cpu, pad_idx, bos_idx, eos_idx,
                exclude_cls=corrupt_cfg.exclude_cls,
                exclude_special=corrupt_cfg.exclude_special
            )
            sm_cpu = build_sentiment_mask_bt(tokens_k_cpu, eligible_cpu, SENTI_WORDS, src_dict=src_dict)
            hit_per_sample = sm_cpu.any(dim=1)  # (B,)
            sent_hit_sum += hit_per_sample.float().sum().item()
            sent_hit_n += hit_per_sample.numel()

            if corrupt_cfg.mode == "sentiment_or_span":
                # fallback happens when a sample has no sentiment hit
                sent_fallback_sum += (~hit_per_sample).float().sum().item()

            if b_idx < 3:
                sym_list = [src_dict[t.item()] for t in tokens_k_cpu[0]]
                roberta_ids = []
                for s in sym_list:
                    if s in ("<s>", "</s>", "<pad>", "<unk>"):
                        roberta_ids.append(hf_tok.convert_tokens_to_ids(s))
                    elif s.isdigit():
                        roberta_ids.append(int(s))
                    else:
                        roberta_ids.append(hf_tok.convert_tokens_to_ids(s))

                bpe = hf_tok.convert_ids_to_tokens(roberta_ids)
                norm = [x.lstrip("Ġ").lower().strip(".,!?;:\"'()[]{}") for x in bpe]
                print("bpe:", bpe[:40])
                print("hits:", [w for w in norm if w in SENTI_WORDS])
        sym_list = [src_dict[t.item()] for t in tokens_k_cpu[0]]
        clean_list = [s.replace("Ġ","").lower().strip(".,!?;:\"'()[]{}") for s in sym_list]
        print(clean_list)
        print("hits:", [w for w in clean_list if w in SENTI_WORDS])

        # move focus mask to device for actual run
        focus_mask_bt = focus_mask_bt_cpu.to(device)
        print("focus tokens per sample:", focus_mask_bt_cpu.sum(dim=1)[:5].tolist())
        # set masks in ctx
        ctx.focus_mask_bt = focus_mask_bt
        ctx.corrupt_mask_bt = focus_mask_bt
        ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_mask_bt, device=device)



        # set masks in ctx
        ctx.focus_mask_bt = focus_mask_bt
        ctx.corrupt_mask_bt = focus_mask_bt  # corruption happens only on focus positions
        ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_mask_bt, device=device)

        # ---- (2) clean run for caching + clean margin
        ctx.clean_cache_attn.clear()
        ctx.clean_cache_ffn.clear()
        ctx.emb_std = None
        ctx.emb_noise = None

        ctx.tokens_bt = tokens_k
        ctx.mode = "clean"
        clean_logits_k = forward_logits(model, tokens_k)
        clean_margin = margin_logit(clean_logits_k, labels_k)  # (B,)

        # ---- (3) corrupt run for corr margin + corr caches (for block)
        ctx.corr_cache_attn.clear()
        ctx.corr_cache_ffn.clear()

        ctx.tokens_bt = tokens_k
        ctx.mode = "corrupt"
        corr_logits = forward_logits(model, tokens_k)
        corr_margin = margin_logit(corr_logits, labels_k)      # (B,)

        # drop used for normalization (full corruption drop)
        drop_vec = (clean_margin - corr_margin)                # (B,)
        # keep only positive-drop samples for stability (optional but recommended)
        good = drop_vec > 0
        if good.sum().item() == 0:
            continue

        tokens_kg = tokens_k[good]
        labels_kg = labels_k[good]
        clean_mg = clean_margin[good]
        corr_mg = corr_margin[good]
        drop_g = drop_vec[good]

        # update masks for good subset
        focus_g = focus_mask_bt[good]
        ctx.focus_mask_bt = focus_g
        ctx.corrupt_mask_bt = focus_g
        ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_g, device=device)

        # IMPORTANT: caches correspond to tokens_k (before good filtering).
        # Recompute caches on good subset to align shapes.
        # clean caches:
        ctx.clean_cache_attn.clear()
        ctx.clean_cache_ffn.clear()
        ctx.emb_std = None
        ctx.emb_noise = None
        ctx.tokens_bt = tokens_kg
        ctx.mode = "clean"
        _ = forward_logits(model, tokens_kg)

        # corrupt caches (+ corr margin recompute to reuse SAME noise in patch if needed):
        ctx.corr_cache_attn.clear()
        ctx.corr_cache_ffn.clear()
        ctx.tokens_bt = tokens_kg
        ctx.mode = "corrupt"
        corr_logits_g = forward_logits(model, tokens_kg)
        corr_mg = margin_logit(corr_logits_g, labels_kg)
        drop_g = (clean_mg - corr_mg)
        good2 = drop_g > 0
        if good2.sum().item() == 0:
            continue

        tokens_kg = tokens_kg[good2]
        labels_kg = labels_kg[good2]
        clean_mg = clean_mg[good2]
        corr_mg = corr_mg[good2]
        drop_g = drop_g[good2]
        focus_g = focus_g[good2]
        ctx.focus_mask_bt = focus_g
        ctx.corrupt_mask_bt = focus_g
        ctx.patch_mask_tb = build_patch_mask_tb(patch_scope, focus_g, device=device)

        # Need caches again aligned with final subset:
        ctx.clean_cache_attn.clear()
        ctx.clean_cache_ffn.clear()
        ctx.emb_std = None
        ctx.emb_noise = None
        ctx.tokens_bt = tokens_kg
        ctx.mode = "clean"
        _ = forward_logits(model, tokens_kg)

        ctx.corr_cache_attn.clear()
        ctx.corr_cache_ffn.clear()
        ctx.tokens_bt = tokens_kg
        ctx.mode = "corrupt"
        corr_logits_g = forward_logits(model, tokens_kg)
        corr_mg = margin_logit(corr_logits_g, labels_kg)
        drop_g = (clean_mg - corr_mg)

        # accumulate normalization denominator
        drop_sum += drop_g.sum().item()
        n_sum += drop_g.numel()

        # ---- (4) patched runs
        for layer_idx in range(n_layers):
            for module_idx, module_name in enumerate(["attn", "ffn"]):
                ctx.mode = "patch"
                if direction == "restore":
                    # run on corrupt input (embedding noise ON), patch in clean activations
                    ctx.patch_cfg = PatchCfg(layer_idx, module_name, patch_scope, source="clean")
                    ctx.tokens_bt = tokens_kg
                    patched_logits = forward_logits(model, tokens_kg)  # still corrupt at embedding
                    patched_mg = margin_logit(patched_logits, labels_kg)
                    eff_vec = (patched_mg - corr_mg)  # (B,)
                    effect_sum[layer_idx, module_idx] += eff_vec.sum().item()

                elif direction == "block":
                    # run on clean input (embedding noise OFF), but patch in corrupt activations
                    # To do that, we temporarily disable embedding noise by setting mode clean,
                    # BUT keep patch hooks active via ctx.mode="patch" while embed_hook must behave clean.
                    # Easiest: reuse ctx.mode="patch" but make embed_hook add noise only when ctx.corrupt_mask is used.
                    # In this script embed_hook adds noise whenever ctx.mode != "clean". So for block we need a trick:
                    # We'll do a clean forward first (no noise) by temporarily setting ctx.mode="clean" and no patch,
                    # then a second forward with ctx.mode="patch_block" is complicated.
                    # Instead: for block, we approximate by using corrupt embeddings OFF by zeroing corrupt_mask temporarily.
                    # This keeps embed_hook from adding noise.
                    saved_corrupt_mask = ctx.corrupt_mask_bt
                    saved_emb_noise = ctx.emb_noise

                    ctx.corrupt_mask_bt = torch.zeros_like(saved_corrupt_mask)
                    ctx.emb_noise = None   # 关键：强制 embed_hook 重新生成噪声（此时mask为0 => 噪声全0）

                    ctx.patch_cfg = PatchCfg(layer_idx, module_name, patch_scope, source="corr")
                    ctx.tokens_bt = tokens_kg
                    blocked_logits = forward_logits(model, tokens_kg)
                    blocked_mg = margin_logit(blocked_logits, labels_kg)
                    eff_vec = (clean_mg - blocked_mg)
                    effect_sum[layer_idx, module_idx] += eff_vec.sum().item()

                    ctx.corrupt_mask_bt = saved_corrupt_mask
                    ctx.emb_noise = saved_emb_noise

                else:
                    raise ValueError("direction must be restore|block")

    for h in handles:
        h.remove()

    if n_sum == 0:
        raise RuntimeError("No valid samples processed (maybe corruption too weak/too strong or keep too low).")

    effect_mean = effect_sum / n_sum
    avg_drop = drop_sum / n_sum
    recovery = effect_sum / max(drop_sum, eps)
    keep_rate = keep_rate_sum / max(keep_batches, 1)
    if corrupt_cfg.mode in ("sentiment", "sentiment_or_span") and sent_hit_n > 0:
        print("=== SENTIMENT STATS (on tokens_k) ===")
        print("sent_hit_rate (overall):", sent_hit_sum / sent_hit_n)
        if corrupt_cfg.mode == "sentiment_or_span":
            print("fallback_rate (overall):", sent_fallback_sum / sent_hit_n)


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
    # reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_path = "log_dir/nodp/checkpoint_best.pt"
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_dict = task.source_dictionary

    # --- choose corruption/patch style here
    corrupt_cfg = CorruptCfg(alpha=0.3, mode="sentiment_or_span", span_len=3, exclude_cls=True, exclude_special=True)

    # 你可以试：
    # patch_scope = "focus"      # 只 patch 被腐蚀的 token（推荐，更像 ROME）
    #patch_scope = "cls"        # 只 patch CLS
    # patch_scope = "focus+cls"  # 两者
    # patch_scope = "all"        # 退回你原来的 B-ALL
    patch_scope = "focus"

    # direction:
    # "restore": corrupt输入上 patch回clean（恢复）
    # "block":   clean输入上注入corr cache（阻断/注入）
    direction = "block"
    print("dict[0:20] =", [src_dict[i] for i in range(20)])
    print("index Ġgood =", src_dict.index("Ġgood"), "unk =", src_dict.unk())
    print("index good  =", src_dict.index("good"),  "unk =", src_dict.unk())

    itr = build_valid_iterator(task, seed=1, max_sentences=16)
    effect, recovery, avg_drop, keep_rate = run_experiment_B_focus(
        model=model,
        dataloader=itr,
        src_dict=src_dict,
        corrupt_cfg=corrupt_cfg,
        patch_scope=patch_scope,
        direction=direction,
        max_batches=50,
        device=device,
    )

    print("=== CONFIG ===")
    print("corrupt_cfg =", corrupt_cfg)
    print("patch_scope =", patch_scope)
    print("direction   =", direction)
    print("keep_rate(clean correct filter) =", keep_rate)
    print("avg_drop(clean-corr, per-sample) =", avg_drop)

    print("Effect mean (shape):", tuple(effect.shape), "  [cols: attn, ffn]")
    print(effect)

    print("Recovery (effect_sum/drop_sum) (shape):", tuple(recovery.shape), "  [cols: attn, ffn]")
    print(recovery)
