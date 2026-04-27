import torch
from types import SimpleNamespace

from fairseq.modules.transformer_sentence_encoder import TransformerSentenceEncoder
from fairseq.lora_utils import LoraLinear, clear_batch_grad


def build_args(lora_mode="standard", shared_modules="attn", k=5):
    return SimpleNamespace(
        k=k,
        fp16=False,
        noln=False,
        lora_mode=lora_mode,
        shared_modules=shared_modules,
    )


def build_encoder(lora_mode="standard", shared_modules="attn", k=5, num_layers=12):
    args = build_args(lora_mode=lora_mode, shared_modules=shared_modules, k=k)
    return TransformerSentenceEncoder(
        args=args,
        padding_idx=1,
        vocab_size=100,
        num_encoder_layers=num_layers,
        embedding_dim=768,
        ffn_embedding_dim=3072,
        num_attention_heads=12,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_seq_len=16,
        num_segments=2,
        use_position_embeddings=True,
        apply_bert_init=False,
        encoder_normalize_before=False,
        embedding_normalize=False,
        rel_pos=False,
    )


def collect_attn_right_refs(enc):
    refs = {}
    for i, layer in enumerate(enc.layers):
        p_in = layer.self_attn.in_proj_right.weight
        p_out = layer.self_attn.out_proj_right.weight
        refs.setdefault(id(p_in), {"param": p_in, "names": []})
        refs[id(p_in)]["names"].append(f"layers.{i}.self_attn.in_proj_right.weight")
        refs.setdefault(id(p_out), {"param": p_out, "names": []})
        refs[id(p_out)]["names"].append(f"layers.{i}.self_attn.out_proj_right.weight")
    return refs


def collect_right_refs(enc, include_ffn=False):
    refs = collect_attn_right_refs(enc)
    if include_ffn:
        for i, layer in enumerate(enc.layers):
            p_fc1 = layer.fc1_right.weight
            p_fc2 = layer.fc2_right.weight
            refs.setdefault(id(p_fc1), {"param": p_fc1, "names": []})
            refs[id(p_fc1)]["names"].append(f"layers.{i}.fc1_right.weight")
            refs.setdefault(id(p_fc2), {"param": p_fc2, "names": []})
            refs[id(p_fc2)]["names"].append(f"layers.{i}.fc2_right.weight")
    return refs


def budget_report(enc, lora_mode, shared_modules, rank, include_ffn=False):
    lora_named = [(n, p) for n, p in enc.named_parameters() if ("left" in n or "right" in n)]
    total_lora_params = sum(p.numel() for _, p in lora_named)
    cls_head_params = 0
    total_trainable_params = total_lora_params + cls_head_params
    refs = collect_right_refs(enc, include_ffn=include_ffn)
    shared_param_count = sum(1 for v in refs.values() if len(v["names"]) > 1)
    non_shared_param_count = len(refs) - shared_param_count
    shared_lora_params = sum(v["param"].numel() for v in refs.values() if len(v["names"]) > 1)
    non_shared_lora_params = sum(v["param"].numel() for v in refs.values() if len(v["names"]) <= 1)
    print("\n[BUDGET]")
    print(f"lora_mode={lora_mode}")
    print(f"shared_modules={shared_modules}")
    print(f"rank={rank}")
    print(f"total_trainable_params={total_trainable_params}")
    print(f"shared_lora_params={shared_lora_params}")
    print(f"non_shared_lora_params={non_shared_lora_params}")
    print(f"cls_head_params={cls_head_params}")
    print(f"shared_param_count={shared_param_count}")
    print(f"non_shared_param_count={non_shared_param_count}")
    return total_trainable_params, shared_param_count, non_shared_param_count, shared_lora_params, non_shared_lora_params, cls_head_params


def run_encoder_backward(enc):
    enc.train()
    tokens = torch.randint(low=4, high=50, size=(2, 6), dtype=torch.long)
    clear_batch_grad([p for _, p in enc.named_parameters() if ("left" in _ or "right" in _)])
    inner_states, sent = enc(tokens)
    loss = inner_states[-1].sum() + sent.sum()
    loss.backward()


def numeric_consistency_check():
    torch.manual_seed(0)
    x = torch.randn(5, 2, 4)

    # standard: two independent modules, sum their batch_grad
    m1 = LoraLinear(4, 3)
    m2 = LoraLinear(4, 3)
    with torch.no_grad():
        m2.weight.copy_(m1.weight)
    clear_batch_grad([m1.weight, m2.weight])
    y_std = m1(x) + m2(x)
    y_std.sum().backward()
    g_std = m1.weight.batch_grad + m2.weight.batch_grad

    # shared: one module used twice
    m_shared = LoraLinear(4, 3)
    with torch.no_grad():
        m_shared.weight.copy_(m1.weight)
    clear_batch_grad([m_shared.weight])
    y_sh = m_shared(x) + m_shared(x)
    y_sh.sum().backward()
    g_sh = m_shared.weight.batch_grad
    max_err = (g_std - g_sh).abs().max().item()
    return max_err, max_err < 1e-6


def print_ref_status(enc, tag, include_ffn=False):
    refs = collect_right_refs(enc, include_ffn=include_ffn)
    print(f"\n[{tag}] attn-right parameter references")
    for v in refs.values():
        is_shared = len(v["names"]) > 1
        print(f"param_id={id(v['param'])} shared={is_shared} refs={v['names']}")


def check_batch_grad_status(enc, include_ffn=False):
    refs = collect_right_refs(enc, include_ffn=include_ffn)
    accum_ok = True
    reset_ok = True
    for v in refs.values():
        p = v["param"]
        bg = getattr(p, "batch_grad", None)
        print(f"param_id={id(p)} backward_batch_grad_shape={None if bg is None else tuple(bg.shape)}")
        if len(v["names"]) > 1 and bg is None:
            accum_ok = False
    clear_batch_grad([v["param"] for v in refs.values()])
    for v in refs.values():
        if getattr(v["param"], "batch_grad", None) is not None:
            reset_ok = False
    return accum_ok, reset_ok


def main():
    torch.manual_seed(0)

    # 1) standard
    enc_std = build_encoder(lora_mode="standard", shared_modules="attn", k=5)
    budget_report(enc_std, "standard", "attn", 5, include_ffn=False)
    print_ref_status(enc_std, "STANDARD", include_ffn=False)
    run_encoder_backward(enc_std)
    _, _ = check_batch_grad_status(enc_std, include_ffn=False)

    # 2) shared_right + attn (Phase A only)
    enc_shared = build_encoder(lora_mode="shared_right", shared_modules="attn", k=6)
    _, shared_param_count, non_shared_param_count, *_ = budget_report(
        enc_shared, "shared_right", "attn", 6, include_ffn=False
    )
    print_ref_status(enc_shared, "SHARED_RIGHT_ATTN", include_ffn=False)
    run_encoder_backward(enc_shared)
    batch_grad_accum_ok, batch_grad_reset_ok = check_batch_grad_status(enc_shared, include_ffn=False)

    # shared object check
    refs = collect_right_refs(enc_shared, include_ffn=False)
    shared_param_ok = any(len(v["names"]) > 1 for v in refs.values())

    # numeric consistency
    max_err, numeric_consistency_ok = numeric_consistency_check()
    print(f"\n[NUMERIC] max_abs_err={max_err:.8f}")

    budget_print_ok = (shared_param_count >= 1 and non_shared_param_count >= 0)
    print("\n[PHASE-A-CHECKS]")
    print(f"shared_param_ok={shared_param_ok}")
    print(f"batch_grad_accum_ok={batch_grad_accum_ok}")
    print(f"batch_grad_reset_ok={batch_grad_reset_ok}")
    print(f"numeric_consistency_ok={numeric_consistency_ok}")
    print(f"budget_print_ok={budget_print_ok}")

    # 3) shared_right + attn,ffn (Phase B)
    enc_shared_b = build_encoder(lora_mode="shared_right", shared_modules="attn,ffn", k=8)
    _, shared_param_count_b, non_shared_param_count_b, *_ = budget_report(
        enc_shared_b, "shared_right", "attn,ffn", 8, include_ffn=True
    )
    print_ref_status(enc_shared_b, "SHARED_RIGHT_ATTN_FFN", include_ffn=True)
    run_encoder_backward(enc_shared_b)
    batch_grad_accum_ok_b, batch_grad_reset_ok_b = check_batch_grad_status(enc_shared_b, include_ffn=True)
    refs_b = collect_right_refs(enc_shared_b, include_ffn=True)
    shared_param_ok_b = any(len(v["names"]) > 1 for v in refs_b.values())
    budget_print_ok_b = (shared_param_count_b >= 1 and non_shared_param_count_b >= 0)
    print("\n[PHASE-B-CHECKS]")
    print(f"shared_param_ok={shared_param_ok_b}")
    print(f"batch_grad_accum_ok={batch_grad_accum_ok_b}")
    print(f"batch_grad_reset_ok={batch_grad_reset_ok_b}")
    print(f"numeric_consistency_ok={numeric_consistency_ok}")
    print(f"budget_print_ok={budget_print_ok_b}")


if __name__ == "__main__":
    main()
