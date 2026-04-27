import argparse
import csv
import re
import subprocess
from pathlib import Path


TRAIN_PARAMS_PATTERNS = {
    "total_trainable_params": re.compile(r"\[TRAIN-PARAMS\]\s+total_trainable_params=(\d+)"),
    "shared_lora_params": re.compile(r"\[TRAIN-PARAMS\]\s+shared_lora_params=(\d+)"),
    "non_shared_lora_params": re.compile(r"\[TRAIN-PARAMS\]\s+non_shared_lora_params=(\d+)"),
    "cls_head_params": re.compile(r"\[TRAIN-PARAMS\]\s+cls_head_params=(\d+)"),
    "lora_trainable_params": re.compile(r"\[DRY-RUN-SUMMARY\]\s+lora_trainable_params=(\d+)"),
}
OPT_PARAM_PATTERN = re.compile(r"\[OPT-PARAM\]\s+(.+)")
GRAD_PATTERN = re.compile(r"\[DEBUG-GRAD\]\s+(\S+)\s+name=(.+?)\s+grad_norm=(.+)")
FWD_PATTERN = re.compile(r"\[DEBUG-LORA-FWD\]\s+module=(\w+)\s+(\S+)=([0-9.eE+\-]+)\s+expected_nonzero=(True|False)")
TRAIN_LOSS_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|")
VALID_ACC_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def parse_log(log_file):
    metrics = {k: "" for k in TRAIN_PARAMS_PATTERNS}
    opt_params = []
    grad_rows = []
    fwd_rows = []
    train_losses = []
    valid_accs = []

    if not log_file.exists():
        return metrics, opt_params, grad_rows, fwd_rows, train_losses, valid_accs

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pat in TRAIN_PARAMS_PATTERNS.items():
                if metrics[key] == "":
                    m = pat.search(line)
                    if m:
                        metrics[key] = m.group(1)
            m = OPT_PARAM_PATTERN.search(line)
            if m:
                opt_params.append(m.group(1))
            m = GRAD_PATTERN.search(line)
            if m:
                grad_rows.append({"tag": m.group(1), "name": m.group(2), "grad_norm": m.group(3)})
            m = FWD_PATTERN.search(line)
            if m:
                fwd_rows.append(
                    {
                        "module": m.group(1),
                        "metric": m.group(2),
                        "norm": m.group(3),
                        "expected_nonzero": m.group(4),
                    }
                )
            m = TRAIN_LOSS_PATTERN.search(line)
            if m:
                train_losses.append((int(m.group(1)), float(m.group(2))))
            m = VALID_ACC_PATTERN.search(line)
            if m:
                valid_accs.append((int(m.group(1)), float(m.group(2))))

    return metrics, opt_params, grad_rows, fwd_rows, train_losses, valid_accs


def build_common(args, sess, method, lora_modules):
    cmd = [
        "python",
        "run_exp.py",
        "--gpu_id",
        str(args.gpu_id),
        "--task",
        "SST-2",
        "--arch",
        "roberta.base",
        "--eps",
        str(args.eps),
        "--delta",
        str(args.delta),
        "--clip",
        str(args.clip),
        "--accountant",
        "prv",
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--epoch",
        str(args.epoch),
        "--batch_size",
        str(args.batch_size),
        "--max_sentences",
        str(args.max_sentences),
        "--max_tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--k",
        str(args.rank),
        "--sess",
        sess,
        "--save_root",
        args.save_root,
        "--route_method",
        method,
        "--lora_mode",
        "standard",
        "--shared_modules",
        "none",
        "--lora_modules",
        lora_modules,
        "--fp32",
    ]
    if args.no_dp:
        cmd.append("--no_dp")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="log_dir")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--eps", type=float, default=7.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--max_sentences", type=int, default=80)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tiny_max_update", type=int, default=20)
    parser.add_argument("--sess_suffix", type=str, default="diagtech")
    parser.add_argument("--no_dp", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    task_dir = (base_dir / args.save_root / "SST-2").resolve()
    task_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("independent_attn_only", "attn"),
        ("independent_ffn_only", "ffn"),
        ("independent_attn_ffn", "attn,ffn"),
    ]

    diag_rows = []
    grad_rows_all = []
    fwd_rows_all = []
    opt_rows = []

    for method, lora_modules in configs:
        sess = f"diag_{method}_r{args.rank}_s{args.seed}_{args.sess_suffix}"
        log_file = task_dir / f"{sess}_train_log.txt"
        if args.run:
            cmd = build_common(args, sess, method, lora_modules)
            cmd.extend(["--dry_run", "--debug_param_list", "--debug_grad_norm_once", "--debug_lora_forward_once"])
            run_cmd(cmd)
            cmd2 = build_common(args, sess, method, lora_modules)
            cmd2.extend(["--max_update", "1", "--debug_grad_norm_once", "--debug_lora_forward_once"])
            run_cmd(cmd2)

        metrics, opt_params, grads, fwds, _, _ = parse_log(log_file)
        diag_rows.append(
            {
                "phase": "diag",
                "method": method,
                "lora_modules": lora_modules,
                "seed": args.seed,
                "rank": args.rank,
                **metrics,
                "num_optimizer_params": len(opt_params),
                "log_file": str(log_file),
            }
        )
        for p in opt_params:
            opt_rows.append({"method": method, "lora_modules": lora_modules, "param_name": p, "log_file": str(log_file)})
        for g in grads:
            g.update({"method": method, "lora_modules": lora_modules, "log_file": str(log_file)})
            grad_rows_all.append(g)
        for f in fwds:
            f.update({"method": method, "lora_modules": lora_modules, "log_file": str(log_file)})
            fwd_rows_all.append(f)

    tiny_cfgs = [
        ("baseline_all12_r5", "attn,ffn"),
        ("independent_attn_only", "attn"),
        ("independent_ffn_only", "ffn"),
    ]
    tiny_rows = []
    for method, lora_modules in tiny_cfgs:
        sess = f"tiny_{method}_r{args.rank}_s{args.seed}_{args.sess_suffix}"
        log_file = task_dir / f"{sess}_train_log.txt"
        if args.run:
            cmd = build_common(args, sess, method, lora_modules)
            cmd.extend(["--max_update", str(args.tiny_max_update)])
            run_cmd(cmd)
        _, _, _, _, train_losses, valid_accs = parse_log(log_file)
        first_train = train_losses[0][1] if train_losses else ""
        last_train = train_losses[-1][1] if train_losses else ""
        first_valid = valid_accs[0][1] if valid_accs else ""
        last_valid = valid_accs[-1][1] if valid_accs else ""
        tiny_rows.append(
            {
                "phase": "tiny_overfit",
                "method": method,
                "lora_modules": lora_modules,
                "seed": args.seed,
                "rank": args.rank,
                "first_train_loss": first_train,
                "last_train_loss": last_train,
                "delta_train_loss": "" if first_train == "" or last_train == "" else (last_train - first_train),
                "first_valid_acc": first_valid,
                "last_valid_acc": last_valid,
                "delta_valid_acc": "" if first_valid == "" or last_valid == "" else (last_valid - first_valid),
                "log_file": str(log_file),
            }
        )

    out_main = base_dir / "module_diag_report.csv"
    out_opt = base_dir / "module_diag_optimizer_params.csv"
    out_grad = base_dir / "module_diag_grad_norms.csv"
    out_fwd = base_dir / "module_diag_forward_norms.csv"
    with out_main.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "phase",
            "method",
            "lora_modules",
            "seed",
            "rank",
            "total_trainable_params",
            "lora_trainable_params",
            "cls_head_params",
            "shared_lora_params",
            "non_shared_lora_params",
            "num_optimizer_params",
            "first_train_loss",
            "last_train_loss",
            "delta_train_loss",
            "first_valid_acc",
            "last_valid_acc",
            "delta_valid_acc",
            "log_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diag_rows + tiny_rows)
    with out_opt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "lora_modules", "param_name", "log_file"])
        writer.writeheader()
        writer.writerows(opt_rows)
    with out_grad.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "lora_modules", "tag", "name", "grad_norm", "log_file"])
        writer.writeheader()
        writer.writerows(grad_rows_all)
    with out_fwd.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "lora_modules", "module", "metric", "norm", "expected_nonzero", "log_file"]
        )
        writer.writeheader()
        writer.writerows(fwd_rows_all)

    print(f"Saved report: {out_main}")
    print(f"Saved optimizer params: {out_opt}")
    print(f"Saved grad norms: {out_grad}")
    print(f"Saved forward norms: {out_fwd}")


if __name__ == "__main__":
    main()
