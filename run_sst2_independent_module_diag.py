import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path


TRAIN_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|")
VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")


def build_sess(method, lora_mode, sharing, lora_modules, rank, seed, sess_suffix=""):
    modules_tag = str(lora_modules).replace(",", "")
    base = f"dp_sst2_{method}_lm_{lora_mode}_sh_{sharing}_mod_{modules_tag}_r{rank}_s{seed}"
    if sess_suffix:
        return f"{base}_{sess_suffix}"
    return base


def parse_final_epoch(log_file):
    final_epoch = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VALID_PATTERN.search(line)
            if m:
                final_epoch = int(m.group(1))
    return final_epoch


def parse_epoch_metrics(log_file):
    train_by_epoch = {}
    valid_by_epoch = {}
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mt = TRAIN_PATTERN.search(line)
            if mt:
                train_by_epoch[int(mt.group(1))] = float(mt.group(2))
            mv = VALID_PATTERN.search(line)
            if mv:
                valid_by_epoch[int(mv.group(1))] = float(mv.group(2))
    epochs = sorted(set(train_by_epoch.keys()).union(valid_by_epoch.keys()))
    rows = []
    for ep in epochs:
        rows.append(
            {
                "epoch": ep,
                "train_loss": train_by_epoch.get(ep, ""),
                "valid_accuracy": valid_by_epoch.get(ep, ""),
            }
        )
    return rows


def build_cmd(args, method, lora_modules, rank, seed):
    lora_mode = "standard"
    sharing = "none"
    shared_modules = "none"
    sess = build_sess(method, lora_mode, sharing, lora_modules, rank, seed, args.sess_suffix)
    cmd = [
        "python", "run_exp.py",
        "--gpu_id", str(args.gpu_id),
        "--task", "SST-2",
        "--arch", "roberta.base",
        "--eps", str(args.eps),
        "--delta", str(args.delta),
        "--clip", str(args.clip),
        "--accountant", "prv",
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--epoch", str(args.epoch),
        "--batch_size", str(args.batch_size),
        "--max_sentences", str(args.max_sentences),
        "--max_tokens", str(args.max_tokens),
        "--seed", str(seed),
        "--k", str(rank),
        "--sess", sess,
        "--save_root", args.save_root,
        "--route_method", method,
        "--lora_mode", lora_mode,
        "--shared_modules", shared_modules,
        "--lora_modules", lora_modules,
    ]
    if args.fp32:
        cmd.append("--fp32")
    return cmd, sess, lora_mode, shared_modules, sharing


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(plan_rows, out_dir):
    detail_rows = []
    run_rows = []
    for row in plan_rows:
        if row["status"] not in {"done", "skip_finished"}:
            continue
        log_file = Path(row["log_file"])
        if not log_file.exists():
            continue
        metrics = parse_epoch_metrics(log_file)
        first_train = next((m["train_loss"] for m in metrics if m["train_loss"] != ""), None)
        last_train = next((m["train_loss"] for m in reversed(metrics) if m["train_loss"] != ""), None)
        first_valid = next((m["valid_accuracy"] for m in metrics if m["valid_accuracy"] != ""), None)
        last_valid = next((m["valid_accuracy"] for m in reversed(metrics) if m["valid_accuracy"] != ""), None)
        delta_train = "" if first_train is None or last_train is None else (last_train - first_train)
        delta_valid = "" if first_valid is None or last_valid is None else (last_valid - first_valid)
        learning_started = "unknown"
        if first_train is not None and last_train is not None and first_valid is not None and last_valid is not None:
            learning_started = "yes" if (last_train < first_train - 1e-3 or last_valid > first_valid + 5e-3) else "no"

        run_rows.append(
            {
                "method": row["method"],
                "lora_mode": row["lora_mode"],
                "sharing": row["sharing"],
                "lora_modules": row["lora_modules"],
                "rank": row["rank"],
                "seed": row["seed"],
                "status": row["status"],
                "first_train_loss": "" if first_train is None else f"{first_train:.6f}",
                "last_train_loss": "" if last_train is None else f"{last_train:.6f}",
                "delta_train_loss": "" if delta_train == "" else f"{delta_train:.6f}",
                "first_valid_accuracy": "" if first_valid is None else f"{first_valid:.6f}",
                "last_valid_accuracy": "" if last_valid is None else f"{last_valid:.6f}",
                "delta_valid_accuracy": "" if delta_valid == "" else f"{delta_valid:.6f}",
                "learning_started": learning_started,
                "log_file": str(log_file),
            }
        )
        for m in metrics:
            detail_rows.append(
                {
                    "method": row["method"],
                    "lora_modules": row["lora_modules"],
                    "rank": row["rank"],
                    "seed": row["seed"],
                    "epoch": m["epoch"],
                    "train_loss": "" if m["train_loss"] == "" else f"{m['train_loss']:.6f}",
                    "valid_accuracy": "" if m["valid_accuracy"] == "" else f"{m['valid_accuracy']:.6f}",
                    "log_file": str(log_file),
                }
            )

    agg_rows = []
    methods = sorted(set(r["method"] for r in run_rows))
    for method in methods:
        sub = [r for r in run_rows if r["method"] == method]
        deltas = [float(r["delta_valid_accuracy"]) for r in sub if r["delta_valid_accuracy"] != ""]
        finals = [float(r["last_valid_accuracy"]) for r in sub if r["last_valid_accuracy"] != ""]
        learn_yes = sum(1 for r in sub if r["learning_started"] == "yes")
        agg_rows.append(
            {
                "method": method,
                "num_runs": len(sub),
                "num_learning_yes": learn_yes,
                "mean_last_valid_accuracy": "" if not finals else f"{statistics.mean(finals):.6f}",
                "mean_delta_valid_accuracy": "" if not deltas else f"{statistics.mean(deltas):.6f}",
                "judgement": "learning" if learn_yes >= max(1, len(sub) // 2) else "not_learning",
            }
        )

    detail_csv = out_dir / "diag_independent_module_epoch_detail.csv"
    run_csv = out_dir / "diag_independent_module_run_summary.csv"
    agg_csv = out_dir / "diag_independent_module_method_summary.csv"
    write_csv(
        detail_csv,
        ["method", "lora_modules", "rank", "seed", "epoch", "train_loss", "valid_accuracy", "log_file"],
        detail_rows,
    )
    write_csv(
        run_csv,
        [
            "method",
            "lora_mode",
            "sharing",
            "lora_modules",
            "rank",
            "seed",
            "status",
            "first_train_loss",
            "last_train_loss",
            "delta_train_loss",
            "first_valid_accuracy",
            "last_valid_accuracy",
            "delta_valid_accuracy",
            "learning_started",
            "log_file",
        ],
        run_rows,
    )
    write_csv(
        agg_csv,
        ["method", "num_runs", "num_learning_yes", "mean_last_valid_accuracy", "mean_delta_valid_accuracy", "judgement"],
        agg_rows,
    )
    print(f"Saved epoch detail: {detail_csv}")
    print(f"Saved run summary: {run_csv}")
    print(f"Saved method summary: {agg_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--eps", type=float, default=7.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--max_sentences", type=int, default=80)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--save_root", type=str, default="log_dir")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--seed_list", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--sess_suffix", type=str, default="diagfixao")
    args = parser.parse_args()

    plans = [
        ("independent_attn_only", "attn"),
        ("independent_ffn_only", "ffn"),
        ("independent_attn_ffn", "attn,ffn"),
    ]

    base_dir = Path(__file__).resolve().parent
    root = Path(args.save_root)
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    task_dir = root / "SST-2"
    task_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for method, lora_modules in plans:
        for seed in args.seed_list:
            cmd, sess, lora_mode, shared_modules, sharing = build_cmd(args, method, lora_modules, args.rank, seed)
            jobs.append((method, lora_mode, shared_modules, sharing, lora_modules, args.rank, seed, sess, cmd))

    print(f"Total jobs: {len(jobs)}")
    rows = []
    for i, (method, lora_mode, shared_modules, sharing, lora_modules, rank, seed, sess, cmd) in enumerate(jobs, 1):
        log_file = task_dir / f"{sess}_train_log.txt"
        print(f"[{i:02d}/{len(jobs):02d}] method={method}, modules={lora_modules}, rank={rank}, seed={seed}, sess={sess}")
        print(" ".join(cmd))
        status = "planned"
        if log_file.exists():
            final_epoch = parse_final_epoch(log_file)
            if final_epoch is not None and final_epoch >= args.epoch:
                print(f"SKIP: already finished sess={sess}, final_epoch={final_epoch}")
                status = "skip_finished"
        if args.run and not args.summary_only and status == "planned":
            try:
                subprocess.run(cmd, check=True)
                status = "done"
            except subprocess.CalledProcessError as e:
                status = f"failed_{e.returncode}"
                print(f"WARNING: failed sess={sess}, returncode={e.returncode}")
        rows.append(
            {
                "method": method,
                "lora_mode": lora_mode,
                "sharing": sharing,
                "shared_modules": shared_modules,
                "lora_modules": lora_modules,
                "rank": rank,
                "seed": seed,
                "sess": sess,
                "status": status,
                "log_file": str(log_file),
            }
        )

    plan_csv = base_dir / "diag_independent_module_plan.csv"
    write_csv(
        plan_csv,
        ["method", "lora_mode", "sharing", "shared_modules", "lora_modules", "rank", "seed", "sess", "status", "log_file"],
        rows,
    )
    print(f"Saved plan/status: {plan_csv}")
    summarize(rows, base_dir)


if __name__ == "__main__":
    main()
