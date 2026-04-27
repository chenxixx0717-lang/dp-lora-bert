import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")


def parse_final_epoch(log_file):
    final_epoch = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VALID_PATTERN.search(line)
            if m:
                final_epoch = int(m.group(1))
    return final_epoch


def adjust_max_sentences(batch_size, max_sentences):
    if max_sentences <= 0:
        return max_sentences
    if batch_size % max_sentences == 0:
        return max_sentences
    divisors = [d for d in range(1, batch_size + 1) if batch_size % d == 0]
    lower = [d for d in divisors if d <= max_sentences]
    if lower:
        return max(lower)
    return min(divisors)


def is_sess_running(sess):
    """
    Best-effort guard to avoid launching duplicate jobs for the same session.
    Works on Linux/WSL via pgrep; on Windows this returns False when pgrep is unavailable.
    """
    if os.name == "nt":
        return False
    patterns = ["python train.py", "python run_exp.py"]
    for pat in patterns:
        try:
            output = subprocess.check_output(
                ["pgrep", "-af", pat], text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            continue
        for line in output.splitlines():
            if f"--sess {sess}" in line:
                return True
    return False


def has_sess_lock(task_dir, sess):
    lock_file = task_dir / f"{sess}.lock"
    return lock_file.exists()


def shared_tag(shared_modules):
    return "attnffn" if shared_modules == "attn,ffn" else "attn"


def build_sess(method, lora_mode, shared_modules, rank, seed, sess_suffix=""):
    base = f"dp_sst2_{method}_lm_{lora_mode}_sm_{shared_tag(shared_modules)}_r{rank}_s{seed}"
    if sess_suffix:
        return f"{base}_{sess_suffix}"
    return base


def build_cmd(args, method, lora_mode, shared_modules, rank, seed):
    sess = build_sess(method, lora_mode, shared_modules, rank, seed, args.sess_suffix)
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
        "--num_workers", str(args.num_workers),
        "--validate_interval_updates", str(args.validate_interval_updates),
        "--seed", str(seed),
        "--k", str(rank),
        "--sess", sess,
        "--save_root", args.save_root,
        "--route_method", method,
        "--lora_mode", lora_mode,
        "--shared_modules", shared_modules,
        "--lora_modules", "attn",
    ]
    if args.fp32:
        cmd.append("--fp32")
    if args.to_console:
        cmd.append("--to_console")
    return cmd, sess


def write_plan(rows, out_csv):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "lora_mode", "shared_modules", "rank", "seed",
                "sess", "status", "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--max_sentences", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--validate_interval_updates", type=int, default=20)
    parser.add_argument("--save_root", type=str, default="log_dir")
    parser.add_argument("--seed_list", nargs="+", type=int, default=[1, 2, 3], help="fallback seeds when not specified by plan")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--to_console", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sess_suffix", type=str, default="", help="optional suffix for session name, e.g. fixao_v2")
    args = parser.parse_args()

    # Focused plan: only补关键组 shared_attn_only rank=5 to 5 seeds.
    plans = [
        ("shared_attn_only", "shared_right", "attn", 5, [1, 2, 3, 4, 5]),
    ]

    base_dir = Path(__file__).resolve().parent
    root = Path(args.save_root)
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    task_dir = root / "SST-2"
    task_dir.mkdir(parents=True, exist_ok=True)
    adjusted_max_sentences = adjust_max_sentences(args.batch_size, args.max_sentences)
    if adjusted_max_sentences != args.max_sentences:
        print(
            f"WARNING: batch_size {args.batch_size} is not divisible by max_sentences "
            f"{args.max_sentences}, use max_sentences={adjusted_max_sentences} instead."
        )
        args.max_sentences = adjusted_max_sentences
    print(
        f"Using update_freq={args.batch_size // args.max_sentences} "
        f"(batch_size={args.batch_size}, max_sentences={args.max_sentences})"
    )

    jobs = []
    for method, lora_mode, shared_modules, rank, plan_seeds in plans:
        seeds = plan_seeds if plan_seeds else args.seed_list
        for seed in seeds:
            cmd, sess = build_cmd(args, method, lora_mode, shared_modules, rank, seed)
            jobs.append((method, lora_mode, shared_modules, rank, seed, sess, cmd))

    print(f"Total jobs: {len(jobs)}")
    rows = []
    for i, (method, lora_mode, shared_modules, rank, seed, sess, cmd) in enumerate(jobs, 1):
        log_file = task_dir / f"{sess}_train_log.txt"
        print(
            f"[{i:02d}/{len(jobs):02d}] method={method}, lora_mode={lora_mode}, "
            f"shared_modules={shared_modules}, rank={rank}, seed={seed}, sess={sess}"
        )
        print(" ".join(cmd))
        status = "planned"
        if log_file.exists():
            final_epoch = parse_final_epoch(log_file)
            if final_epoch is not None and final_epoch >= args.epoch:
                print(f"SKIP: already finished sess={sess}, final_epoch={final_epoch}")
                status = "skip_finished"
                rows.append({
                    "method": method,
                    "lora_mode": lora_mode,
                    "shared_modules": shared_modules,
                    "rank": rank,
                    "seed": seed,
                    "sess": sess,
                    "status": status,
                    "log_file": str(log_file),
                })
                continue
        if args.run and is_sess_running(sess):
            print(f"SKIP: detected running process for sess={sess}")
            status = "skip_running"
            rows.append({
                "method": method,
                "lora_mode": lora_mode,
                "shared_modules": shared_modules,
                "rank": rank,
                "seed": seed,
                "sess": sess,
                "status": status,
                "log_file": str(log_file),
            })
            continue
        if args.run and has_sess_lock(task_dir, sess):
            print(f"SKIP: detected session lock for sess={sess}")
            status = "skip_locked"
            rows.append({
                "method": method,
                "lora_mode": lora_mode,
                "shared_modules": shared_modules,
                "rank": rank,
                "seed": seed,
                "sess": sess,
                "status": status,
                "log_file": str(log_file),
            })
            continue
        if args.run:
            try:
                subprocess.run(cmd, check=True)
                status = "done"
            except subprocess.CalledProcessError as e:
                status = f"failed_{e.returncode}"
                print(f"WARNING: failed sess={sess}, returncode={e.returncode}")
        rows.append({
            "method": method,
            "lora_mode": lora_mode,
            "shared_modules": shared_modules,
            "rank": rank,
            "seed": seed,
            "sess": sess,
            "status": status,
            "log_file": str(log_file),
        })

    out_csv = base_dir / "shared_right_train_plan.csv"
    write_plan(rows, out_csv)
    print(f"\nSaved plan/status: {out_csv}")
    print("sess format: dp_sst2_<method>_lm_<lora_mode>_sm_<shared_modules>_r<rank>_s<seed>[_<sess_suffix>]")


if __name__ == "__main__":
    main()
