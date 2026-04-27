import argparse
import csv
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


def build_sess(method, lora_mode, shared_modules, rank, seed, sess_suffix=""):
    tag = "attnffn" if shared_modules == "attn,ffn" else "attn"
    base = f"dp_sst2_{method}_lm_{lora_mode}_sm_{tag}_r{rank}_s{seed}"
    if sess_suffix:
        return f"{base}_{sess_suffix}"
    return base


def build_cmd(args, rank, seed):
    method = "independent_attn_only"
    lora_mode = "standard"
    shared_modules = "attn"
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
    return cmd, method, lora_mode, shared_modules, sess


def write_plan(rows, out_csv):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "lora_mode", "shared_modules", "rank", "seed", "sess", "status", "log_file"],
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
    parser.add_argument("--seed_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--rank_list", nargs="+", type=int, default=[5])
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sess_suffix", type=str, default="", help="optional suffix for session name, e.g. fixao_v2")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    root = Path(args.save_root)
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    task_dir = root / "SST-2"
    task_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rank in args.rank_list:
        for seed in args.seed_list:
            cmd, method, lora_mode, shared_modules, sess = build_cmd(args, rank, seed)
            jobs.append((method, lora_mode, shared_modules, rank, seed, sess, cmd))

    print(f"Total jobs: {len(jobs)}")
    rows = []
    for i, (method, lora_mode, shared_modules, rank, seed, sess, cmd) in enumerate(jobs, 1):
        log_file = task_dir / f"{sess}_train_log.txt"
        print(f"[{i:02d}/{len(jobs):02d}] method={method}, lora_mode={lora_mode}, shared_modules={shared_modules}, rank={rank}, seed={seed}, sess={sess}")
        print(" ".join(cmd))
        status = "planned"
        if log_file.exists():
            final_epoch = parse_final_epoch(log_file)
            if final_epoch is not None and final_epoch >= args.epoch:
                print(f"SKIP: already finished sess={sess}, final_epoch={final_epoch}")
                status = "skip_finished"
                rows.append({
                    "method": method, "lora_mode": lora_mode, "shared_modules": shared_modules,
                    "rank": rank, "seed": seed, "sess": sess, "status": status, "log_file": str(log_file),
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
            "method": method, "lora_mode": lora_mode, "shared_modules": shared_modules,
            "rank": rank, "seed": seed, "sess": sess, "status": status, "log_file": str(log_file),
        })

    out_csv = base_dir / "independent_attn_only_train_plan.csv"
    write_plan(rows, out_csv)
    print(f"\nSaved plan/status: {out_csv}")


if __name__ == "__main__":
    main()
