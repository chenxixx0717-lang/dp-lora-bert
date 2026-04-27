import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_final_epoch(log_file):
    pattern = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")
    final_epoch = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                final_epoch = int(m.group(1))
    return final_epoch


def layer_tag(active_layers):
    if active_layers == "all":
        return "lall"
    return "l" + "".join(str(x) for x in active_layers.split(","))


def method_tag(name):
    return {
        "First-4-r15": "first",
        "Last-4-r15": "last",
        "Causal-top4-r15": "causal",
        "Random-top4-r15": "rand",
        "Uniform-all12-r5": "uniform",
    }[name]


def build_sess(name, active_layers, k, seed):
    return f"dp_sst2_{method_tag(name)}_{layer_tag(active_layers)}_r{k}_s{seed}"


def build_cmd(args, method_name, active_layers, k, seed):
    sess = build_sess(method_name, active_layers, k, seed)
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
        "--k", str(k),
        "--sess", sess,
        "--save_root", args.save_root,
        "--active_layers", active_layers,
        "--route_method", method_name,
    ]
    if args.fp32:
        cmd.append("--fp32")
    if args.to_console:
        cmd.append("--to_console")
    return cmd, sess


def write_plan_csv(rows):
    out_csv = Path(__file__).resolve().parent / "layerroute_train_plan.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "active_layers", "rank", "seed", "sess", "status", "log_file"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


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
    parser.add_argument("--seed_list", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--include_uniform", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--to_console", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    plans = [
        ("First-4-r15", "0,1,2,3", 15),
        ("Last-4-r15", "8,9,10,11", 15),
        ("Causal-top4-r15", "0,1,2,5", 15),
        ("Random-top4-r15", "0,2,4,7", 15),
        ("Random-top4-r15", "1,3,5,8", 15),
        ("Random-top4-r15", "2,4,6,10", 15),
        ("Random-top4-r15", "0,5,7,11", 15),
        ("Random-top4-r15", "1,6,8,10", 15),
    ]
    if args.include_uniform:
        plans = [("Uniform-all12-r5", "all", 5)] + plans

    root = Path(args.save_root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
    task_dir = root / "SST-2"
    task_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for method_name, active_layers, k in plans:
        for seed in args.seed_list:
            cmd, sess = build_cmd(args, method_name, active_layers, k, seed)
            jobs.append((method_name, active_layers, k, seed, sess, cmd))

    print(f"Total jobs: {len(jobs)}")
    rows = []
    for i, (method_name, active_layers, k, seed, sess, cmd) in enumerate(jobs, 1):
        log_file = task_dir / f"{sess}_train_log.txt"
        print(f"[{i:02d}/{len(jobs):02d}] method={method_name}, layers={active_layers}, rank={k}, seed={seed}")
        print(" ".join(cmd))
        status = "planned"
        if log_file.exists():
            final_epoch = parse_final_epoch(log_file)
            if final_epoch is not None and final_epoch >= args.epoch:
                print(f"SKIP: already finished sess={sess}, final_epoch={final_epoch}")
                status = "skip_finished"
                rows.append({
                    "method": method_name,
                    "active_layers": active_layers,
                    "rank": k,
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
            "method": method_name,
            "active_layers": active_layers,
            "rank": k,
            "seed": seed,
            "sess": sess,
            "status": status,
            "log_file": str(log_file),
        })

    out_csv = write_plan_csv(rows)
    print(f"\nSaved train plan/status CSV: {out_csv}")
    print("NOTE: sess naming format is dp_sst2_<method>_<layers>_r<rank>_s<seed>.")


if __name__ == "__main__":
    main()
