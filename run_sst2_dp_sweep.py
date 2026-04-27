import argparse
import csv
import re
import statistics
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def parse_valid_accuracy(log_file):
    pattern = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")
    valid_points = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                valid_points.append((int(m.group(1)), float(m.group(2))))
    if not valid_points:
        return None, None, None, None
    best_epoch, best_acc = max(valid_points, key=lambda x: x[1])
    final_epoch, final_acc = valid_points[-1]
    return best_acc, best_epoch, final_acc, final_epoch


def parse_param_info(log_file):
    total_pattern = re.compile(r"num\. model params:\s+(\d+)\s+\(num\. trained:\s+(\d+)\)")
    trainable_k_pattern = re.compile(r"number of trainable parameters:\s+([0-9.]+)\s+K")
    total_params = None
    trained_params = None
    trainable_k = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if total_params is None:
                mt = total_pattern.search(line)
                if mt:
                    total_params = int(mt.group(1))
                    trained_params = int(mt.group(2))
            if trainable_k is None:
                mk = trainable_k_pattern.search(line)
                if mk:
                    trainable_k = float(mk.group(1))
            if total_params is not None and trainable_k is not None:
                break
    if trainable_k is not None:
        trained_params = int(round(trainable_k * 1000.0))
    trainable_ratio = None
    if total_params and trained_params is not None:
        trainable_ratio = trained_params / total_params
    return total_params, trained_params, trainable_k, trainable_ratio


def eps_tag(eps):
    text = str(eps)
    return text.replace(".", "p")


def session_name(prefix, eps, k, seed):
    return f"{prefix}_eps{eps_tag(eps)}_k{k}_seed{seed}"


def resolve_log_file(args, sess):
    primary = Path(args.save_root) / args.task / f"{sess}_train_log.txt"
    if primary.exists():
        return primary
    candidates = list(BASE_DIR.rglob(f"{sess}_train_log.txt"))
    if candidates:
        return candidates[0]
    return primary


def write_csv_with_fallback(path, fieldnames, rows):
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        with fallback.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"WARNING: cannot write {path}, saved to {fallback} instead.")
        return fallback


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


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_experiment_runs(args):
    runs = []
    # Priority plan:
    # 1) eps in {3,7}, k in {3,7,9}, base seeds (default 3 seeds)
    # 2) eps in {3,7}, k in {2,5,8,16}, total seeds to 5 (base + extra seeds)
    full_seed_list = unique_keep_order(args.seed_list + args.extra_seed_list)
    for eps in args.eps_list:
        for k in args.new_k_list:
            for seed in args.seed_list:
                runs.append((eps, k, seed))
        for k in args.key_k_list:
            for seed in full_seed_list:
                runs.append((eps, k, seed))
    return runs


def build_mechanism_artifacts(task_dir, sess_prefix, summary_rows):
    valid_rows = [
        r for r in summary_rows
        if r["num_runs"] > 0 and r["final_acc_mean"] != "" and r["final_acc_std"] != "" and r["trainable_params_mean"] != ""
    ]
    if not valid_rows:
        print("WARNING: no valid rows for mechanism artifacts.")
        return
    mechanism_rows = []
    for r in valid_rows:
        trainable_params = int(r["trainable_params_mean"])
        mechanism_rows.append({
            "k": r["k"],
            "trainable_params": trainable_params,
            "trainable_ratio": r["trainable_ratio_mean"],
            "best_acc_mean": r["best_acc_mean"],
            "best_acc_std": r["best_acc_std"],
            "final_acc_mean": r["final_acc_mean"],
            "final_acc_std": r["final_acc_std"],
            "noise_proxy_dim": trainable_params,
            "noise_proxy_sqrt_dim": f"{trainable_params ** 0.5:.6f}",
        })
    mechanism_csv = task_dir / f"{sess_prefix}_mechanism_table.csv"
    mechanism_csv = write_csv_with_fallback(
        mechanism_csv,
        [
            "k", "trainable_params", "trainable_ratio", "best_acc_mean", "best_acc_std",
            "final_acc_mean", "final_acc_std", "noise_proxy_dim", "noise_proxy_sqrt_dim",
        ],
        mechanism_rows,
    )
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARNING: matplotlib unavailable, skip plotting ({e}).")
        print(f"Saved mechanism table: {mechanism_csv}")
        return
    ks = [f"e{r['eps']}_k{r['k']}" for r in mechanism_rows]
    best_means = [float(r["best_acc_mean"]) for r in mechanism_rows]
    final_means = [float(r["final_acc_mean"]) for r in mechanism_rows]
    final_stds = [float(r["final_acc_std"]) for r in mechanism_rows]
    trainable_params = [int(r["trainable_params"]) for r in mechanism_rows]
    trainable_ratios = [float(r["trainable_ratio"]) for r in mechanism_rows]
    noise_sqrt = [float(r["noise_proxy_sqrt_dim"]) for r in mechanism_rows]

    fig_a, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(ks, best_means, marker="o", linewidth=1.8, label="best_acc_mean")
    ax1.plot(ks, final_means, marker="s", linewidth=1.8, label="final_acc_mean")
    ax2.plot(ks, trainable_params, marker="^", linewidth=1.8, color="tab:red", label="trainable_params")
    ax1.set_xlabel("eps-k")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Trainable Params")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    fig_a.tight_layout()
    fig_a_path = task_dir / f"{sess_prefix}_figureA_capacity_accuracy.png"
    fig_a.savefig(fig_a_path, dpi=200)
    plt.close(fig_a)

    fig_b, axb = plt.subplots(figsize=(7, 5))
    axb.plot(noise_sqrt, final_means, marker="o", linewidth=1.8)
    axb.set_xlabel("Noise Proxy sqrt(D(k))")
    axb.set_ylabel("Final Accuracy Mean")
    axb.grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b_path = task_dir / f"{sess_prefix}_figureB_noise_proxy.png"
    fig_b.savefig(fig_b_path, dpi=200)
    plt.close(fig_b)

    fig_c, axc = plt.subplots(figsize=(7, 5))
    axc.plot(ks, final_stds, marker="o", linewidth=1.8, label="final_acc_std")
    axc.set_xlabel("eps-k")
    axc.set_ylabel("Final Accuracy Std")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=9)
    fig_c.tight_layout()
    fig_c_path = task_dir / f"{sess_prefix}_figureC_stability.png"
    fig_c.savefig(fig_c_path, dpi=200)
    plt.close(fig_c)

    print(f"Saved mechanism table: {mechanism_csv}")
    print(f"Saved Figure A: {fig_a_path}")
    print(f"Saved Figure B: {fig_b_path}")
    print(f"Saved Figure C: {fig_c_path}")
    if trainable_ratios:
        print(f"Trainable ratio range: {min(trainable_ratios):.6f} -> {max(trainable_ratios):.6f}")


def build_command(args, eps, k, seed):
    sess = session_name(args.sess_prefix, eps, k, seed)
    cmd = [
        "python", "run_exp.py",
        "--gpu_id", str(args.gpu_id),
        "--task", args.task,
        "--arch", "roberta.base",
        "--eps", str(eps),
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
    ]
    if args.no_dp:
        cmd.append("--no_dp")
    if args.fp32:
        cmd.append("--fp32")
    if args.to_console:
        cmd.append("--to_console")
    return cmd, sess


def export_trainable_params(args):
    task_dir = Path(args.save_root) / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    output_csv = task_dir / f"{args.sess_prefix}_k_trainable_params.csv"
    base_total_params = 124698203
    lora_per_k = 147456
    cls_head_params = 1538
    rows = []
    for k in args.k_list:
        trainable_params = lora_per_k * k + cls_head_params
        total_params = base_total_params + lora_per_k * k
        trainable_ratio = trainable_params / total_params
        rows.append({
            "k": k,
            "trainable_params": trainable_params,
            "trainable_k": f"{trainable_params / 1000.0:.3f}",
            "total_params": total_params,
            "trainable_ratio": f"{trainable_ratio:.6f}",
        })
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "trainable_params", "trainable_k", "total_params", "trainable_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("\nPer-k trainable params:")
    print("k\ttrainable_params\ttrainable_k\ttotal_params\ttrainable_ratio")
    for r in rows:
        print(f"{r['k']}\t{r['trainable_params']}\t{r['trainable_k']}\t{r['total_params']}\t{r['trainable_ratio']}")
    print(f"\nSaved params: {output_csv}")


def summarize(args):
    task_dir = Path(args.save_root) / args.task
    rows = []
    runs = build_experiment_runs(args)
    missing_count = 0
    for eps, k, seed in runs:
        sess = session_name(args.sess_prefix, eps, k, seed)
        log_file = resolve_log_file(args, sess)
        if not log_file.exists():
            missing_count += 1
            rows.append({
                "eps": eps,
                "k": k,
                "seed": seed,
                "sess": sess,
                "best_acc": "",
                "best_epoch": "",
                "final_acc": "",
                "final_epoch": "",
                "total_params": "",
                "trained_params": "",
                "trainable_k": "",
                "trainable_ratio": "",
                "log_file": str(log_file),
                "status": "missing_log",
            })
            continue
        best_acc, best_epoch, final_acc, final_epoch = parse_valid_accuracy(log_file)
        total_params, trained_params, trainable_k, trainable_ratio = parse_param_info(log_file)
        rows.append({
            "eps": eps,
            "k": k,
            "seed": seed,
            "sess": sess,
            "best_acc": "" if best_acc is None else f"{best_acc:.6f}",
            "best_epoch": "" if best_epoch is None else best_epoch,
            "final_acc": "" if final_acc is None else f"{final_acc:.6f}",
            "final_epoch": "" if final_epoch is None else final_epoch,
            "total_params": "" if total_params is None else total_params,
            "trained_params": "" if trained_params is None else trained_params,
            "trainable_k": "" if trainable_k is None else f"{trainable_k:.3f}",
            "trainable_ratio": "" if trainable_ratio is None else f"{trainable_ratio:.6f}",
            "log_file": str(log_file),
            "status": "ok" if best_acc is not None else "no_valid_metric",
        })

    detail_csv = task_dir / f"{args.sess_prefix}_k_study_detail.csv"
    summary_csv = task_dir / f"{args.sess_prefix}_k_study_summary.csv"
    task_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = write_csv_with_fallback(
        detail_csv,
        [
            "eps", "k", "seed", "sess", "best_acc", "best_epoch", "final_acc", "final_epoch",
            "total_params", "trained_params", "trainable_k", "trainable_ratio", "log_file", "status",
        ],
        rows,
    )

    summary_rows = []
    pair_list = unique_keep_order([(eps, k) for eps, k, _ in runs])
    for eps, k in pair_list:
            sub = [r for r in rows if r["eps"] == eps and r["k"] == k and r["status"] == "ok"]
            best_vals = [float(r["best_acc"]) for r in sub if r["best_acc"] != ""]
            final_vals = [float(r["final_acc"]) for r in sub if r["final_acc"] != ""]
            ratio_vals = [float(r["trainable_ratio"]) for r in sub if r["trainable_ratio"] != ""]
            trainable_params_vals = [int(r["trained_params"]) for r in sub if r["trained_params"] != ""]
            summary_rows.append({
                "eps": eps,
                "k": k,
                "num_runs": len(sub),
                "best_acc_mean": "" if not best_vals else f"{statistics.mean(best_vals):.6f}",
                "best_acc_std": "" if len(best_vals) < 2 else f"{statistics.stdev(best_vals):.6f}",
                "final_acc_mean": "" if not final_vals else f"{statistics.mean(final_vals):.6f}",
                "final_acc_std": "" if len(final_vals) < 2 else f"{statistics.stdev(final_vals):.6f}",
                "trainable_params_mean": "" if not trainable_params_vals else int(round(statistics.mean(trainable_params_vals))),
                "trainable_ratio_mean": "" if not ratio_vals else f"{statistics.mean(ratio_vals):.6f}",
                "trainable_ratio_std": "" if len(ratio_vals) < 2 else f"{statistics.stdev(ratio_vals):.6f}",
            })
    summary_csv = write_csv_with_fallback(
        summary_csv,
        [
            "eps", "k", "num_runs", "best_acc_mean", "best_acc_std", "final_acc_mean", "final_acc_std",
            "trainable_params_mean", "trainable_ratio_mean", "trainable_ratio_std",
        ],
        summary_rows,
    )

    print("\nPer-k summary:")
    print("eps\tk\tnum_runs\tbest_mean\tbest_std\tfinal_mean\tfinal_std\ttrainable_ratio_mean")
    for r in summary_rows:
        print(
            f"{r['eps']}\t{r['k']}\t{r['num_runs']}\t{r['best_acc_mean']}\t{r['best_acc_std']}\t"
            f"{r['final_acc_mean']}\t{r['final_acc_std']}\t{r['trainable_ratio_mean']}"
        )
    print(f"\nSaved detail: {detail_csv}")
    print(f"Saved summary: {summary_csv}")
    if len(args.eps_list) == 1:
        build_mechanism_artifacts(task_dir, args.sess_prefix, summary_rows)
    if missing_count == len(runs):
        print("WARNING: no matching logs found. Please check --sess_prefix and --save_root.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--task", type=str, default="SST-2", choices=["MNLI", "QNLI", "QQP", "SST-2", "SNLI"])
    parser.add_argument("--eps_list", nargs="+", type=float, default=[3, 7])
    parser.add_argument("--new_k_list", nargs="+", type=int, default=[3, 7, 9])
    parser.add_argument("--key_k_list", nargs="+", type=int, default=[2, 5, 8, 16])
    parser.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 5, 7, 8, 9, 16])
    parser.add_argument("--seed_list", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--extra_seed_list", nargs="+", type=int, default=[45, 46])
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--eps", type=float, default=6.7)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--max_sentences", type=int, default=80)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sess_prefix", type=str, default="dp_lora_sst2_epsk")
    parser.add_argument("--save_root", type=str, default="log_dir")
    parser.add_argument("--no_dp", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--to_console", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--params_only", action="store_true")
    args = parser.parse_args()
    if not Path(args.save_root).is_absolute():
        args.save_root = str((BASE_DIR / args.save_root).resolve())
    if args.no_dp and args.sess_prefix == "dp_lora_sst2_epsk":
        args.sess_prefix = "nodp_lora_sst2_epsk"
    adjusted_max_sentences = adjust_max_sentences(args.batch_size, args.max_sentences)
    if adjusted_max_sentences != args.max_sentences:
        print(
            f"WARNING: batch_size {args.batch_size} is not divisible by max_sentences {args.max_sentences}, "
            f"use max_sentences={adjusted_max_sentences} instead."
        )
        args.max_sentences = adjusted_max_sentences
    print(f"Using update_freq={args.batch_size // args.max_sentences} (batch_size={args.batch_size}, max_sentences={args.max_sentences})")
    if args.params_only:
        export_trainable_params(args)
        return

    jobs = []
    for eps, k, seed in build_experiment_runs(args):
        cmd, sess = build_command(args, eps, k, seed)
        jobs.append((eps, k, seed, sess, cmd))

    print(f"Total jobs: {len(jobs)}")
    for i, (eps, k, seed, sess, cmd) in enumerate(jobs, 1):
        cmd_text = " ".join(cmd)
        print(f"[{i:02d}/{len(jobs):02d}] eps={eps}, k={k}, seed={seed}, sess={sess}")
        print(cmd_text)
        if args.run and not args.summary_only:
            log_file = Path(args.save_root) / args.task / f"{sess}_train_log.txt"
            if log_file.exists():
                _, _, _, final_epoch = parse_valid_accuracy(log_file)
                if final_epoch is not None and final_epoch >= args.epoch:
                    print(f"SKIP: already finished eps={eps}, k={k}, seed={seed}, final_epoch={final_epoch}")
                    continue
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"WARNING: job failed for eps={eps}, k={k}, seed={seed}, sess={sess}, returncode={e.returncode}")

    summarize(args)


if __name__ == "__main__":
    main()
