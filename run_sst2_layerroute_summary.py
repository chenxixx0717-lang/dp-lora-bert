import argparse
import csv
import statistics
import re
from pathlib import Path


VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")


def parse_valid_accuracy(log_file):
    points = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VALID_PATTERN.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
    if not points:
        return None, None, None, None
    best_epoch, best_acc = max(points, key=lambda x: x[1])
    final_epoch, final_acc = points[-1]
    return best_acc, best_epoch, final_acc, final_epoch


def fmt_mean_std(vals):
    if not vals:
        return "", ""
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


def load_plan_rows(plan_csv):
    rows = []
    with plan_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize(args):
    base_dir = Path(__file__).resolve().parent
    plan_csv = Path(args.plan_csv)
    if not plan_csv.is_absolute():
        plan_csv = (base_dir / plan_csv).resolve()
    plan_rows = load_plan_rows(plan_csv)

    run_rows = []
    for row in plan_rows:
        status = row.get("status", "")
        if status not in {"done", "skip_finished"}:
            continue
        log_file = Path(row["log_file"])
        if not log_file.is_absolute():
            log_file = (base_dir / log_file).resolve()
        if not log_file.exists():
            run_rows.append({
                **row,
                "best_acc": "",
                "best_epoch": "",
                "final_acc": "",
                "final_epoch": "",
                "parse_status": "missing_log",
            })
            continue
        best_acc, best_epoch, final_acc, final_epoch = parse_valid_accuracy(log_file)
        run_rows.append({
            **row,
            "best_acc": "" if best_acc is None else f"{best_acc:.6f}",
            "best_epoch": "" if best_epoch is None else best_epoch,
            "final_acc": "" if final_acc is None else f"{final_acc:.6f}",
            "final_epoch": "" if final_epoch is None else final_epoch,
            "parse_status": "ok" if best_acc is not None else "no_valid_metric",
        })

    detail_csv = plan_csv.with_name("layerroute_result_detail.csv")
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "active_layers", "rank", "seed", "sess", "status", "log_file",
                "best_acc", "best_epoch", "final_acc", "final_epoch", "parse_status",
            ],
        )
        writer.writeheader()
        writer.writerows(run_rows)

    ok_rows = [r for r in run_rows if r["parse_status"] == "ok"]

    # Random-top4: aggregate across 5 random sets * 3 seeds = 15 runs
    random_rows = [r for r in ok_rows if r["method"] == "Random-top4-r15"]
    random_set_summary = []
    for layers in sorted({r["active_layers"] for r in random_rows}):
        sub = [r for r in random_rows if r["active_layers"] == layers]
        best_vals = [float(r["best_acc"]) for r in sub]
        final_vals = [float(r["final_acc"]) for r in sub]
        bm, bs = fmt_mean_std(best_vals)
        fm, fs = fmt_mean_std(final_vals)
        random_set_summary.append({
            "method": "Random-top4-r15",
            "active_layers": layers,
            "rank": 15,
            "runs": len(sub),
            "best_acc_mean": bm,
            "best_acc_std": bs,
            "final_acc_mean": fm,
            "final_acc_std": fs,
        })

    random_set_csv = plan_csv.with_name("layerroute_random_set_summary.csv")
    with random_set_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "active_layers", "rank", "runs",
                "best_acc_mean", "best_acc_std", "final_acc_mean", "final_acc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(random_set_summary)

    # Final paper table
    summary_rows = []
    method_order = [
        "Uniform-all12-r5",
        "First-4-r15",
        "Last-4-r15",
        "Random-top4-r15",
        "Causal-top4-r15",
    ]
    for method in method_order:
        if method == "Random-top4-r15":
            sub = random_rows
            layer_disp = "5 random sets avg"
            rank = "15"
        else:
            sub = [r for r in ok_rows if r["method"] == method]
            layer_disp = "" if not sub else sorted({r["active_layers"] for r in sub})[0]
            rank = "" if not sub else str(sub[0]["rank"])
        best_vals = [float(r["best_acc"]) for r in sub]
        final_vals = [float(r["final_acc"]) for r in sub]
        bm, bs = fmt_mean_std(best_vals)
        fm, fs = fmt_mean_std(final_vals)
        summary_rows.append({
            "method": method,
            "active_layers": layer_disp,
            "rank": rank,
            "runs": len(sub),
            "best_acc_mean": bm,
            "best_acc_std": bs,
            "final_acc_mean": fm,
            "final_acc_std": fs,
            "best_acc_mean_pm_std": "" if not bm else f"{bm} ± {bs}",
            "final_acc_mean_pm_std": "" if not fm else f"{fm} ± {fs}",
        })

    if args.add_uniform_baseline:
        for r in summary_rows:
            if r["method"] == "Uniform-all12-r5":
                r["active_layers"] = "all 12 layers"
                r["rank"] = "5"
                r["runs"] = args.uniform_runs
                r["best_acc_mean"] = f"{args.uniform_best_mean:.6f}"
                r["best_acc_std"] = f"{args.uniform_best_std:.6f}"
                r["final_acc_mean"] = f"{args.uniform_final_mean:.6f}"
                r["final_acc_std"] = f"{args.uniform_final_std:.6f}"
                r["best_acc_mean_pm_std"] = f"{r['best_acc_mean']} ± {r['best_acc_std']}"
                r["final_acc_mean_pm_std"] = f"{r['final_acc_mean']} ± {r['final_acc_std']}"

    summary_csv = plan_csv.with_name("layerroute_method_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "active_layers", "rank", "runs",
                "best_acc_mean", "best_acc_std", "best_acc_mean_pm_std",
                "final_acc_mean", "final_acc_std", "final_acc_mean_pm_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved detail: {detail_csv}")
    print(f"Saved random-set summary: {random_set_csv}")
    print(f"Saved method summary: {summary_csv}")
    print("\nFinal table preview:")
    print("method\tactive_layers\trank\tbest_acc_mean±std\tfinal_acc_mean±std")
    for r in summary_rows:
        print(
            f"{r['method']}\t{r['active_layers']}\t{r['rank']}\t"
            f"{r['best_acc_mean_pm_std']}\t{r['final_acc_mean_pm_std']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_csv", type=str, default="layerroute_train_plan.csv")
    parser.add_argument("--add_uniform_baseline", action="store_true")
    parser.add_argument("--uniform_runs", type=int, default=3)
    parser.add_argument("--uniform_best_mean", type=float, default=0.928481)
    parser.add_argument("--uniform_best_std", type=float, default=0.006051)
    parser.add_argument("--uniform_final_mean", type=float, default=0.927217)
    parser.add_argument("--uniform_final_std", type=float, default=0.006441)
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
