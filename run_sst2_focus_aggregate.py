import argparse
import csv
import re
import statistics
from pathlib import Path


VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")
TRAIN_PARAMS_PATTERN = re.compile(r"\[TRAIN-PARAMS\]\s+total_trainable_params=(\d+)")
ROUTE_PATTERN = re.compile(r"route_method='([^']+)'")
RANK_PATTERN = re.compile(r"\bk=(\d+)\b")


def parse_log(log_file: Path):
    best = None
    final = None
    params = None
    method = None
    rank = None
    points = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if method is None:
                m = ROUTE_PATTERN.search(line)
                if m:
                    method = m.group(1)
            if rank is None:
                m = RANK_PATTERN.search(line)
                if m:
                    rank = int(m.group(1))
            if params is None:
                m = TRAIN_PARAMS_PATTERN.search(line)
                if m:
                    params = int(m.group(1))
            m = VALID_PATTERN.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
    if points:
        best = max(v for _, v in points)
        final = points[-1][1]
    return method, rank, params, best, final


def map_series(method: str):
    if method == "baseline_all12_r5":
        return "baseline"
    if method == "independent_attn_only":
        return "independent_attn_only"
    if method == "shared_attn_only":
        return "shared_attn_only"
    return None


def mean_std(values):
    if not values:
        return "", ""
    if len(values) == 1:
        return f"{values[0]:.6f}", "0.000000"
    return f"{statistics.mean(values):.6f}", f"{statistics.stdev(values):.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="log_dir/SST-2")
    parser.add_argument("--sess_suffix", type=str, default="fixao_v4")
    parser.add_argument(
        "--methods",
        nargs="+",
        type=str,
        default=["independent_attn_only", "shared_attn_only"],
        help="Only aggregate these route_method values.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[5],
        help="Only aggregate these LoRA ranks.",
    )
    parser.add_argument("--out_agg_csv", type=str, default="attn_only_focus_aggregate.csv")
    parser.add_argument("--out_plot1_csv", type=str, default="attn_only_focus_plot1_final_vs_params.csv")
    parser.add_argument("--out_plot2_csv", type=str, default="attn_only_focus_plot2_best_vs_rank.csv")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = (base_dir / log_dir).resolve()

    out_agg_csv = Path(args.out_agg_csv)
    if not out_agg_csv.is_absolute():
        out_agg_csv = (base_dir / out_agg_csv).resolve()
    out_agg_csv.parent.mkdir(parents=True, exist_ok=True)

    out_plot1_csv = Path(args.out_plot1_csv)
    if not out_plot1_csv.is_absolute():
        out_plot1_csv = (base_dir / out_plot1_csv).resolve()
    out_plot1_csv.parent.mkdir(parents=True, exist_ok=True)

    out_plot2_csv = Path(args.out_plot2_csv)
    if not out_plot2_csv.is_absolute():
        out_plot2_csv = (base_dir / out_plot2_csv).resolve()
    out_plot2_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    allowed_methods = set(args.methods)
    allowed_ranks = set(args.ranks)
    pattern = f"*_{args.sess_suffix}_train_log.txt"
    for log_file in sorted(log_dir.glob(pattern)):
        method, rank, params, best, final = parse_log(log_file)
        series = map_series(method or "")
        if series is None:
            continue
        if method not in allowed_methods:
            continue
        if rank not in allowed_ranks:
            continue
        if rank is None or best is None or final is None or params is None:
            continue
        rows.append(
            {
                "series": series,
                "method": method,
                "rank": int(rank),
                "total_trainable_params": int(params),
                "best_acc": float(best),
                "final_acc": float(final),
                "log_file": str(log_file),
            }
        )

    grouped = {}
    for r in rows:
        key = (r["method"], r["rank"])
        grouped.setdefault(key, []).append(r)

    agg_rows = []
    plot1_rows = []
    plot2_rows = []
    for (method, rank), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        params_set = sorted({g["total_trainable_params"] for g in group})
        params = params_set[0] if params_set else ""
        best_vals = [g["best_acc"] for g in group]
        final_vals = [g["final_acc"] for g in group]
        best_mean, best_std = mean_std(best_vals)
        final_mean, final_std = mean_std(final_vals)
        agg_rows.append(
            {
                "method": method,
                "rank": rank,
                "num_runs": len(group),
                "total_trainable_params": params,
                "best_acc_mean": best_mean,
                "best_acc_std": best_std,
                "final_acc_mean": final_mean,
                "final_acc_std": final_std,
            }
        )
        plot1_rows.append(
            {
                "series": map_series(method),
                "method": method,
                "rank": rank,
                "x_total_trainable_params": params,
                "y_final_acc_mean": final_mean,
                "y_final_acc_std": final_std,
            }
        )
        plot2_rows.append(
            {
                "series": map_series(method),
                "method": method,
                "rank": rank,
                "x_rank": rank,
                "y_best_acc_mean": best_mean,
                "y_best_acc_std": best_std,
            }
        )

    with out_agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "rank",
                "num_runs",
                "total_trainable_params",
                "best_acc_mean",
                "best_acc_std",
                "final_acc_mean",
                "final_acc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    with out_plot1_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "method",
                "rank",
                "x_total_trainable_params",
                "y_final_acc_mean",
                "y_final_acc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(plot1_rows)

    with out_plot2_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "method",
                "rank",
                "x_rank",
                "y_best_acc_mean",
                "y_best_acc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(plot2_rows)

    print(f"Saved aggregate table: {out_agg_csv}")
    print(f"Saved plot1 data (final_acc_mean vs total_trainable_params): {out_plot1_csv}")
    print(f"Saved plot2 data (best_acc_mean vs rank): {out_plot2_csv}")


if __name__ == "__main__":
    main()
