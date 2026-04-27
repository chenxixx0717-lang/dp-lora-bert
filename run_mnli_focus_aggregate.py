import argparse
import csv
import re
import statistics
from pathlib import Path


VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on '(valid|valid1)' subset \|.*?\| accuracy ([0-9.]+)")
TRAIN_PARAMS_PATTERN = re.compile(r"\[TRAIN-PARAMS\]\s+total_trainable_params=(\d+)")
ROUTE_PATTERN = re.compile(r"route_method='([^']+)'")
RANK_PATTERN = re.compile(r"\bk=(\d+)\b")


def mean_or_empty(values):
    if not values:
        return ""
    return statistics.mean(values)


def parse_log(log_file: Path):
    route_method = None
    rank = None
    params = None
    by_subset = {"valid": [], "valid1": []}

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if route_method is None:
                m = ROUTE_PATTERN.search(line)
                if m:
                    route_method = m.group(1)
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
                subset = m.group(2)
                by_subset.setdefault(subset, []).append((int(m.group(1)), float(m.group(3))))

    metrics = {}
    for subset in ("valid", "valid1"):
        points = by_subset.get(subset, [])
        if points:
            metrics[subset] = {
                "best_acc": max(v for _, v in points),
                "final_acc": points[-1][1],
            }
        else:
            metrics[subset] = {
                "best_acc": None,
                "final_acc": None,
            }

    return route_method, rank, params, metrics


def method_from_route(route_method: str):
    if route_method == "baseline_all12_r5":
        return "Uniform-all12-r5"
    if route_method == "independent_attn_only":
        return "independent_attn_only"
    if route_method == "shared_attn_only":
        return "shared_attn_only"
    return None


def fmt6(value):
    if value == "":
        return ""
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="log_dir/MNLI")
    parser.add_argument("--sess_suffix", type=str, default="mnli_focus_r1")
    parser.add_argument("--subset_for_main", type=str, choices=["valid", "valid1", "mean"], default="valid")
    parser.add_argument("--out_agg_csv", type=str, default="mnli_focus_aggregate_r1.csv")
    parser.add_argument("--out_detail_csv", type=str, default="mnli_focus_detail_by_split_r1.csv")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = (base_dir / log_dir).resolve()

    out_agg_csv = Path(args.out_agg_csv)
    if not out_agg_csv.is_absolute():
        out_agg_csv = (base_dir / out_agg_csv).resolve()
    out_agg_csv.parent.mkdir(parents=True, exist_ok=True)

    out_detail_csv = Path(args.out_detail_csv)
    if not out_detail_csv.is_absolute():
        out_detail_csv = (base_dir / out_detail_csv).resolve()
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    detail_rows = []
    pattern = f"*_{args.sess_suffix}_train_log.txt"
    for log_file in sorted(log_dir.glob(pattern)):
        route_method, rank, params, metrics = parse_log(log_file)
        method = method_from_route(route_method or "")
        if method is None:
            continue
        if rank is None or params is None:
            continue
        valid_best = metrics["valid"]["best_acc"]
        valid_final = metrics["valid"]["final_acc"]
        valid1_best = metrics["valid1"]["best_acc"]
        valid1_final = metrics["valid1"]["final_acc"]
        if valid_best is None and valid1_best is None:
            continue

        rows.append(
            {
                "method": method,
                "rank": int(rank),
                "total_trainable_params": int(params),
                "best_acc_valid": valid_best,
                "final_acc_valid": valid_final,
                "best_acc_valid1": valid1_best,
                "final_acc_valid1": valid1_final,
                "log_file": str(log_file),
            }
        )
        detail_rows.append(
            {
                "method": method,
                "rank": int(rank),
                "total_trainable_params": int(params),
                "best_acc_valid": "" if valid_best is None else f"{valid_best:.6f}",
                "final_acc_valid": "" if valid_final is None else f"{valid_final:.6f}",
                "best_acc_valid1": "" if valid1_best is None else f"{valid1_best:.6f}",
                "final_acc_valid1": "" if valid1_final is None else f"{valid1_final:.6f}",
                "log_file": str(log_file),
            }
        )

    grouped = {}
    for r in rows:
        key = (r["method"], r["rank"])
        grouped.setdefault(key, []).append(r)

    agg_rows = []
    for (method, rank), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        params_set = sorted({g["total_trainable_params"] for g in group})
        params = params_set[0] if params_set else ""

        best_vals_valid = [g["best_acc_valid"] for g in group if g["best_acc_valid"] is not None]
        final_vals_valid = [g["final_acc_valid"] for g in group if g["final_acc_valid"] is not None]
        best_vals_valid1 = [g["best_acc_valid1"] for g in group if g["best_acc_valid1"] is not None]
        final_vals_valid1 = [g["final_acc_valid1"] for g in group if g["final_acc_valid1"] is not None]

        best_valid_mean = mean_or_empty(best_vals_valid)
        final_valid_mean = mean_or_empty(final_vals_valid)
        best_valid1_mean = mean_or_empty(best_vals_valid1)
        final_valid1_mean = mean_or_empty(final_vals_valid1)

        if args.subset_for_main == "valid":
            main_best = best_valid_mean
            main_final = final_valid_mean
        elif args.subset_for_main == "valid1":
            main_best = best_valid1_mean
            main_final = final_valid1_mean
        else:
            main_best = mean_or_empty([x for x in (best_valid_mean, best_valid1_mean) if x != ""])
            main_final = mean_or_empty([x for x in (final_valid_mean, final_valid1_mean) if x != ""])

        agg_rows.append(
            {
                "method": method,
                "rank": rank,
                "num_runs": len(group),
                "total_trainable_params": params,
                "best_acc": fmt6(main_best) if main_best != "" else "",
                "final_acc": fmt6(main_final) if main_final != "" else "",
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
                "best_acc",
                "final_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    with out_detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "rank",
                "total_trainable_params",
                "best_acc_valid",
                "final_acc_valid",
                "best_acc_valid1",
                "final_acc_valid1",
                "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Saved aggregate table: {out_agg_csv}")
    print(f"Saved split detail table: {out_detail_csv}")
    print(f"Main summary subset: {args.subset_for_main}")


if __name__ == "__main__":
    main()
