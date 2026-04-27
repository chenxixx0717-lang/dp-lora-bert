import argparse
import csv
import re
from pathlib import Path


VALID_PATTERN = re.compile(r"\| epoch\s+(\d+)\s+\| valid on 'valid' subset \|.*?\| accuracy ([0-9.]+)")
TRAIN_PARAMS_PATTERN = re.compile(r"\[TRAIN-PARAMS\]\s+total_trainable_params=(\d+)")


def parse_log(log_file):
    points = []
    total_trainable_params = None
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VALID_PATTERN.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
            p = TRAIN_PARAMS_PATTERN.search(line)
            if p and total_trainable_params is None:
                total_trainable_params = int(p.group(1))
    if not points:
        return None, None, total_trainable_params
    best_acc = max(v for _, v in points)
    final_acc = points[-1][1]
    return best_acc, final_acc, total_trainable_params


def collect_rows(plan_csv, base_dir):
    rows = []
    with plan_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") not in {"done", "skip_finished"}:
                continue
            log_file = Path(row["log_file"])
            if not log_file.is_absolute():
                log_file = (base_dir / log_file).resolve()
            best_acc, final_acc, total_params = (None, None, None)
            if log_file.exists():
                best_acc, final_acc, total_params = parse_log(log_file)
            rows.append({
                "method": row["method"],
                "lora_mode": row["lora_mode"],
                "shared_modules": row["shared_modules"],
                "lora_modules": "attn",
                "sharing": "shared_right" if row["lora_mode"] == "shared_right" else "none",
                "rank": row["rank"],
                "seed": row["seed"],
                "total_trainable_params": "" if total_params is None else total_params,
                "best_acc": "" if best_acc is None else f"{best_acc:.6f}",
                "final_acc": "" if final_acc is None else f"{final_acc:.6f}",
                "log_file": str(log_file),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared_plan_csv", type=str, default="shared_right_train_plan.csv")
    parser.add_argument("--independent_plan_csv", type=str, default="independent_attn_only_train_plan.csv")
    parser.add_argument("--out_csv", type=str, default="attn_only_shared_vs_independent_detail.csv")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    shared_plan = Path(args.shared_plan_csv)
    if not shared_plan.is_absolute():
        shared_plan = (base_dir / shared_plan).resolve()
    independent_plan = Path(args.independent_plan_csv)
    if not independent_plan.is_absolute():
        independent_plan = (base_dir / independent_plan).resolve()
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = (base_dir / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if shared_plan.exists():
        rows.extend(collect_rows(shared_plan, base_dir))
    if independent_plan.exists():
        rows.extend(collect_rows(independent_plan, base_dir))

    rows.sort(key=lambda x: (x["method"], int(x["rank"]), int(x["seed"])))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "lora_mode", "shared_modules", "lora_modules", "sharing", "rank", "seed",
                "total_trainable_params", "best_acc", "final_acc", "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved compare detail: {out_csv}")


if __name__ == "__main__":
    main()
