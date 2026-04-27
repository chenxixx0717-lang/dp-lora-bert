import subprocess
import csv
import re
from pathlib import Path


def build_cmd(method, active_layers, k):
    return [
        "python", "run_exp.py",
        "--gpu_id", "0",
        "--task", "SST-2",
        "--arch", "roberta.base",
        "--eps", "7",
        "--delta", "1e-5",
        "--clip", "10",
        "--accountant", "prv",
        "--lr", "0.002",
        "--weight_decay", "0.01",
        "--epoch", "20",
        "--batch_size", "2000",
        "--max_sentences", "80",
        "--max_tokens", "4096",
        "--seed", "1",
        "--k", str(k),
        "--sess", f"dryrun_{method}",
        "--save_root", "log_dir",
        "--active_layers", active_layers,
        "--route_method", method,
        "--fp32",
        "--to_console",
        "--dry_run",
    ]


def parse_dryrun_output(text):
    summary = {}
    layer_lines = []
    for line in text.splitlines():
        if line.startswith("[DRY-RUN-SUMMARY] "):
            body = line.replace("[DRY-RUN-SUMMARY] ", "", 1)
            if "=" in body:
                k, v = body.split("=", 1)
                summary[k.strip()] = v.strip()
        elif line.startswith("[DRY-RUN-LAYER] "):
            layer_lines.append(line.replace("[DRY-RUN-LAYER] ", "", 1))
    if layer_lines:
        summary["per_layer_active_inactive"] = " | ".join(layer_lines)
        active_layers = []
        for item in layer_lines:
            m = re.search(r"layer=(\d+)\s+active_params=(\d+)", item)
            if m and int(m.group(2)) > 0:
                active_layers.append(int(m.group(1)))
        summary["active_layer_indices_detected"] = ",".join(str(x) for x in active_layers)
    return summary


def write_summary_csv(rows):
    out_csv = Path(__file__).resolve().parent / "layerroute_dryrun_budget_summary.csv"
    fieldnames = [
        "method",
        "active_layers",
        "rank(k)",
        "lora_trainable_params",
        "cls_head_params",
        "total_trainable_params",
        "active_layer_indices_detected",
        "per_layer_active_inactive",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def main():
    plans = [
        ("Uniform-all12-r5", "all", 5),
        ("Causal-top4-r15", "0,1,2,5", 15),
        ("First-4-r15", "0,1,2,3", 15),
        ("Last-4-r15", "8,9,10,11", 15),
        ("Random-top4-r15", "0,2,4,7", 15),
    ]
    rows = []
    for i, (method, active_layers, k) in enumerate(plans, 1):
        cmd = build_cmd(method, active_layers, k)
        print(f"\n[{i:02d}/{len(plans):02d}] DRY-RUN {method}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        print(output)
        parsed = parse_dryrun_output(output)
        rows.append({
            "method": method,
            "active_layers": parsed.get("active_layers", active_layers),
            "rank(k)": parsed.get("rank(k)", str(k)),
            "lora_trainable_params": parsed.get("lora_trainable_params", ""),
            "cls_head_params": parsed.get("cls_head_params", ""),
            "total_trainable_params": parsed.get("total_trainable_params", ""),
            "active_layer_indices_detected": parsed.get("active_layer_indices_detected", ""),
            "per_layer_active_inactive": parsed.get("per_layer_active_inactive", ""),
        })
    out_csv = write_summary_csv(rows)
    print(f"\nSaved dry-run budget summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
