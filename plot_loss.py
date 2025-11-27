# plot_loss.py
import json
import sys
import os

import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_single_train_loss(task_name: str):
    log_path = os.path.join("logs", f"single_{task_name}_train_loss.jsonl")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    rows = load_jsonl(log_path)
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title(f"Single-task Train Loss ({task_name})")
    plt.grid(True)
    out_path = os.path.join("logs", f"single_{task_name}_train_loss.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


def plot_seq_eval_loss():
    log_path = os.path.join("logs", "seq_eval_loss.jsonl")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    rows = load_jsonl(log_path)

    # 支持老日志：如果没有 method 字段，默认为 ds_lora_slsd
    for r in rows:
        if "method" not in r:
            r["method"] = "ds_lora_slsd"

    stages = sorted(list(set(r["stage"] for r in rows)))
    stage_to_idx = {s: i for i, s in enumerate(stages)}

    # (method, eval_task) -> xs, ys
    curves = {}
    for r in rows:
        m = r["method"]
        t = r["eval_task"]
        key = (m, t)
        if key not in curves:
            curves[key] = {"xs": [], "ys": []}
        curves[key]["xs"].append(stage_to_idx[r["stage"]])
        curves[key]["ys"].append(r["loss"])

    plt.figure()
    for (m, t), data in curves.items():
        label = f"{m}-{t}"
        plt.plot(data["xs"], data["ys"], marker="o", label=label)

    plt.xlabel("Stage (index)")
    plt.ylabel("Eval Loss")
    plt.title("Seq Training: Eval Loss on Tasks (by method)")
    plt.xticks(list(range(len(stages))), stages, rotation=45)
    plt.legend()
    plt.grid(True)
    out_path = os.path.join("logs", "seq_eval_loss.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    # 简单命令行用法：
    #   python plot_loss.py single T1_general
    #   python plot_loss.py seq
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plot_loss.py single <TASK_NAME>")
        print("  python plot_loss.py seq")
        sys.exit(0)

    mode = sys.argv[1]
    if mode == "single":
        if len(sys.argv) < 3:
            print("Please provide task name, e.g. T1_general")
            sys.exit(0)
        task_name = sys.argv[2]
        plot_single_train_loss(task_name)
    elif mode == "seq":
        plot_seq_eval_loss()
    else:
        print("Unknown mode:", mode)
