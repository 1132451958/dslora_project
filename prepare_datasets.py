# prepare_datasets.py
"""
构建五个 FULL 数据集：
  - T1_general
  - T2_math
  - T3_code
  - T4_tool  (使用 ToolBench)
  - T5_safety

输出到 data/full/
"""

import os
import json
from datasets import load_dataset

os.makedirs("data/full", exist_ok=True)


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(rows)} rows → {path}")


# ======================================================
# T1
# ======================================================

def build_T1_general_full():
    rows = []

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in alpaca:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", "") or "",
            "output": ex["output"],
        })

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in dolly:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("context", "") or "",
            "output": ex["response"],
        })

    save_jsonl("data/full/T1_general_full.jsonl", rows)


# ======================================================
# T2
# ======================================================

def build_T2_math_full():
    gsm = load_dataset("openai/gsm8k", "main", split="train")

    rows = []
    for ex in gsm:
        rows.append({
            "instruction": "Solve the math problem step by step.",
            "input": ex["question"],
            "output": ex["answer"],
        })

    save_jsonl("data/full/T2_math_full.jsonl", rows)


# ======================================================
# T3
# ======================================================

def build_T3_code_full():
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    rows = []
    for ex in code:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"],
        })

    save_jsonl("data/full/T3_code_full.jsonl", rows)


# ======================================================
# ⭐ T4（使用 ToolBench）
# ======================================================

def build_T4_tool_full(max_samples: int | None = None):
    """
    使用 madroid/glaive-function-calling-openai (openai_function_calling)
    构建 T4_tool_full.jsonl

    - 解析 ex["json"] （JSON 字符串）
    - instruction = 最后一个 user 的 content
    - output = 第一次出现的 assistant.tool_calls
    """

    from datasets import load_dataset
    import json as pyjson

    print("[T4] Loading madroid/glaive-function-calling-openai ...")

    try:
        ds = load_dataset(
            "madroid/glaive-function-calling-openai",
            "openai_function_calling",
            split="train",
        )
    except Exception as e:
        raise RuntimeError(
            "[T4 ERROR] Cannot load madroid/glaive-function-calling-openai\n"
            f"Original error: {e}"
        )

    rows = []

    for ex in ds:

        # ex["json"] 是 JSON 字符串
        try:
            data = pyjson.loads(ex["json"])
        except Exception:
            continue

        messages = data.get("messages", [])
        if not messages:
            continue

        # -----------------------------
        # 1) 提取用户指令（最后一条 user 消息）
        # -----------------------------
        instruction = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                instruction = m.get("content", "").strip()
                break

        if not instruction:
            continue

        # -----------------------------
        # 2) 提取函数调用（assistant.tool_calls）
        # -----------------------------
        function_call = None

        for m in messages:
            if m.get("role") == "assistant" and "tool_calls" in m:
                # 直接取 tool_calls（可能有多个）
                function_call = m["tool_calls"]
                break

        if function_call is None:
            continue

        # -----------------------------
        # 构建训练样本
        # -----------------------------
        rows.append({
            "instruction": instruction,
            "input": "",
            "output": pyjson.dumps(function_call, ensure_ascii=False),
        })

        if max_samples is not None and len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError("[T4 ERROR] No valid samples parsed from Glaive function-calling dataset.")

    os.makedirs("data/full", exist_ok=True)
    out_path = "data/full/T4_tool_full.jsonl"
    save_jsonl(out_path, rows)
    print(f"[T4] Parsed {len(rows)} samples → {out_path}")

# ======================================================
# T5
# ======================================================

def build_T5_safety_full():
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    rows = []
    for ex in ds:
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")

        if isinstance(chosen, dict):
            chosen = chosen.get("text", "") or chosen.get("completion", "")

        rows.append({
            "instruction": prompt or "Please respond safely.",
            "input": "",
            "output": chosen or rejected,
        })

    save_jsonl("data/full/T5_safety_full.jsonl", rows)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    print("=== Building FULL datasets ===")
    build_T1_general_full()
    build_T2_math_full()
    build_T3_code_full()
    build_T4_tool_full()     
    build_T5_safety_full()
    print("DONE → data/full/")

