# prepare_datasets.py
"""
自动下载并转换五个任务的数据集，生成：
  data/T1_general.jsonl
  data/T2_math.jsonl
  data/T3_code.jsonl
  data/T4_tool.jsonl
  data/T5_safety.jsonl

每行都是：
  {"instruction": "...", "input": "...", "output": "..."}
"""

import os
import json
from datasets import load_dataset

os.makedirs("data", exist_ok=True)


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(rows)} rows to {path}")


# ---------- T1: 一般指令 / QA（Alpaca + Dolly，可先只用 Alpaca） ----------

def build_T1_general(max_samples_alpaca=2000, max_samples_dolly=0):
    rows = []

    print("[T1] Loading tatsu-lab/alpaca ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    n = min(max_samples_alpaca, len(alpaca))
    for ex in alpaca.select(range(n)):
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", "") or "",
            "output": ex["output"],
        })

    if max_samples_dolly > 0:
        print("[T1] Loading databricks/databricks-dolly-15k ...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        m = min(max_samples_dolly, len(dolly))
        for ex in dolly.select(range(m)):
            rows.append({
                "instruction": ex["instruction"],
                "input": ex.get("context", "") or "",
                "output": ex["response"],
            })

    save_jsonl("data/T1_general.jsonl", rows)


# ---------- T2: 数学推理（GSM8K） ----------

def build_T2_math(max_samples=2000):
    print("[T2] Loading openai/gsm8k ...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    n = min(max_samples, len(gsm))
    for ex in gsm.select(range(n)):
        q = ex["question"]
        a = ex["answer"]  # 原始答案字符串，里面通常含有“#### 123”这样的最终答案
        rows.append({
            "instruction": "Solve the math problem step by step and give the final answer.",
            "input": q,
            "output": a,
        })
    save_jsonl("data/T2_math.jsonl", rows)


# ---------- T3: 代码生成（CodeAlpaca） ----------

def build_T3_code(max_samples=2000):
    print("[T3] Loading sahil2801/CodeAlpaca-20k ...")
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    rows = []
    n = min(max_samples, len(code))
    for ex in code.select(range(n)):
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", "") or "",
            "output": ex["output"],
        })
    save_jsonl("data/T3_code.jsonl", rows)


# ---------- T4: 函数调用 / 工具使用（这里先构造一个小型示例集） ----------

def build_T4_tool():
    """
    这里先用手工构造的 function-calling 风格数据做本地调试：
    后续你上服务器、网络更顺畅时，我们可以再换成 ToolBench/OpenFunctions 的真实子集。
    """
    print("[T4] Building small synthetic tool-calling dataset ...")
    rows = [
        {
            "instruction": "Call the weather API to get the weather in Beijing tomorrow.",
            "input": "",
            "output": '{"tool_name": "get_weather", "arguments": {"location": "Beijing", "date": "tomorrow"}}',
        },
        {
            "instruction": "Generate a function call to search flights from Shanghai to Tokyo on 2025-10-01.",
            "input": "",
            "output": '{"tool_name": "search_flights", "arguments": {"from": "Shanghai", "to": "Tokyo", "date": "2025-10-01"}}',
        },
        {
            "instruction": "Given a user asking for Italian restaurants in Nanjing tonight at 7pm, create a function call.",
            "input": "",
            "output": '{"tool_name": "search_restaurants", "arguments": {"city": "Nanjing", "cuisine": "Italian", "time": "19:00"}}',
        },
        {
            "instruction": "The user wants to convert 100 USD to CNY. Generate a function call to a currency converter API.",
            "input": "",
            "output": '{"tool_name": "convert_currency", "arguments": {"from": "USD", "to": "CNY", "amount": 100}}',
        },
        {
            "instruction": "Create a function call that adds an event 'Thesis defense' on 2025-06-30 at 15:00 in Beijing.",
            "input": "",
            "output": '{"tool_name": "create_calendar_event", "arguments": {"title": "Thesis defense", "date": "2025-06-30", "time": "15:00", "location": "Beijing"}}',
        },
    ]
    save_jsonl("data/T4_tool.jsonl", rows)


# ---------- T5: 安全对齐 / 拒绝样本（HH-RLHF 的 harmless 子集） ----------

def build_T5_safety(max_samples=1000):
    """
    使用 Anthropic/hh-rlhf 的默认配置。
    数据结构通常包含：
        - "prompt"
        - "chosen"   （偏安全的回答）
        - "rejected" （不安全或较差的回答）
    我们取 prompt + chosen 来构造安全对齐训练数据。
    """
    print("[T5] Loading Anthropic/hh-rlhf (default config) ...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")  # ✔ 使用 default，无需 config 名称

    rows = []
    n = min(max_samples, len(ds))

    # 打印一下字段，帮助我们了解结构
    first = ds[0]
    keys = list(first.keys())
    print("[T5] Example keys:", keys)
    # 通常是 ['prompt', 'chosen', 'rejected']

    for ex in ds.select(range(n)):
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")

        # 如果 chosen 是 dict（某些版本可能有不同结构），做点兼容处理
        if isinstance(chosen, dict):
            chosen = chosen.get("text", "") or chosen.get("completion", "") or ""

        if not prompt:
            prompt = "Please respond to the following request in a safe and helpful way."

        rows.append({
            "instruction": prompt,
            "input": "",
            "output": chosen or rejected or "",
        })

    save_jsonl("data/T5_safety.jsonl", rows)


if __name__ == "__main__":
    # 注意：本地显存/内存和网络有限，先用比较小的样本数测试
    build_T1_general(max_samples_alpaca=1000, max_samples_dolly=0)
    build_T2_math(max_samples=1000)
    build_T3_code(max_samples=1000)
    build_T4_tool()
    build_T5_safety(max_samples=1000)
    print("All done.")
