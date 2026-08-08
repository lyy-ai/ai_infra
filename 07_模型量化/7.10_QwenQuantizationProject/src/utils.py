import json
import os


def save_json(data, path):
    """保存数据为 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary_table(results_dict):
    """打印 benchmark 汇总表格"""
    print("\n" + "=" * 90)
    print(f"{'Precision':>12} {'Model Memory(GB)':>18} {'Total GPU(GB)':>16} {'Avg TPS':>12} {'Avg Time(s)':>14}")
    print("=" * 90)
    for precision, data in results_dict.items():
        memory = data.get("memory", {})
        generations = data.get("generations", [])
        avg_tps = sum(g["tokens_per_sec"] for g in generations) / len(generations) if generations else 0
        avg_time = sum(g["time_sec"] for g in generations) / len(generations) if generations else 0
        print(
            f"{precision:>12} "
            f"{memory.get('model_memory_gb', 0):>18.2f} "
            f"{memory.get('total_gb', 0):>16.2f} "
            f"{avg_tps:>12.2f} "
            f"{avg_time:>14.3f}"
        )
    print("=" * 90 + "\n")
