"""JSON 保存与表格打印工具。"""
import json
import os
from datetime import datetime


def save_json(data: dict, path: str):
    """保存 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    """返回当前时间 ISO 字符串。"""
    return datetime.now().isoformat(timespec="seconds")


def print_table(headers, rows):
    """打印简单表格。"""
    line = "-" * (sum(len(str(h)) for h in headers) + 3 * len(headers) + 1)
    print(line)
    print(" | ".join(str(h) for h in headers))
    print(line)
    for row in rows:
        print(" | ".join(str(cell) for cell in row))
    print(line)
