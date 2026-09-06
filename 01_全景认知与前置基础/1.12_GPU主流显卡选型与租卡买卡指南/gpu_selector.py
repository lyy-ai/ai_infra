#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1.12 GPU 主流显卡选型工具

内置主流 NVIDIA 显卡规格数据库，支持：
  1. --list                        打印全量规格对比表
  2. --params P --task T --dtype D 估算显存需求并推荐候选卡
  3. --plot                        生成显存-带宽选型地图（需 matplotlib）

显存估算模型（与讲义 1.12 第 4 节一致）：
  推理   : P(B) × bytes(dtype) × 1.3   （权重 + KV cache + 激活）
  LoRA   : P(B) × 2.5
  全参微调: P(B) × 16                   （AdamW 混合精度，总显存池）

示例：
  python gpu_selector.py --list
  python gpu_selector.py --params 7  --task lora
  python gpu_selector.py --params 70 --task inference --dtype int4
  python gpu_selector.py --params 70 --task full_finetune
"""

import argparse

# ---------------------------------------------------------------------------
# 规格数据库
# 字段: 显存(GB), 带宽(TB/s), FP16 稠密算力(TFLOPS), NVLink(GB/s, 0=无),
#       TDP(W), 类别(datacenter/consumer), 参考时租(¥/h, 0=不适合租赁参考),
#       单卡参考价(¥, 0=不单卖; 2025 年国内行情量级，波动大仅供估算)
# ---------------------------------------------------------------------------
GPU_DB = {
    "RTX 3090":    dict(vram=24,  bw=0.94, fp16=142,  nvlink=112,  tdp=350,  kind="consumer",    rent=1.2,  price=7000),
    "RTX 4090":    dict(vram=24,  bw=1.01, fp16=165,  nvlink=0,    tdp=450,  kind="consumer",    rent=2.0,  price=18000),
    "RTX 5090":    dict(vram=32,  bw=1.79, fp16=419,  nvlink=0,    tdp=575,  kind="consumer",    rent=3.0,  price=20000),
    "L4":          dict(vram=24,  bw=0.30, fp16=121,  nvlink=0,    tdp=72,   kind="datacenter",  rent=1.0,  price=17000),
    "L20":         dict(vram=48,  bw=0.86, fp16=119,  nvlink=0,    tdp=275,  kind="datacenter",  rent=2.5,  price=30000),
    "L40S":        dict(vram=48,  bw=0.86, fp16=362,  nvlink=0,    tdp=350,  kind="datacenter",  rent=4.0,  price=65000),
    "A100 40G":    dict(vram=40,  bw=1.56, fp16=312,  nvlink=600,  tdp=400,  kind="datacenter",  rent=4.0,  price=35000),
    "A100 80G":    dict(vram=80,  bw=2.04, fp16=312,  nvlink=600,  tdp=400,  kind="datacenter",  rent=6.5,  price=85000),
    "H20":         dict(vram=96,  bw=4.00, fp16=148,  nvlink=900,  tdp=400,  kind="datacenter",  rent=8.0,  price=115000),
    "H100":        dict(vram=80,  bw=3.35, fp16=989,  nvlink=900,  tdp=700,  kind="datacenter",  rent=20.0, price=250000),
    "H200":        dict(vram=141, bw=4.80, fp16=989,  nvlink=900,  tdp=700,  kind="datacenter",  rent=28.0, price=320000),
    "B200":        dict(vram=192, bw=8.00, fp16=2250, nvlink=1800, tdp=1000, kind="datacenter",  rent=0,    price=0),  # 多以 GB200 NVL72 整机柜交付
}

DTYPE_BYTES = {"fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


def _fmt_price(price):
    return f"{price / 10000:.1f}万" if price >= 10000 else (f"{price}" if price > 0 else "不单卖")


def list_gpus():
    """打印全量规格对比表。"""
    header = f"{'型号':<10} {'显存(GB)':>8} {'带宽(TB/s)':>10} {'FP16(TF)':>9} {'NVLink(GB/s)':>13} {'TDP(W)':>7} {'时租(¥/h)':>9} {'单卡价(¥)':>9}"
    print(header)
    print("-" * len(header))
    for name, s in sorted(GPU_DB.items(), key=lambda kv: kv[1]["vram"]):
        rent = f"{s['rent']:.1f}" if s["rent"] > 0 else "-"
        nv = s["nvlink"] if s["nvlink"] > 0 else "无"
        print(f"{name:<10} {s['vram']:>8} {s['bw']:>10.2f} {s['fp16']:>9} {nv:>13} {s['tdp']:>7} {rent:>9} {_fmt_price(s['price']):>9}")


def estimate_vram_gb(params_b: float, task: str, dtype: str) -> float:
    """按讲义公式估算显存需求（GB）。"""
    if task == "inference":
        return params_b * DTYPE_BYTES[dtype] * 1.3
    if task == "lora":
        return params_b * 2.5
    if task == "full_finetune":
        return params_b * 16.0
    raise ValueError(f"unknown task: {task}")


def recommend(params_b: float, task: str, dtype: str):
    """估算显存并按"单卡装下 → 多卡拼"的顺序推荐候选方案。"""
    need = estimate_vram_gb(params_b, task, dtype)
    task_name = {"inference": f"推理({dtype.upper()})", "lora": "LoRA 微调", "full_finetune": "全参微调"}[task]
    print(f"\n场景: {params_b:g}B 模型 / {task_name}")
    print(f"估算显存需求: {need:.0f} GB"
          + ("（总显存池，可用 ZeRO/FSDP 分摊到多卡）" if task == "full_finetune" else "（含 KV cache/激活约 30% 余量）"))

    # 单卡方案
    single = [(n, s) for n, s in GPU_DB.items() if s["vram"] >= need]
    # 多卡方案：2/4/8 卡，优先有 NVLink 的数据中心卡
    multi = []
    for n, s in GPU_DB.items():
        for k in (2, 4, 8):
            if s["vram"] * k >= need:
                multi.append((k, n, s))
                break

    def sort_key(item):
        # 推理按带宽优先（decode 是 memory-bound），训练按算力优先
        s = item[-1]
        return (-s["bw"] if task == "inference" else -s["fp16"], s["rent"])

    print("\n[单卡方案]（推荐度按" + ("带宽" if task == "inference" else "算力") + "排序）")
    if single:
        for n, s in sorted(single, key=sort_key)[:4]:
            print(f"  ✔ {n:<10} 显存 {s['vram']:>3}G  带宽 {s['bw']:.2f}TB/s  FP16 {s['fp16']:>5}TF"
                  f"  时租 ¥{s['rent'] or '-'}  单卡价 ¥{_fmt_price(s['price'])}")
    else:
        print("  ✘ 没有单卡能装下，必须多卡。")

    print("\n[多卡方案]")
    for k, n, s in sorted(multi, key=sort_key)[:5]:
        nv = f"NVLink {s['nvlink']}GB/s" if s["nvlink"] else "仅 PCIe（TP 效率低）"
        cost = f"仅卡成本约 ¥{_fmt_price(k * s['price'])}" if s["price"] else "以整机柜报价为准"
        print(f"  ✔ {k}×{n:<9} 总显存 {k * s['vram']:>4}G  {nv}  {cost}")

    # 整机配套与租买算账：选"最便宜可购买"的可行方案（性能榜首常是不单卖的旗舰）
    buyable_single = [(1, n, s) for n, s in single if s["price"] > 0]
    buyable_multi = [(k, n, s) for k, n, s in multi if s["price"] > 0]
    if buyable_single or buyable_multi:
        k, name, s = min(buyable_single + buyable_multi, key=lambda t: t[0] * t[2]["price"])
        total_vram = s["vram"] * k
        total_price = s["price"] * k * 1.5  # ×1.5 计入整机其他部件
        print(f"\n[落地成本估算] 最经济可购买方案: {f'{k}×' if k > 1 else ''}{name}")
        print(f"  仅卡成本约 ¥{_fmt_price(s['price'] * k)}，含整机约 ¥{_fmt_price(total_price)}")
        print(f"  内存 ≥ {total_vram * 2:.0f} GB（≈2× 显存）；系统盘 1TB NVMe + 数据盘 ≥2TB NVMe；")
        print(f"  电源按 TDP×数量×1.5 预留（约 {s['tdp'] * k * 1.5:.0f}W）；多卡注意 PCIe 通道数与涡轮散热。")
        if s["rent"]:
            for hours in (8, 20):
                days = total_price / (s["rent"] * k * hours)
                print(f"  租 vs 买：每天使用 {hours:>2} 小时 → 约 {days:.0f} 天（{days / 365:.1f} 年）回本；"
                      f"回本周期 >2 年则租更划算。")


def plot():
    """生成显存-带宽选型地图（需要 matplotlib）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib：pip install matplotlib 后重试。")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"consumer": "#e8734a", "datacenter": "#3a7ca5"}
    for name, s in GPU_DB.items():
        ax.scatter(s["vram"], s["bw"], s=s["fp16"] ** 0.5 * 12 + 40,
                   c=colors[s["kind"]], alpha=0.65, edgecolors="black", linewidths=0.5)
        ax.annotate(name, (s["vram"], s["bw"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    for vram, label in [(18, "7B FP16 推理"), (45, "70B INT4 推理"), (112, "7B 全参微调")]:
        ax.axvline(vram, ls="--", lw=0.8, c="gray")
        ax.text(vram + 1, 7.2, label, rotation=90, fontsize=8, color="gray")
    ax.set_xlabel("Memory (GB)")
    ax.set_ylabel("Memory Bandwidth (TB/s)")
    ax.set_title("GPU Selection Map: VRAM vs Bandwidth (bubble size = FP16 TFLOPS)")
    ax.set_ylim(0, 8.5)
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", c=colors[k], label=v)
                       for k, v in {"consumer": "Consumer", "datacenter": "Datacenter"}.items()])
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "gpu_vram_bandwidth_map.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


def main():
    p = argparse.ArgumentParser(description="GPU 选型工具（1.12 配套代码）")
    p.add_argument("--list", action="store_true", help="打印全部显卡规格表")
    p.add_argument("--params", type=float, help="模型参数量（单位：B，如 7、70）")
    p.add_argument("--task", choices=["inference", "lora", "full_finetune"], default="inference")
    p.add_argument("--dtype", choices=list(DTYPE_BYTES), default="fp16")
    p.add_argument("--plot", action="store_true", help="生成显存-带宽选型地图")
    args = p.parse_args()

    if args.list or not (args.params or args.plot):
        list_gpus()
    if args.params:
        recommend(args.params, args.task, args.dtype)
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
