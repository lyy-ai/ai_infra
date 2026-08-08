# tradeoff_visualization.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_quantization_tradeoff():
    """绘制不同量化方案的压缩比 vs 精度损失散点图"""

    # (方案名, 压缩比, 相对精度损失百分比)
    methods = [
        ("FP16", 1.0, 0.0),
        ("INT8", 2.0, 0.5),
        ("W8A16", 2.0, 0.3),
        ("QAT INT8", 2.0, 0.1),
        ("W4A16", 3.2, 1.0),
        ("AWQ", 3.2, 0.5),
        ("QAT W4A16", 3.2, 0.4),
        ("W4A8", 4.0, 2.0),
        ("KV Cache INT8", 2.0, 0.5),  # 仅 KV Cache
        ("KIVI 2-bit", 8.0, 2.5),    # 仅 KV Cache
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, compression, loss in methods:
        ax.scatter(compression, loss, s=150)
        ax.annotate(name, (compression, loss), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Compression Ratio (higher is better)")
    ax.set_ylabel("Relative Quality Loss % (lower is better)")
    ax.set_title("Quantization Methods: Compression vs Quality Trade-off")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.invert_yaxis()

    # 添加理想区域提示
    ax.axvline(x=3.0, color='green', linestyle='--', alpha=0.3, label="Common target zone")
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("quantization_tradeoff.png")
    print("Saved quantization trade-off plot to quantization_tradeoff.png")


if __name__ == "__main__":
    plot_quantization_tradeoff()
