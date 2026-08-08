# 7.5 流水线并行：GPipe 与 1F1B 气泡率估算
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/liyangyang/qwen35_env/bin/python 6.5_流水线并行/pipeline_bubble_calc.py


def gpipe_bubble_rate(num_stages, num_micro_batches):
    return (num_stages - 1) / num_micro_batches


def one_f_one_b_bubble_rate(num_stages, num_micro_batches):
    return (num_stages - 1) / (num_micro_batches + num_stages - 1)


def main():
    for P in [2, 4, 8, 16]:
        for M in [4, 8, 16, 32]:
            gpipe = gpipe_bubble_rate(P, M)
            f1b1 = one_f_one_b_bubble_rate(P, M)
            print(f"P={P:2d} M={M:2d} | GPipe bubble={gpipe:.2%} | 1F1B bubble={f1b1:.2%}")


if __name__ == "__main__":
    main()
