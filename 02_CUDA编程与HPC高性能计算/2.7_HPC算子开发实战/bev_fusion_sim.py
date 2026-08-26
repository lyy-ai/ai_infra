# 3.7 HPC 算子开发实战：BEV 特征融合分块思想模拟
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.7_HPC算子开发实战/bev_fusion_sim.py
import numpy as np


def naive_fusion(features):
    """朴素实现：逐个元素相加"""
    out = np.zeros_like(features[0])
    for f in features:
        out += f
    return out


def tiled_fusion(features, tile_h=8, tile_w=8):
    """分块实现：每次处理 tile 大小块，便于映射到共享内存"""
    H, W, C = features[0].shape
    out = np.zeros_like(features[0])
    for th in range(0, H, tile_h):
        for tw in range(0, W, tile_w):
            tile = np.zeros((tile_h, tile_w, C))
            for f in features:
                h_slice = slice(th, min(th + tile_h, H))
                w_slice = slice(tw, min(tw + tile_w, W))
                tile[:h_slice.stop - h_slice.start, :w_slice.stop - w_slice.start, :] += f[h_slice, w_slice, :]
            out[th:th + tile_h, tw:tw + tile_w, :] = tile
    return out


def main():
    H, W, C = 64, 64, 128
    features = [np.random.randn(H, W, C).astype(np.float32) for _ in range(4)]

    out1 = naive_fusion(features)
    out2 = tiled_fusion(features)

    diff = np.max(np.abs(out1 - out2))
    print(f"max diff: {diff:.6f}")


if __name__ == "__main__":
    main()
