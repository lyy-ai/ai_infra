#!/usr/bin/env python3
"""
端侧多模态模型显存估算示例。

本脚本估算 VLM（以 Qwen2.5-VL/Qwen3.5 量级为例）在端侧部署时的内存组成：
1. LLM 权重：FP16 / INT8 / INT4
2. Vision Encoder 与 Projector 权重
3. 图像 token + 文本 token 对应的 KV Cache
4. 运行时 workspace 余量

无需 GPU，可直接运行。

运行方式：
    python examples/edge_multimodal_memory_estimator.py
"""
from dataclasses import dataclass, replace


NUM_LAYERS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_DTYPE_BYTES = 2
WORKSPACE_RATIO = 0.20


@dataclass
class VLMConfig:
    name: str
    llm_params_b: float
    vision_params_b: float
    projector_params_b: float
    llm_bits: int
    image_tokens: int
    text_tokens: int


def weight_gib(params_b: float, bits: int) -> float:
    """权重内存，按 GiB 估算。"""
    return params_b * 1e9 * bits / 8 / (1024 ** 3)


def kv_gib(tokens: int) -> float:
    """KV Cache 内存，按 GiB 估算。"""
    num_bytes = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * KV_DTYPE_BYTES * tokens
    return num_bytes / (1024 ** 3)


def image_token_count(image_size=448, patch_size=14, tiles=1) -> int:
    """估算图像 token 数，含少量特殊 token。"""
    grid = image_size // patch_size
    return tiles * grid * grid + 2


def estimate(cfg: VLMConfig) -> dict:
    """估算单个配置的总内存。"""
    llm = weight_gib(cfg.llm_params_b, cfg.llm_bits)
    vision = weight_gib(cfg.vision_params_b, 16)
    projector = weight_gib(cfg.projector_params_b, 16)
    kv = kv_gib(cfg.image_tokens + cfg.text_tokens)
    workspace = WORKSPACE_RATIO * (llm + vision + projector)
    total = llm + vision + projector + kv + workspace
    return {
        "llm": llm,
        "vision": vision,
        "projector": projector,
        "kv": kv,
        "workspace": workspace,
        "total": total,
    }


def print_estimate(cfg: VLMConfig):
    """打印配置结果。"""
    est = estimate(cfg)
    print(f"\n[{cfg.name}]")
    print(f"  llm({cfg.llm_bits}-bit): {est['llm']:.2f} GiB")
    print(f"  vision(fp16): {est['vision']:.2f} GiB")
    print(f"  projector(fp16): {est['projector']:.2f} GiB")
    print(f"  kv(image+text={cfg.image_tokens + cfg.text_tokens} tokens): {est['kv']:.2f} GiB")
    print(f"  workspace: {est['workspace']:.2f} GiB")
    print(f"  total: {est['total']:.2f} GiB")
    return est["total"]


def main():
    print("=" * 84)
    print("Edge Multimodal Memory Estimator")
    print("=" * 84)
    print(f"KV shape: layers={NUM_LAYERS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, dtype={KV_DTYPE_BYTES}B")

    base_image_tokens = image_token_count(image_size=448, patch_size=14, tiles=1)
    hires_image_tokens = image_token_count(image_size=448, patch_size=14, tiles=4)
    print(f"image tokens: 448px/1 tile={base_image_tokens}, 448px/4 tiles={hires_image_tokens}")

    configs = [
        VLMConfig("vlm_7b_fp16_lowres", 7.0, 0.6, 0.1, 16, base_image_tokens, 1024),
        VLMConfig("vlm_7b_int8_lowres", 7.0, 0.6, 0.1, 8, base_image_tokens, 1024),
        VLMConfig("vlm_7b_int4_lowres", 7.0, 0.6, 0.1, 4, base_image_tokens, 1024),
        VLMConfig("vlm_7b_int4_hires", 7.0, 0.6, 0.1, 4, hires_image_tokens, 1024),
        VLMConfig("vlm_9b_int4_lowres", 9.0, 0.6, 0.1, 4, base_image_tokens, 1024),
    ]

    totals = {}
    for cfg in configs:
        totals[cfg.name] = print_estimate(cfg)

    print("\n" + "=" * 84)
    print("Takeaway")
    print("=" * 84)
    print("- Weight-only INT4/INT8 is usually the first edge optimization for LLM weights.")
    print("- High-resolution images increase both vision prefill cost and KV cache through image tokens.")
    print("- On edge devices, memory headroom matters as much as peak TOPS because KV/workspace can dominate.")


if __name__ == "__main__":
    main()
