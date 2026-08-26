# mlc_llm_qwen_demo.py
# Qwen3.5-35B-A3B-GPTQ-Int4 在 MLC-LLM 中的导出与推理示例

import os
import subprocess


# ============================================================
# 第 1 部分：MLC-LLM 命令行导出 Qwen3.5-35B-A3B-GPTQ-Int4
# ============================================================
def convert_via_cli(
    hf_model_path: str,
    output_dir: str,
    quantization: str = "q4f16_1",
    target: str = "cuda",
):
    """
    使用 mlc_llm 命令行完成：gen_config -> convert -> compile

    参数说明：
      hf_model_path: 本地 HuggingFace 模型目录
      output_dir:    MLC 模型输出目录
      quantization:  MLC 量化模式，W4A16 常用 q4f16_1 或 q4f16_2
      target:        编译目标，如 cuda、metal、vulkan、android 等
    """
    os.makedirs(output_dir, exist_ok=True)

    cmds = [
        [
            "mlc_llm", "gen_config", hf_model_path,
            "--quantization", quantization,
            "--conv-template", "qwen3_5",
            "--output", output_dir,
        ],
        [
            "mlc_llm", "convert", hf_model_path,
            "--quantization", quantization,
            "--output", output_dir,
        ],
        [
            "mlc_llm", "compile", output_dir,
            "--device", target,
            "--output", os.path.join(output_dir, "lib.so"),
        ],
    ]

    for cmd in cmds:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"MLC model exported to: {output_dir}")


# ============================================================
# 第 2 部分：Python API 推理
# ============================================================
def chat_with_model(mlc_model_dir: str):
    """使用 ChatModule 做同步推理"""
    try:
        from mlc_llm import ChatModule
    except ImportError as e:
        print(f"无法导入 mlc_llm: {e}")
        return

    cm = ChatModule(
        model=mlc_model_dir,
        device="cuda",
    )

    prompts = [
        "请用一句话介绍 Qwen3.5 MoE 模型。",
        "W4A8 量化的优缺点是什么？",
    ]
    for p in prompts:
        print(f"\nUser: {p}")
        print(f"Assistant: {cm.generate(p)}")


def stream_with_model(mlc_model_dir: str):
    """使用 MLCEngine 做流式推理"""
    try:
        from mlc_llm import MLCEngine
    except ImportError as e:
        print(f"无法导入 MLCEngine: {e}")
        return

    engine = MLCEngine(
        model=mlc_model_dir,
        device="cuda",
    )
    print("\nUser: 你好，请介绍一下自己。")
    print("Assistant: ", end="", flush=True)
    for text in engine.generate("你好，请介绍一下自己。"):
        print(text, end="", flush=True)
    print()


# ============================================================
# 第 3 部分：直接读取 config 查看量化参数（无需 mlc-llm 安装）
# ============================================================
def inspect_quantization_config(hf_model_path: str):
    """打印 HuggingFace 模型的 quantization_config"""
    import json

    config_path = os.path.join(hf_model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"config.json 不存在: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})
    print("=" * 50)
    print("Quantization Config")
    print("=" * 50)
    for k, v in qc.items():
        print(f"{k}: {v}")

    text_cfg = config.get("text_config", {})
    print("\n" + "=" * 50)
    print("Key Text Model Config")
    print("=" * 50)
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_experts", "num_experts_per_tok", "vocab_size"]:
        print(f"{k}: {text_cfg.get(k)}")


if __name__ == "__main__":
    # 请根据实际情况修改本地路径
    HF_MODEL = "/path/to/Qwen3.5-35B-A3B-GPTQ-Int4"
    MLC_DIR = "/data/ai_infra/07_模型量化/7.7_W4A8/qwen3.5-35b-a3b-gptq-int4-mlc"

    # 1) 查看模型量化配置（无需安装 mlc-llm）
    inspect_quantization_config(HF_MODEL)

    # 2) 导出（只需执行一次，需要安装 mlc-llm 和 CUDA 环境）
    # convert_via_cli(HF_MODEL, MLC_DIR, quantization="q4f16_1")

    # 3) 推理（需要 mlc-llm 和已导出的 MLC_DIR）
    # chat_with_model(MLC_DIR)
    # stream_with_model(MLC_DIR)

    print("\n请将 HF_MODEL 改为本地模型路径，并取消注释对应步骤的调用。")
