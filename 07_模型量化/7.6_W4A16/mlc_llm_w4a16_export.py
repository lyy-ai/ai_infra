# mlc_llm_w4a16_export.py
# W4A16 模型在 MLC-LLM 中的 TVM 导出与推理示例

import os
import subprocess


# ============================================================
# 第 1 部分：使用命令行导出 W4A16 TVM 模型
# ============================================================
def export_w4a16_via_cli(
    hf_model_path: str,
    output_dir: str,
    quantization: str = "q4f16_1",
    conv_template: str = "llama-2",
    target: str = "cuda",
):
    """
    使用 mlc_llm CLI 完成 gen_config -> convert -> compile。

    参数：
      hf_model_path: 本地 HuggingFace 模型目录
      output_dir:    MLC 输出目录
      quantization:  MLC 量化模式，W4A16 常用 q4f16_1
      conv_template: 对话模板，如 llama-2, qwen2, mistral
      target:        编译目标，如 cuda, metal, vulkan
    """
    os.makedirs(output_dir, exist_ok=True)

    cmds = [
        [
            "mlc_llm", "gen_config", hf_model_path,
            "--quantization", quantization,
            "--conv-template", conv_template,
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
    print(f"MLC W4A16 model exported to: {output_dir}")


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
    print("Assistant:", cm.generate("请介绍一下 W4A16 量化。"))


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
    print("Assistant: ", end="", flush=True)
    for text in engine.generate("请介绍一下 W4A16 量化。"):
        print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    # 请根据实际情况修改本地路径
    HF_MODEL = "/path/to/your-hf-model"
    MLC_DIR = "/data/ai_infra/W4A16/your-model-w4a16-mlc"

    # 1) 导出（只需执行一次，需要安装 mlc-llm）
    # export_w4a16_via_cli(HF_MODEL, MLC_DIR, quantization="q4f16_1", conv_template="llama-2")

    # 2) 推理
    # chat_with_model(MLC_DIR)
    # stream_with_model(MLC_DIR)

    print("请将 HF_MODEL 改为本地模型路径，并取消注释对应步骤的调用。")
