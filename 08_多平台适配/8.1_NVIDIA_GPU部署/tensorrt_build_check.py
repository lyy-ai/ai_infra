# 9.1 NVIDIA GPU 部署：TensorRT / GPU 环境检查
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.1_NVIDIA_GPU部署/tensorrt_build_check.py

import sys


def check_tensorrt():
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("TensorRT not installed (expected on CPU-only env).")
        return False


def check_torch_cuda():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:    {torch.version.cuda}")
            print(f"GPU count:       {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except ImportError:
        print("PyTorch not installed.")
        return False


def main():
    print("=== NVIDIA GPU / TensorRT environment check ===")
    print()
    check_torch_cuda()
    print()
    check_tensorrt()
    print()
    print("Typical TensorRT-LLM build command (for reference):")
    print("  python build.py --model_dir ./llama-7b-hf --dtype float16 \\")
    print("                  --use_gpt_attention --use_gemm_plugin float16 \\")
    print("                  --output_dir ./llama_7b_trt_engines")


if __name__ == "__main__":
    main()
