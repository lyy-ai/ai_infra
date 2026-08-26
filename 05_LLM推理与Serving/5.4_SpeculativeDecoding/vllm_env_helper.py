#!/usr/bin/env python3
"""
vLLM 环境初始化辅助模块。

部分 vLLM 0.25.0 版本在运行时依赖 nvidia-cu13 提供的 libcudart.so.13，
但系统默认的 LD_LIBRARY_PATH 可能无法找到该库。本模块在导入 vllm 之前，
自动完成以下环境设置：

1. 设置 LD_LIBRARY_PATH，让 vLLM 及其子进程能找到 CUDA 13 动态库。
2. 预加载 libcudart.so.13 和 libnvrtc.so.13 到当前进程。
3. 将当前虚拟环境的 bin 目录加入 PATH，确保 ninja 等工具可找到。
4. 在主进程选择空闲显存最多的 GPU；vLLM 子进程直接继承该设置。

使用方法：
    import vllm_env_helper  # 必须放在 import vllm 之前
    from vllm import LLM, SamplingParams

注意：本模块只处理环境配置问题，vLLM 本身需要单独安装。
"""
import ctypes
import glob
import multiprocessing
import os
import subprocess
import sys


_VLLM_SETUP_DONE = False


def setup_cuda_visible_devices():
    """在主进程选择空闲显存最多的 GPU 作为 vLLM 使用的设备；子进程直接继承该环境变量。"""
    # 子进程继承主进程设置，不需要重新选择
    if multiprocessing.current_process().name != "MainProcess":
        return

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mibs = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
        if not free_mibs:
            return
        best_idx = max(range(len(free_mibs)), key=lambda i: free_mibs[i])

        current = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if current:
            # 如果已指定 GPU，仅当该 GPU 不是空闲显存最多的设备时才覆盖
            try:
                selected = int(current.split(",")[0])
                if selected == best_idx:
                    return
            except ValueError:
                pass

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    except Exception:
        # 如果 nvidia-smi 不可用，则保持默认行为
        pass


def setup_vllm_env():
    """
    检测并加载 vLLM 运行所需的 libcudart.so.13，并确保 PATH 能找到 venv 内的工具（如 ninja）。

    如果 vllm 已经可以正常导入，则不做任何操作。
    """
    global _VLLM_SETUP_DONE
    if _VLLM_SETUP_DONE:
        return True

    # 在主进程选择空闲显存最多的 GPU；子进程继承该环境变量
    setup_cuda_visible_devices()

    # 搜索 nvidia/cu13 的 lib 目录
    search_patterns = [
        os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cu13/lib"),
    ]
    lib_dirs = []
    for pattern in search_patterns:
        lib_dirs.extend(glob.glob(pattern))

    if lib_dirs:
        lib_dir = lib_dirs[0]
        # 设置 LD_LIBRARY_PATH，以便 vLLM 子进程也能找到 CUDA 13 库
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in current_ld_path.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + current_ld_path if current_ld_path else "")

    # 确保 venv 的 bin 目录在 PATH 中，vLLM/flashinfer 编译内核时需要调用 ninja
    venv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    # 先尝试直接导入
    try:
        import vllm  # noqa: F401
        _VLLM_SETUP_DONE = True
        return True
    except ImportError as e:
        err_msg = str(e)
        if "libcudart.so.13" not in err_msg and "_C_stable_libtorch" not in err_msg:
            # 不是已知的库路径问题，直接报错
            print(f"Error: vLLM is not installed.\n{err_msg}")
            print("Please install it with: /data/qwen35_env/bin/pip install vllm")
            return False

    if not lib_dirs:
        print("Error: vLLM is not installed or cannot find required CUDA runtime library.")
        print("Please install it with: /data/qwen35_env/bin/pip install vllm")
        return False

    lib_dir = lib_dirs[0]
    # 预加载 libcudart.so.13 和 libnvrtc.so.13（torchcodec / vllm 可能依赖）
    libs_to_preload = ["libcudart.so.13", "libnvrtc.so.13"]
    for lib_name in libs_to_preload:
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"Warning: failed to preload {lib_path}: {e}")
                # 继续尝试，也许该库不是必需的

    # 再次尝试导入 vllm
    try:
        import vllm  # noqa: F401
        _VLLM_SETUP_DONE = True
        return True
    except ImportError as e:
        print(f"Error: vLLM still cannot be loaded after setting up library path.\n{e}")
        return False


# 模块被导入时自动执行 setup
setup_vllm_env()
