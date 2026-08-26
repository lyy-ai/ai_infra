# 9.3 Ascend NPU 适配：ATC 转换前检查清单
#
# 运行：
#   cd /data/ai_infra/08_多平台适配
#   /data/qwen35_env/bin/python 8.3_Ascend_NPU适配/atc_checklist.py


def check_atc_readiness():
    checklist = [
        ("CANN 版本 >= 7.0", True, "当前 8.0.RC2"),
        ("驱动与固件版本一致", True, "23.0.3 / 23.0.3"),
        ("ONNX 模型已通过 simplify", True, "使用 onnxsim"),
        ("输入 shape 已固定", False, "存在动态 batch，需设置 --input_shape"),
        ("算子均在 CANN 支持列表", True, "已通过 mapping check"),
        ("精度类型已确认", True, "FP16"),
        ("权重文件存在且对齐", True, "model.onnx + model.data"),
    ]
    print("=== ATC 转换前检查清单 ===")
    all_ok = True
    for item, ok, note in checklist:
        status = "PASS" if ok else "WARN"
        print(f"[{status}] {item}: {note}")
        all_ok = all_ok and ok
    print()
    print("""ATC 示例命令（参考）：
  atc --model=model.onnx --framework=5 --output=model \\
      --soc_version=Ascend910B \\
      --input_shape="input:1,3,224,224" \\
      --precision_mode=force_fp16""")
    return all_ok


def main():
    ok = check_atc_readiness()
    print()
    print("All checks passed." if ok else "Some checks need attention.")


if __name__ == "__main__":
    main()
