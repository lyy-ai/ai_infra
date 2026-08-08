from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_tokenizer(model_path):
    """加载 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path, quant_config=None, device_map="auto"):
    """加载模型，支持可选量化配置"""
    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        # 默认使用 FP16/BF16，节省显存
        kwargs["torch_dtype"] = torch.float16
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def get_int8_config():
    """INT8 量化配置（bitsandbytes）"""
    return BitsAndBytesConfig(load_in_8bit=True)


def get_int4_config():
    """INT4 NF4 量化配置（bitsandbytes）"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def get_quantization_config(name):
    """根据名称获取量化配置"""
    if name == "int8":
        return get_int8_config()
    elif name == "int4":
        return get_int4_config()
    elif name == "fp16" or name is None:
        return None
    else:
        raise ValueError(f"Unsupported quantization: {name}")
