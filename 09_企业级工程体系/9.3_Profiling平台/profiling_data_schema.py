# 10.3 Profiling 平台：统一事件 Schema
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系
#   /data/liyangyang/qwen35_env/bin/python 9.3_Profiling平台/profiling_data_schema.py

import json
from dataclasses import dataclass, asdict


@dataclass
class ProfileEvent:
    run_id: str
    commit: str
    platform: str
    event_type: str
    name: str
    start_us: int
    duration_us: int
    stream: int = 0


def serialize_events(events):
    return json.dumps([asdict(e) for e in events], indent=2, ensure_ascii=False)


def main():
    events = [
        ProfileEvent("run-001", "a1b2c3d", "A100", "kernel", "gemm_fp16", 1000, 500, 0),
        ProfileEvent("run-001", "a1b2c3d", "A100", "memcpy", "H2D", 1600, 100, 1),
        ProfileEvent("run-001", "a1b2c3d", "A100", "kernel", "softmax", 1800, 80, 0),
    ]
    print(serialize_events(events))


if __name__ == "__main__":
    main()
