#!/bin/bash
# 4.9 编译耗时优化方案：分析每个 .cc 源文件的编译阶段耗时
#
# 运行：
#   cd /data/liyangyang/ai_infra/03_AI编译器
#   bash 3.9_编译耗时优化方案/compile_time_analysis.sh /path/to/cc_sources
#
# 输出：file, cc->out(s), out->aom(s), total(s)
set -e

SRC_DIR=${1:-.}

echo "file, cc->out(s), out->aom(s), total(s)"
for cc in "$SRC_DIR"/*.cc; do
    [ -e "$cc" ] || continue
    base=$(basename "$cc" .cc)
    start=$(date +%s)
    g++ -O0 -fPIC "$cc" -o "$SRC_DIR/$base.out"
    out_end=$(date +%s)
    # 占位：打包 .aom，实际项目中替换为真实打包命令
    aom_end=$(date +%s)
    echo "$base, $((out_end - start)), $((aom_end - out_end)), $((aom_end - start))"
done
