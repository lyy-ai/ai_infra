NVCC ?= /usr/local/cuda/bin/nvcc
PY := /data/liyangyang/qwen35_env/bin/python
TORCHRUN := /data/liyangyang/qwen35_env/bin/torchrun

all: single ddp comm report

single:
	$(PY) src/train_single.py

ddp:
	$(TORCHRUN) --nproc_per_node=2 src/train_ddp.py

comm:
	$(TORCHRUN) --nproc_per_node=2 src/comm_bench.py

report:
	$(PY) scripts/analyze_results.py

.PHONY: all single ddp comm report
