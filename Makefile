# Makefile for benchmarking Rust vs pure-Python PGBART implementations.
#
# Creates isolated uv virtual environments for each implementation,
# installs dependencies, runs bench_runner.py, and compares results.
#
# Usage:
#   make bench              # full pipeline: setup both envs, run, compare
#   make bench-rust         # run only the Rust benchmark
#   make bench-python       # run only the Python benchmark
#   make compare            # compare existing result files
#   make clean              # remove benchmark venvs and result files
#
# Configuration (override on the command line):
#   make bench MODEL=propensity TREES=20 PARTICLES=20 STEPS=50

MODEL     ?= coal
TREES     ?= 50
PARTICLES ?= 10
STEPS     ?= 20
WARMUP    ?= 5
PYTHON    ?= 3.12

PROJECT_ROOT := $(CURDIR)
RUST_VENV    := $(PROJECT_ROOT)/.venv-bench-rust
PYTHON_VENV  := $(PROJECT_ROOT)/.venv-bench-python
RESULTS_DIR  := bench_results

RUST_JSON    := $(RESULTS_DIR)/rust_$(MODEL).json
PYTHON_JSON  := $(RESULTS_DIR)/python_$(MODEL).json
COMPARE_JSON := $(RESULTS_DIR)/comparison_$(MODEL).json

RUNNER       := examples/bench_runner.py
COMPARATOR   := examples/bench_rust_vs_python.py

BENCH_ARGS   := --model $(MODEL) --trees $(TREES) --particles $(PARTICLES) \
                --steps $(STEPS) --warmup $(WARMUP)

.PHONY: bench bench-rust bench-python compare clean setup-rust setup-python

bench: bench-rust bench-python compare

compare: $(RUST_JSON) $(PYTHON_JSON)
	@echo "=== Comparing results ==="
	python $(COMPARATOR) $(RUST_JSON) $(PYTHON_JSON) -o $(COMPARE_JSON)

# Rust environment
setup-rust: $(RUST_VENV)/.installed

$(RUST_VENV)/.installed: pyproject.toml Cargo.toml src/*.rs
	@echo "=== Setting up Rust environment ==="
	uv venv $(RUST_VENV) --python $(PYTHON) --quiet
	uv pip install --python $(RUST_VENV)/bin/python maturin --quiet
	VIRTUAL_ENV=$(RUST_VENV) $(RUST_VENV)/bin/maturin develop --release --uv 2>&1 | tail -1
	uv pip install --python $(RUST_VENV)/bin/python pymc pandas --quiet
	@touch $@

bench-rust: setup-rust | $(RESULTS_DIR)
	@echo "=== Running Rust benchmark (model=$(MODEL)) ==="
	$(RUST_VENV)/bin/python $(RUNNER) $(BENCH_ARGS) > $(RUST_JSON)
	@echo "  -> $(RUST_JSON)"

# Python environment
setup-python: $(PYTHON_VENV)/.installed

$(PYTHON_VENV)/.installed: pymc-bart/setup.py pymc-bart/requirements.txt
	@echo "=== Setting up Python environment ==="
	uv venv $(PYTHON_VENV) --python $(PYTHON) --quiet
	uv pip install --python $(PYTHON_VENV)/bin/python \
		-e ./pymc-bart --quiet
	@touch $@

bench-python: setup-python | $(RESULTS_DIR)
	@echo "=== Running Python benchmark (model=$(MODEL)) ==="
	$(PYTHON_VENV)/bin/python $(RUNNER) $(BENCH_ARGS) > $(PYTHON_JSON)
	@echo "  -> $(PYTHON_JSON)"

# Delete results
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

clean:
	rm -rf $(RUST_VENV) $(PYTHON_VENV) $(RESULTS_DIR)
