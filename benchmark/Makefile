.PHONY: all setup benchmark-python

CONDA_ACTIVATE = soure $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_PYTHON_ENV = bart_python
CONDA_RUST_ENV = bart_rust

all: setup benchmark_python

setup: create_python_env

create_python_env:
	@if ! conda info --envs | grep -q "$(CONDA_PYTHON_ENV)"; then \
		echo "Creating environment $(CONDA_PYTHON_ENV)"; \
		conda env create -f environments/bart_python.yml -n $(CONDA_PYTHON_ENV); \
	else \
		echo "Environment $(CONDA_PYTHON_ENV) already exists"; \
	fi

benchmark_python:
	bash scripts/python_benchmark.sh

# benchmark-rust:
# 	bash scripts/benchmark_rust.sh

# clean:
# 	rm -rf results/*
# 	conda env remove -n bart_python
# 	conda env remove -n bart_rust
