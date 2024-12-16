.PHONY: all setup benchmark_python benchmark_rust

# Anaconda-related variables
CONDA_ACTIVATE = soure $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_PYTHON_ENV = bart_python
CONDA_RUST_ENV = bart_rust

BENCHMARK_SCRIPT = scripts/benchmark.sh

all: setup benchmark_python benchmark_rust

setup: create_python_env create_rust_env

create_python_env:
	@if ! conda info --envs | grep -q "$(CONDA_PYTHON_ENV)"; then \
		echo "Creating environment $(CONDA_PYTHON_ENV)"; \
		conda env create -f environments/bart_python.yml -n $(CONDA_PYTHON_ENV); \
	else \
		echo "Environment $(CONDA_PYTHON_ENV) already exists"; \
	fi

create_rust_env:
	@if ! conda info --envs | grep -q "$(CONDA_RUST_ENV)"; then \
		echo "Creating environment $(CONDA_RUST_ENV)"; \
		conda env create -f environments/bart_rust.yml -n $(CONDA_RUST_ENV); \
	else \
		echo "Environment $(CONDA_RUST_ENV) already exists"; \
	fi

benchmark_python:
	bash ${BENCHMARK_SCRIPT} ${CONDA_PYTHON_ENV}

benchmark_rust:
	bash ${BENCHMARK_SCRIPT} ${CONDA_RUST_ENV}
