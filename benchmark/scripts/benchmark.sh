#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

# Source the utils.sh file
source "$(dirname "$0")/utils.sh"

# Get environment name from command line argument
ENV_NAME=$1
if [ -z "$ENV_NAME" ]; then
    e_error "Please provide environment name as argument"
    exit 1
fi

conda activate $ENV_NAME

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${ENV_NAME}_benchmark_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Hyperparameters
MODELS=("coal" "bikes" "propensity")
TREES=(50 100 200)
PARTICLES=(20 40 60)
TUNE=(1000)
DRAWS=(1000)
BATCH=("1.0 1.0" "0.5 0.5" "0.1 0.1")
CORES=(4) # TODO!!!
ITERATIONS=1

benchmark() {
    local trees=$1
    local particles=$2
    local tune=$3
    local draws=$4
    local batch="$5"
    local model=$6
    local output_file="${RESULTS_DIR}/benchmark_t${trees}_p${particles}_tune${tune}_draws${draws}_batch${batch// /_}_model_${model}.txt"

    # Time the execution
    start_time=$(date +%s)
    python examples/bart_examples.py --model $model --trees $trees --particle $particles --tune $tune --draws $draws --batch $batch
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo $duration > "${output_file}"

    formatted_duration=$(textifyDuration $duration)
    e_success "Benchmark completed: trees=$trees, particles=$particles, tune=$tune, draws=$draws, batch=$batch, model=$model"
    e_note "Duration: $formatted_duration"
    }

    e_header "Starting BART Benchmark ($ENV_NAME Implementation)"

# Run benchmarks
for trees in "${TREES[@]}"; do
    for particles in "${PARTICLES[@]}"; do
        for tune in "${TUNE[@]}"; do
            for draws in "${DRAWS[@]}"; do
                for batch in "${BATCH[@]}"; do
                    for model in "${MODELS[@]}"; do
                        echo "Running benchmark on model=$model, trees=$trees, particles=$particles, tune=$tune, draws=$draws, batch=$batch"
                        benchmark "$trees" "$particles" "$tune" "$draws" "$batch" "$model"
                    done
                done
            done
        done
    done
done

# Aggregate results - TODO!!!
# python scripts/aggregate_results.py ${RESULTS_DIR}
