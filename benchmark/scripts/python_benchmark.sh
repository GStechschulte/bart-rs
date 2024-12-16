#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bart_python

# Source the utils.sh file
source "$(dirname "$0")/utils.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/bart_python_benchmark_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Hyperparameters
TREES=(50 100 200)
PARTICLES=(20 40 60)
TUNE=(1000)
DRAWS=(1000)
BATCH=("1.0 1.0" "0.5 0.5" "0.1 0.1")
CORES=(4) # TODO!!!
ITERATIONS=5

benchmark() {
    local trees=$1
    local particles=$2
    local tune=$3
    local draws=$4
    local batch="$5"
    local iter=$6
    local output_file="${RESULTS_DIR}/benchmark_t${trees}_p${particles}_tune${tune}_draws${draws}_batch${batch// /_}_iter${iter}.txt"

    # Time the execution
        start_time=$(date +%s)
        python examples/bart_biking.py --trees $trees --particle $particles --tune $tune --draws $draws --batch $batch
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        echo $duration > "${output_file}"

        formatted_duration=$(textifyDuration $duration)
        e_success "Benchmark completed: trees=$trees, particles=$particles, tune=$tune, draws=$draws, batch=$batch, iteration=$iter"
        e_note "Duration: $formatted_duration"
    }

    e_header "Starting BART Benchmark (Python Implementation)"

# Run benchmarks
for trees in "${TREES[@]}"; do
    for particles in "${PARTICLES[@]}"; do
        for tune in "${TUNE[@]}"; do
            for draws in "${DRAWS[@]}"; do
                for batch in "${BATCH[@]}"; do
                    echo "Running Python implementation benchmark with trees=$trees, particles=$particles, tune=$tune, draws=$draws, batch=$batch"
                    for iter in $(seq 1 $ITERATIONS); do
                        benchmark "$trees" "$particles" "$tune" "$draws" "$batch" "$iter"
                    done
                done
            done
        done
    done
done

# Aggregate results - TODO!!!
# python scripts/aggregate_results.py ${RESULTS_DIR}
