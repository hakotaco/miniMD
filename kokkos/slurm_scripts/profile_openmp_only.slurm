#!/bin/bash
#SBATCH --job-name=miniMD_2omp
#SBATCH --partition=cn-ib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32               
#SBATCH --time=05:00:00          # 5 hours
#SBATCH --output=profiling_2openmp_%j.log
#SBATCH --error=profiling_2openmp_error_%j.log

# Pure OpenMP Scaling Study for miniMD
# Tests 1 MPI process with varying OpenMP thread counts

echo "=========================================="
echo "miniMD Pure OpenMP Scaling Study"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module purge
module load GCC
module load CMake
module load OpenMPI

# Configuration
PROBLEM_SIZE=40
NUM_STEPS=1000
INPUT_FILE="in.lj.miniMD"
MINIMD_DIR="/home/akaushik/miniMD/kokkos"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${MINIMD_DIR}/profiling_openmp_${TIMESTAMP}_job${SLURM_JOB_ID}"

# Change to miniMD directory
cd $MINIMD_DIR

# Create results directory
mkdir -p $RESULTS_DIR

echo "Results directory: $RESULTS_DIR"
echo "Testing pure OpenMP scaling with 1 MPI process"
echo ""

# Thread counts to test
THREAD_COUNTS=(1 2 4 8 16 32)

echo "Testing thread counts: ${THREAD_COUNTS[@]}"
echo ""

# Function to run with specific thread count
run_openmp_test() {
    local THREADS=$1
    local CONFIG_DIR="$RESULTS_DIR/threads_${THREADS}"
    
    mkdir -p $CONFIG_DIR
    
    echo "=========================================="
    echo "Testing with $THREADS OpenMP threads"
    echo "=========================================="
    echo "Start time: $(date)"
    
    # Set OpenMP environment
    export OMP_NUM_THREADS=$THREADS
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads
    
    # Run with 1 MPI process
    ./build/miniMD \
        -i $INPUT_FILE \
        -s $PROBLEM_SIZE \
        -n $NUM_STEPS \
        -t $THREADS \
        > $CONFIG_DIR/output.txt 2>&1
    
    # Extract summary
    grep "PERF_SUMMARY" $CONFIG_DIR/output.txt > $CONFIG_DIR/summary.txt 2>&1 || true
    
    # Show results
    if [ -f $CONFIG_DIR/summary.txt ]; then
        TOTAL_TIME=$(awk '{print $6}' $CONFIG_DIR/summary.txt)
        THROUGHPUT=$(awk '{print $11}' $CONFIG_DIR/summary.txt)
        echo "  Time: ${TOTAL_TIME}s"
        echo "  Throughput: ${THROUGHPUT} atom-steps/s"
    else
        echo "  ERROR: Failed"
    fi
    
    echo "End time: $(date)"
    echo ""
}

# Run all tests
for threads in "${THREAD_COUNTS[@]}"; do
    run_openmp_test $threads
done

# Generate summary
echo "=========================================="
echo "Generating summary..."
echo "=========================================="

cat > $RESULTS_DIR/SUMMARY.txt << 'EOF'
========================================
miniMD Pure OpenMP Scaling Summary
========================================
Problem: 40^3 atoms (256,000), 1000 timesteps
Configuration: 1 MPI process, varying OpenMP threads

OPENMP SCALING RESULTS
======================
Threads  Time(s)  Throughput  Speedup  Efficiency(%)
EOF

BASELINE_TIME=""
for threads in "${THREAD_COUNTS[@]}"; do
    CONFIG_DIR="$RESULTS_DIR/threads_${threads}"
    
    if [ -f $CONFIG_DIR/summary.txt ]; then
        TOTAL_TIME=$(awk '{print $6}' $CONFIG_DIR/summary.txt)
        THROUGHPUT=$(awk '{print $11}' $CONFIG_DIR/summary.txt)
        
        if [ -z "$BASELINE_TIME" ]; then
            BASELINE_TIME=$TOTAL_TIME
        fi
        
        SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $TOTAL_TIME" | bc)
        EFFICIENCY=$(echo "scale=1; 100 * $SPEEDUP / $threads" | bc)
        
        printf "%7d %8.2f %12.0f   %6.2fx      %5.1f%%\n" \
            $threads $TOTAL_TIME $THROUGHPUT $SPEEDUP $EFFICIENCY >> $RESULTS_DIR/SUMMARY.txt
    else
        printf "%7d   FAILED\n" $threads >> $RESULTS_DIR/SUMMARY.txt
    fi
done

cat >> $RESULTS_DIR/SUMMARY.txt << 'EOF'

ANALYSIS
========
- Speedup shows how much faster than single-threaded
- Efficiency shows parallel overhead (100% = perfect, <70% = poor)

EOF

echo ""
cat $RESULTS_DIR/SUMMARY.txt
echo ""
echo "Job finished: $(date)"
