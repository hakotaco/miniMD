#!/bin/bash
#SBATCH --job-name=miniMD_hybrid_omp
#SBATCH --partition=cn-ib
#SBATCH --nodes=1
#SBATCH --ntasks=2               # Max 2 MPI tasks for OpenMP-heavy configs
#SBATCH --cpus-per-task=16       # For up to 16 threads per task
#SBATCH --time=03:00:00
#SBATCH --output=profiling_hybrid_omp_%j.log
#SBATCH --error=profiling_hybrid_omp_error_%j.log

# Hybrid MPI+OpenMP Profiling - OpenMP-Heavy Configurations
# Tests: 2×8, 1×16 (pure OpenMP)

echo "=========================================="
echo "miniMD Hybrid Profiling - OpenMP-Heavy Tests"
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
TOTAL_CORES=16

# Change to miniMD directory
cd $MINIMD_DIR

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${MINIMD_DIR}/profiling_hybrid_omp_${TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR

echo "Results directory: $RESULTS_DIR"
echo "Total cores per run: $TOTAL_CORES"
echo ""

# Define test configurations: (MPI_ranks, OpenMP_threads)
declare -a CONFIGS=(
    "2:8"     # Heavy OpenMP
    "1:16"    # Pure OpenMP
)

echo "Testing OpenMP-heavy configurations:"
for config in "${CONFIGS[@]}"; do
    MPI_RANKS=$(echo $config | cut -d':' -f1)
    OMP_THREADS=$(echo $config | cut -d':' -f2)
    echo "  - $MPI_RANKS MPI ranks × $OMP_THREADS OpenMP threads = $((MPI_RANKS * OMP_THREADS)) cores"
done
echo ""

# Function to run a single configuration
run_configuration() {
    local MPI_RANKS=$1
    local OMP_THREADS=$2
    local CONFIG_DIR="$RESULTS_DIR/np${MPI_RANKS}_nt${OMP_THREADS}"
    
    mkdir -p $CONFIG_DIR
    
    echo "=========================================="
    echo "Configuration: $MPI_RANKS MPI × $OMP_THREADS OpenMP"
    echo "=========================================="
    echo "Start time: $(date)"
    
    # Set OpenMP environment
    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads
    
    echo "Running baseline performance test..."
    mpirun -np $MPI_RANKS \
        --map-by ppr:${MPI_RANKS}:node:PE=${OMP_THREADS} \
        --bind-to core \
        ./build/miniMD \
        -i $INPUT_FILE \
        -s $PROBLEM_SIZE \
        -n $NUM_STEPS \
        -t $OMP_THREADS \
        > $CONFIG_DIR/baseline_output.txt 2>&1
    
    # Extract summary
    grep "PERF_SUMMARY" $CONFIG_DIR/baseline_output.txt > $CONFIG_DIR/baseline_summary.txt 2>&1 || true
    
    # Extract key metrics
    if [ -f $CONFIG_DIR/baseline_summary.txt ]; then
        TOTAL_TIME=$(awk '{print $6}' $CONFIG_DIR/baseline_summary.txt)
        FORCE_TIME=$(awk '{print $7}' $CONFIG_DIR/baseline_summary.txt)
        NEIGH_TIME=$(awk '{print $8}' $CONFIG_DIR/baseline_summary.txt)
        COMM_TIME=$(awk '{print $9}' $CONFIG_DIR/baseline_summary.txt)
        THROUGHPUT=$(awk '{print $11}' $CONFIG_DIR/baseline_summary.txt)
        
        echo "Results:"
        echo "  Total time: ${TOTAL_TIME}s"
        echo "  Force: ${FORCE_TIME}s"
        echo "  Neighbor: ${NEIGH_TIME}s"
        echo "  Comm: ${COMM_TIME}s"
        echo "  Throughput: ${THROUGHPUT} atom-steps/s"
    else
        echo "  ERROR: Failed to complete"
    fi
    
    echo "End time: $(date)"
    echo ""
}

# Run all configurations
for config in "${CONFIGS[@]}"; do
    MPI_RANKS=$(echo $config | cut -d':' -f1)
    OMP_THREADS=$(echo $config | cut -d':' -f2)
    run_configuration $MPI_RANKS $OMP_THREADS
done

# Generate summary
echo "=========================================="
echo "Generating summary report..."
echo "=========================================="

cat > $RESULTS_DIR/SUMMARY_OMP_HEAVY.txt << 'EOF'
========================================
miniMD Hybrid Profiling - OpenMP-Heavy Tests
========================================
Date: $(date)
Problem: 40^3 atoms (256,000 total), 1000 timesteps
Total cores per run: 16

HYBRID SCALING RESULTS (OpenMP-Heavy)
======================================
MPI×OMP    Total_Time(s)  Force(s)  Neighbor(s)  Comm(s)  Throughput(atom-steps/s)  Speedup  Notes
EOF

# Collect results
BASELINE_TIME=""
for config in "${CONFIGS[@]}"; do
    MPI_RANKS=$(echo $config | cut -d':' -f1)
    OMP_THREADS=$(echo $config | cut -d':' -f2)
    CONFIG_DIR="$RESULTS_DIR/np${MPI_RANKS}_nt${OMP_THREADS}"
    
    if [ -f $CONFIG_DIR/baseline_summary.txt ]; then
        TOTAL_TIME=$(awk '{print $6}' $CONFIG_DIR/baseline_summary.txt)
        FORCE_TIME=$(awk '{print $7}' $CONFIG_DIR/baseline_summary.txt)
        NEIGH_TIME=$(awk '{print $8}' $CONFIG_DIR/baseline_summary.txt)
        COMM_TIME=$(awk '{print $9}' $CONFIG_DIR/baseline_summary.txt)
        THROUGHPUT=$(awk '{print $11}' $CONFIG_DIR/baseline_summary.txt)
        
        if [ -z "$BASELINE_TIME" ]; then
            BASELINE_TIME=$TOTAL_TIME
        fi
        
        SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $TOTAL_TIME" | bc)
        
        if [ $MPI_RANKS -eq 1 ]; then
            NOTE="Pure OpenMP"
        else
            NOTE="Hybrid"
        fi
        
        printf "%2dx%-2d   %10.2f   %8.2f   %8.2f   %8.2f   %15.0f      %4.2fx    %s\n" \
            $MPI_RANKS $OMP_THREADS $TOTAL_TIME $FORCE_TIME $NEIGH_TIME $COMM_TIME \
            $THROUGHPUT $SPEEDUP "$NOTE" >> $RESULTS_DIR/SUMMARY_OMP_HEAVY.txt
    else
        printf "%2dx%-2d   FAILED\n" $MPI_RANKS $OMP_THREADS >> $RESULTS_DIR/SUMMARY_OMP_HEAVY.txt
    fi
done

cat >> $RESULTS_DIR/SUMMARY_OMP_HEAVY.txt << 'EOF'

NOTE: This is part 2 of 2 (OpenMP-heavy configurations)
Combine with results from profile_hybrid_mpi_heavy.slurm for complete analysis
EOF

echo ""
cat $RESULTS_DIR/SUMMARY_OMP_HEAVY.txt
echo ""
echo "Job finished: $(date)"
