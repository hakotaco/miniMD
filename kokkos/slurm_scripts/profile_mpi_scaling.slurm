#!/bin/bash
#SBATCH --job-name=miniMD_mpi_profile
#SBATCH --partition=cn-ib
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=profiling_mpi_output_%j.log
#SBATCH --error=profiling_mpi_error_%j.log

echo "=========================================="
echo "miniMD Multi-Process MPI Profiling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
module load GCC
module load CMake
module load OpenMPI

# Set OpenMP environment (1 thread per MPI rank)
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Configuration
PROBLEM_SIZE=40        # 40^3 = 64,000 atoms per process with 4 ranks = 256,000 total
NUM_STEPS=1000
INPUT_FILE="in.lj.miniMD"
MINIMD_DIR="/home/akaushik/miniMD/kokkos"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${MINIMD_DIR}/profiling_mpi_${TIMESTAMP}_job${SLURM_JOB_ID}"

# Create results directory
mkdir -p $RESULTS_DIR

echo "Results directory: $RESULTS_DIR"
echo ""
echo "Configuration:"
echo "  MPI ranks: $SLURM_NTASKS"
echo "  OMP threads per rank: $OMP_NUM_THREADS"
echo "  Problem size: ${PROBLEM_SIZE}x${PROBLEM_SIZE}x${PROBLEM_SIZE}"
echo "  Timesteps: $NUM_STEPS"
echo "  Input: $INPUT_FILE"
echo ""

cd $MINIMD_DIR

# Save system information
cat > $RESULTS_DIR/system_info.txt << EOF
System Information
==================
Date: $(date)
Hostname: $(hostname)
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST
MPI Ranks: $SLURM_NTASKS
Threads per rank: $OMP_NUM_THREADS

CPU Info:
$(lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|Core|Socket")

Compiler:
$(g++ --version | head -1)

MPI:
$(mpirun --version | head -1)

Environment:
OMP_NUM_THREADS=$OMP_NUM_THREADS
OMP_PROC_BIND=$OMP_PROC_BIND
OMP_PLACES=$OMP_PLACES
EOF

echo "Collecting system information..."
echo "  ✓ System info saved"

# Test different MPI process counts
MPI_COUNTS="2 4 8 16 32"

for NP in $MPI_COUNTS; do
    if [ $NP -gt $SLURM_NTASKS ]; then
        echo "Skipping np=$NP (exceeds allocated tasks: $SLURM_NTASKS)"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Running with $NP MPI processes"
    echo "=========================================="
    
    RUN_DIR="$RESULTS_DIR/np_${NP}"
    mkdir -p $RUN_DIR
    
    # 1. Baseline run
    echo "[1/3] Baseline performance (np=$NP)..."
    mpirun -np $NP ./build/miniMD -i $INPUT_FILE -s $PROBLEM_SIZE -n $NUM_STEPS \
        > $RUN_DIR/baseline_output.txt 2>&1
    
    # Extract performance summary
    grep "PERF_SUMMARY" $RUN_DIR/baseline_output.txt > $RUN_DIR/baseline_summary.txt || true
    
    T_TOTAL=$(grep PERF_SUMMARY $RUN_DIR/baseline_output.txt | awk '{print $6}')
    T_FORCE=$(grep PERF_SUMMARY $RUN_DIR/baseline_output.txt | awk '{print $7}')
    T_NEIGH=$(grep PERF_SUMMARY $RUN_DIR/baseline_output.txt | awk '{print $8}')
    T_COMM=$(grep PERF_SUMMARY $RUN_DIR/baseline_output.txt | awk '{print $9}')
    PERF=$(grep PERF_SUMMARY $RUN_DIR/baseline_output.txt | awk '{print $11}')
    
    echo "  Total time: ${T_TOTAL}s"
    echo "    Force:    ${T_FORCE}s ($(echo "scale=1; $T_FORCE*100/$T_TOTAL" | bc 2>/dev/null || echo "N/A")%)"
    echo "    Neighbor: ${T_NEIGH}s ($(echo "scale=1; $T_NEIGH*100/$T_TOTAL" | bc 2>/dev/null || echo "N/A")%)"
    echo "    Comm:     ${T_COMM}s ($(echo "scale=1; $T_COMM*100/$T_TOTAL" | bc 2>/dev/null || echo "N/A")%)"
    echo "    Performance: ${PERF} atom-steps/s"
    
    # 2. Profile rank 0 with perf record
    echo "[2/3] Profiling with perf record (rank 0 only)..."
    mpirun -np $NP bash -c "
        if [ \$OMPI_COMM_WORLD_RANK -eq 0 ]; then
            perf record -g -F 999 -o $RUN_DIR/perf_rank0.data \
                ./build/miniMD -i $INPUT_FILE -s $PROBLEM_SIZE -n $NUM_STEPS > /dev/null 2>&1
        else
            ./build/miniMD -i $INPUT_FILE -s $PROBLEM_SIZE -n $NUM_STEPS > /dev/null 2>&1
        fi
    "
    
    if [ -f "$RUN_DIR/perf_rank0.data" ]; then
        echo "  Generating perf report..."
        perf report -i $RUN_DIR/perf_rank0.data --stdio -n --sort symbol \
            > $RUN_DIR/profile_cycles_by_symbol.txt 2>&1
        
        # Show top 5 functions
        echo "  Top 5 hottest functions (rank 0):"
        head -50 $RUN_DIR/profile_cycles_by_symbol.txt | grep -E "^\s+[0-9]" | head -5 | \
            awk '{printf "    %6.2f%% %s\n", $1, $NF}'
    else
        echo "  Warning: perf data file not created"
    fi
    
    # 3. Hardware counters with perf stat (rank 0 only)
    echo "[3/3] Collecting hardware counters..."
    mpirun -np $NP bash -c "
        if [ \$OMPI_COMM_WORLD_RANK -eq 0 ]; then
            perf stat -d ./build/miniMD -i $INPUT_FILE -s $PROBLEM_SIZE -n $NUM_STEPS \
                2>&1 | tee $RUN_DIR/perf_stat_output.txt > /dev/null
        else
            ./build/miniMD -i $INPUT_FILE -s $PROBLEM_SIZE -n $NUM_STEPS > /dev/null 2>&1
        fi
    " > /dev/null 2>&1
    
    echo "  ✓ Profiling complete for np=$NP"
done

# Generate comparison summary
echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

cat > $RESULTS_DIR/SUMMARY.txt << 'EOFSUM'
========================================
miniMD Multi-Process MPI Profiling Summary
========================================
Date: $(date)
Problem: 40^3 atoms per process, 1000 timesteps

SCALING RESULTS
===============
EOFSUM

echo "MPI_Ranks  Total_Time(s)  Force(s)  Neighbor(s)  Comm(s)  Performance(atom-steps/s)  Speedup  Efficiency(%)" \
    >> $RESULTS_DIR/SUMMARY.txt

BASELINE_TIME=""
for NP in $MPI_COUNTS; do
    if [ $NP -gt $SLURM_NTASKS ]; then
        continue
    fi
    
    RUN_DIR="$RESULTS_DIR/np_${NP}"
    if [ -f "$RUN_DIR/baseline_summary.txt" ]; then
        T_TOTAL=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $6}')
        T_FORCE=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $7}')
        T_NEIGH=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $8}')
        T_COMM=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $9}')
        PERF=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $11}')
        
        if [ -z "$BASELINE_TIME" ]; then
            BASELINE_TIME=$T_TOTAL
        fi
        
        SPEEDUP=$(echo "scale=2; $BASELINE_TIME/$T_TOTAL" | bc 2>/dev/null || echo "N/A")
        EFFICIENCY=$(echo "scale=1; $SPEEDUP*100/$NP" | bc 2>/dev/null || echo "N/A")
        
        printf "%9d  %13.2f  %8.2f  %11.2f  %7.2f  %25.0f  %7s  %13s\n" \
            $NP $T_TOTAL $T_FORCE $T_NEIGH $T_COMM $PERF $SPEEDUP $EFFICIENCY \
            >> $RESULTS_DIR/SUMMARY.txt
    fi
done

cat >> $RESULTS_DIR/SUMMARY.txt << 'EOFSUM2'

FILES GENERATED (per MPI rank count)
====================================
np_X/baseline_output.txt           - Full miniMD output
np_X/baseline_summary.txt          - Performance summary
np_X/perf_rank0.data              - perf record data (rank 0)
np_X/profile_cycles_by_symbol.txt - Function profile
np_X/perf_stat_output.txt         - Hardware counters

For interactive viewing:
  perf report -i <results_dir>/np_X/perf_rank0.data

EOFSUM2

cat $RESULTS_DIR/SUMMARY.txt

echo ""
echo "=========================================="
echo "✓ Multi-process profiling complete!"
echo "=========================================="
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Quick comparison (if multiple ranks tested):"
for NP in $MPI_COUNTS; do
    if [ $NP -gt $SLURM_NTASKS ]; then
        continue
    fi
    RUN_DIR="$RESULTS_DIR/np_${NP}"
    if [ -f "$RUN_DIR/baseline_summary.txt" ]; then
        T_TOTAL=$(grep PERF_SUMMARY $RUN_DIR/baseline_summary.txt | awk '{print $6}')
        echo "  np=$NP: ${T_TOTAL}s"
    fi
done
echo ""
