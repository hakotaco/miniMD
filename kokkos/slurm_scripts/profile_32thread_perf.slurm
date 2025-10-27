#!/bin/bash
#SBATCH --job-name=minimd_perf_32t
#SBATCH --partition=cn-ib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=perf_32thread_%j.log
#SBATCH --error=perf_32thread_error_%j.log

# Profiling miniMD with perf - 32 OpenMP threads
echo "=========================================="
echo "MiniMD Perf Profiling - 32 Threads"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load GCC
module load CMake

# Set directory
MINIMD_DIR="/home/akaushik/miniMD/kokkos"
cd $MINIMD_DIR

# Create output directory
OUTPUT_DIR="profiling_perf_32thread_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo ""

# OpenMP configuration for 32 threads
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "OpenMP Configuration:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo ""

# Input file
INPUT_FILE="in.lj.miniMD"

echo "Running miniMD with perf profiling..."
echo "Input: $INPUT_FILE"
echo ""

# Run with perf record - detailed profiling
echo "=== Running perf record with call-graph ==="
perf record -F 999 -g --call-graph dwarf -o ${OUTPUT_DIR}/perf.data \
    ./build/miniMD -i $INPUT_FILE --ntypes 1

echo ""
echo "=== Running perf stat for hardware counters ==="
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,branches,branch-misses \
    -o ${OUTPUT_DIR}/perf_stat.txt \
    ./build/miniMD -i $INPUT_FILE --ntypes 1

echo ""
echo "=== Generating perf report ==="
perf report -i ${OUTPUT_DIR}/perf.data --stdio > ${OUTPUT_DIR}/perf_report.txt

echo ""
echo "=== Generating annotated source ==="
perf annotate -i ${OUTPUT_DIR}/perf.data --stdio > ${OUTPUT_DIR}/perf_annotate.txt

echo ""
echo "=== Generating flame graph data ==="
perf script -i ${OUTPUT_DIR}/perf.data > ${OUTPUT_DIR}/perf_script.txt

echo ""
echo "=== Summary of top functions ==="
echo "Top 20 functions by CPU time:"
perf report -i ${OUTPUT_DIR}/perf.data --stdio -n --sort comm,dso,symbol | head -50 | tee ${OUTPUT_DIR}/perf_top_functions.txt

echo ""
echo "Profiling complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - perf.data: Raw profiling data"
echo "  - perf_report.txt: Detailed performance report"
echo "  - perf_stat.txt: Hardware counter statistics"
echo "  - perf_annotate.txt: Annotated source code with performance data"
echo "  - perf_script.txt: Flame graph data (can be used with flamegraph.pl)"
echo ""
echo "Job finished: $(date)"
