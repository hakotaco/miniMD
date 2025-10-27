#!/bin/bash
#SBATCH --job-name=minimd_gprof_32t
#SBATCH --partition=cn-ib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=gprof_32thread_%j.log
#SBATCH --error=gprof_32thread_error_%j.log

# Profiling miniMD with gprof - 32 OpenMP threads
echo "=========================================="
echo "MiniMD Gprof Profiling - 32 Threads"
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
OUTPUT_DIR="profiling_gprof_32thread_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Rebuild with -pg flag for gprof
echo "=== Step 1: Rebuilding miniMD with -pg flag ==="
cd build

# Backup current build
if [ -f miniMD ]; then
    cp miniMD miniMD.backup
    echo "Backed up current miniMD executable"
fi

# Reconfigure with -pg flag
echo "Reconfiguring CMake with -pg flag..."
cmake -DCMAKE_CXX_FLAGS="-O3 -g -pg" \
      -DCMAKE_EXE_LINKER_FLAGS="-pg" \
      ..

# Rebuild
echo "Rebuilding..."
make clean
make -j 8

if [ ! -f miniMD ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo "Build successful!"
echo ""

cd $MINIMD_DIR

# OpenMP configuration for 32 threads
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Set gprof output prefix for multi-threaded profiling
# Each thread will create gmon.out.<thread_id>
export GMON_OUT_PREFIX=${OUTPUT_DIR}/gmon.out

echo "OpenMP Configuration:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo "  GMON_OUT_PREFIX=$GMON_OUT_PREFIX"
echo ""

# Input file
INPUT_FILE="in.lj.miniMD"

# Step 2: Run miniMD to generate profiling data
echo "=== Step 2: Running miniMD with gprof profiling ==="
echo "Input: $INPUT_FILE"
echo ""

# Change to output directory so gmon.out files are created there
cd $OUTPUT_DIR
..//build/miniMD -i ../$INPUT_FILE --ntypes 1

echo ""
echo "Execution complete. Gmon.out files generated."
ls -lh gmon.out* 2>/dev/null || ls -lh ../gmon.out 2>/dev/null || echo "Warning: No gmon.out files found!"
echo ""

# Step 3: Merge gmon.out files if multiple exist (for multi-threaded runs)
echo "=== Step 3: Processing gprof data ==="

# Count gmon files
GMON_COUNT=$(ls gmon.out* 2>/dev/null | wc -l)

if [ $GMON_COUNT -eq 0 ]; then
    # Check parent directory
    if [ -f ../gmon.out ]; then
        echo "Found gmon.out in parent directory"
        mv ../gmon.out ./gmon.out.merged
        GMON_COUNT=1
    else
        echo "ERROR: No gmon.out files found!"
        exit 1
    fi
elif [ $GMON_COUNT -eq 1 ]; then
    echo "Single gmon.out file found"
    mv gmon.out* gmon.out.merged
else
    echo "Multiple gmon.out files found ($GMON_COUNT files)"
    echo "Merging gmon.out files..."
    
    # Use gprof -s to sum the profile data
    gprof -s ../build/miniMD gmon.out*
    
    if [ -f gmon.sum ]; then
        mv gmon.sum gmon.out.merged
        echo "Merged into gmon.out.merged"
    else
        echo "Merge failed, using first gmon.out file"
        cp gmon.out.0 gmon.out.merged 2>/dev/null || cp $(ls gmon.out* | head -1) gmon.out.merged
    fi
fi

# Step 4: Generate reports
echo ""
echo "=== Step 4: Generating gprof reports ==="

# Flat profile
echo "Generating flat profile..."
gprof -b -p ../build/miniMD gmon.out.merged > gprof_flat_profile.txt

# Call graph
echo "Generating call graph..."
gprof -b -q ../build/miniMD gmon.out.merged > gprof_call_graph.txt

# Full report
echo "Generating full report..."
gprof ../build/miniMD gmon.out.merged > gprof_full_report.txt

# Annotated source (if available)
echo "Generating annotated listing..."
gprof -A ../build/miniMD gmon.out.merged > gprof_annotated.txt 2>/dev/null || echo "Annotated listing not available"

# Brief summary
echo "Generating brief summary..."
gprof -b ../build/miniMD gmon.out.merged | head -100 > gprof_summary.txt

echo ""
echo "=== Step 5: Restoring original build ==="
cd ../build
if [ -f miniMD.backup ]; then
    mv miniMD miniMD.gprof
    mv miniMD.backup miniMD
    echo "Restored original miniMD executable"
    echo "Gprof-enabled executable saved as miniMD.gprof"
fi

cd $MINIMD_DIR

echo ""
echo "Profiling complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - gmon.out.merged: Merged profiling data from all threads"
echo "  - gprof_flat_profile.txt: Function-level time breakdown"
echo "  - gprof_call_graph.txt: Call graph with time attribution"
echo "  - gprof_full_report.txt: Complete profiling report"
echo "  - gprof_summary.txt: Brief summary of top functions"
echo ""
echo "Top 10 time-consuming functions:"
head -30 ${OUTPUT_DIR}/gprof_flat_profile.txt | tail -20
echo ""
echo "Job finished: $(date)"
