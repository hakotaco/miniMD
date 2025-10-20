# Building and Running miniMD Kokkos Variant

## System Information
- **Platform**: ARM Kunpeng-920 (aarch64) HPC Cluster
- **OS**: Red Hat Enterprise Linux 8
- **Compiler**: GCC 8.5.0 (system) / GCC 14.1.0 (module)
- **Date**: October 13, 2025

## Prerequisites

### Required Software
- GCC 8.5+ (Kokkos requires C++14 support, minimum GCC 5.3)
- CMake 3.16+
- MPI implementation (OpenMPI, MPICH, etc.)
- Git

### Minimum Versions
- **g++**: 5.3+ (we have 8.5/14.1)
- **CMake**: 3.16+ (we have 3.29.2)
- **Kokkos**: 4.4.01 (what we'll install)

## Step-by-Step Build Instructions

### Step 1: Load Required Modules

If on an HPC system with environment modules, load the required modules:

```bash
# Load GCC compiler (if not using system default)
module load GCC

# Load CMake
module load CMake

# Load MPI library (required for miniMD)
module load OpenMPI
```

**Verify the loaded environment:**
```bash
g++ --version      # Should show GCC 8.5+ or higher
cmake --version    # Should show 3.16+
which mpicc        # Should find MPI C compiler
which mpicxx       # Should find MPI C++ compiler
lscpu              # Check CPU architecture
```

### Step 2: Install Kokkos

Kokkos is a performance portability library that miniMD-Kokkos depends on.

#### 2.1 Clone Kokkos Source

```bash
cd ~
git clone --depth 1 --branch 4.4.01 https://github.com/kokkos/kokkos.git kokkos-source
```

**Note**: Using `--depth 1` for a shallow clone to save space. Version 4.4.01 is the latest stable release as of this date.

#### 2.2 Create Build and Install Directories

```bash
cd ~
mkdir -p kokkos-build kokkos-install
```

**Directory Structure:**
- `kokkos-source/`: Source code
- `kokkos-build/`: Build directory (can be deleted after installation)
- `kokkos-install/`: Installation directory (keep this)

#### 2.3 Configure Kokkos with CMake

```bash
cd ~/kokkos-build

cmake ../kokkos-source \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/kokkos-install \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_ARMV8_THUNDERX2=ON \
  -DCMAKE_CXX_COMPILER=g++
```

**CMake Options Explained:**
- `CMAKE_BUILD_TYPE=Release`: Optimized build with `-O3` flags
- `CMAKE_INSTALL_PREFIX`: Where to install Kokkos
- `Kokkos_ENABLE_OPENMP=ON`: Enable OpenMP backend for multi-threading
- `Kokkos_ENABLE_SERIAL=ON`: Enable serial execution (useful for debugging)
- `Kokkos_ARCH_ARMV8_THUNDERX2=ON`: Optimize for ARM Kunpeng-920 architecture
- `CMAKE_CXX_COMPILER=g++`: Use g++ compiler


#### 2.4 Build and Install Kokkos

```bash
cd ~/kokkos-build
make -j 8          # Use 8 parallel jobs (adjust based on your system)
make install
```

**Verify Installation:**
```bash
ls ~/kokkos-install/include/Kokkos_Core.hpp    # Should exist
ls ~/kokkos-install/lib64/libkokkoscore.a      # Should exist
```

### Step 3: Build miniMD-Kokkos

#### 3.1 Navigate to miniMD Kokkos Directory

```bash
cd /home/akaushik/miniMD/kokkos
```

#### 3.2 Create Build Directory

```bash
mkdir -p build
cd build
```

#### 3.3 Configure miniMD with CMake

```bash
cmake .. \
  -DKokkos_ROOT=~/kokkos-install \
  -DCMAKE_CXX_COMPILER=mpicxx
```

**CMake Options Explained:**
- `Kokkos_ROOT`: Path to Kokkos installation
- `CMAKE_CXX_COMPILER=mpicxx`: Use MPI C++ compiler wrapper (required for MPI support)

**Alternative: Build without MPI (Serial Only)**

If you want to build without MPI for single-node testing, you can use MPI stubs:
```bash
cmake .. \
  -DKokkos_ROOT=~/kokkos-install \
  -DMPI_STUBS=ON
```

#### 3.4 Compile miniMD

```bash
make -j 8
```

**Verify Build:**
```bash
ls miniMD          # Should show executable
file miniMD        # Shows it's an ARM executable
./miniMD --help    # Display help information
```

The executable will be located at: `/home/akaushik/miniMD/kokkos/build/miniMD`

### Step 4: Run a Test Case

#### 4.1 Set OpenMP Environment Variables

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

**Environment Variables Explained:**
- `OMP_NUM_THREADS`: Number of OpenMP threads per MPI rank
- `OMP_PROC_BIND=spread`: Bind threads to spread across cores
- `OMP_PLACES=threads`: Bind to hardware threads

#### 4.2 Run a Small Test (Serial - 1 MPI rank)

```bash
cd /home/akaushik/miniMD/kokkos
./build/miniMD -i in.lj.miniMD -s 20 -n 100
```

**Command-line Options:**
- `-i in.lj.miniMD`: Input file (Lennard-Jones potential)
- `-s 20`: Problem size (20×20×20 unit cells = 32,000 atoms)
- `-n 100`: Number of timesteps

**Expected Output:**
```
# Create System:
# Done .... 
# miniMD-Reference 1.2 (MPI+OpenMP) output ...
# Run Settings: 
        # MPI processes: 1
        # Host Threads: 1
        # Atoms: 32000
        # System size: 33.59 33.59 33.59 (unit cells: 20 20 20)
# Starting dynamics ...
# Timestep T U P Time
0 1.440000e+00 -6.773368e+00 -5.019707e+00  0.000
100 7.310774e-01 -5.712185e+00 1.204501e+00  [time]
```

#### 4.3 Run with Multiple MPI Ranks

```bash
mpirun -np 4 ./build/miniMD -i in.lj.miniMD -s 32 -n 100
```

**MPI Options:**
- `-np 4`: Use 4 MPI processes

#### 4.4 Run with MPI + OpenMP Hybrid

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./build/miniMD -t 4 -i in.lj.miniMD -s 40 -n 500
```

**Hybrid Parallelism:**
- 2 MPI ranks × 4 OpenMP threads = 8 total threads
- `-t 4`: Set 4 OpenMP threads per MPI rank

### Step 5: Profile with perf

#### 5.1 Profile with perf stat (Quick Overview)

```bash
cd /home/akaushik/miniMD/kokkos

# Basic statistics
perf stat ./build/miniMD -i in.lj.miniMD -s 30 -n 1000

# Detailed statistics with cache metrics
perf stat -d ./build/miniMD -i in.lj.miniMD -s 30 -n 1000

# Very detailed statistics
perf stat -d -d -d ./build/miniMD -i in.lj.miniMD -s 30 -n 1000
```

**What perf stat shows:**
- Execution time
- Instructions executed
- CPU cycles
- Cache misses (L1, L2, L3)
- Branch mispredictions
- IPC (Instructions Per Cycle)

#### 5.2 Profile with perf record (Detailed Profiling)

```bash
# Record profile data with call graphs
perf record -g ./build/miniMD -i in.lj.miniMD -s 40 -n 1000

# Record specific events
perf record -g -e cycles,instructions,cache-misses \
  ./build/miniMD -i in.lj.miniMD -s 40 -n 1000

# View the profile report
perf report

# View annotated source (if compiled with debug symbols)
perf annotate

# Generate flamegraph-friendly output
perf script > perf.data.script
```

**Useful perf record options:**
- `-g`: Record call graphs (stack traces)
- `-e events`: Specify hardware events to record
- `--call-graph dwarf`: Better call graph accuracy

#### 5.3 Profile MPI Runs

For MPI applications, profile a specific rank or all ranks:

```bash
# Profile only rank 0
mpirun -np 4 bash -c \
  'if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then \
     perf record -g -o perf.data.rank0 ./build/miniMD -i in.lj.miniMD -s 40 -n 1000; \
   else \
     ./build/miniMD -i in.lj.miniMD -s 40 -n 1000; \
   fi'

# Profile all ranks (creates separate files)
mpirun -np 4 perf record -g -o perf.data.\$OMPI_COMM_WORLD_RANK \
  ./build/miniMD -i in.lj.miniMD -s 40 -n 1000
```

#### 5.4 Recommended Test Cases for Profiling

**Small Test (Quick, for debugging):**
```bash
perf record -g ./build/miniMD -i in.lj.miniMD -s 20 -n 100
```

**Medium Test (Good for profiling):**
```bash
perf record -g ./build/miniMD -i in.lj.miniMD -s 40 -n 1000
```

**Large Test (Production-like workload):**
```bash
perf record -g ./build/miniMD -i in.lj.miniMD -s 60 -n 5000
```

**EAM Potential (More complex physics):**
```bash
perf record -g ./build/miniMD -i in.eam.miniMD -s 30 -n 1000
```

### Step 6: Analyzing Performance

#### 6.1 Understanding perf report Output

```bash
perf report --stdio
```

Look for:
- **Hot functions**: Functions consuming most CPU time
- **Call chains**: Where hot functions are called from
- **Cache misses**: Functions with high cache miss rates

#### 6.2 Key Performance Metrics

From miniMD's performance output:
```
# Performance Summary:
# MPI_proc OMP_threads nsteps natoms t_total t_force t_neigh t_comm t_other
```

**Metrics to track:**
- `t_force`: Time in force calculation (main computational kernel)
- `t_neigh`: Time in neighbor list construction
- `t_comm`: Time in MPI communication
- `performance`: Atom-timesteps per second

#### 6.3 Performance Engineering Targets

Focus optimization on:
1. **Force calculation** (`force_lj.cpp` or `force_eam.cpp`) - Usually 60-70% of time
2. **Neighbor list** (`neighbor.cpp`) - Usually 20-30% of time
3. **Communication** (`comm.cpp`) - Depends on MPI scaling

## Available Input Files

Located in `/home/akaushik/miniMD/kokkos/`:

### Lennard-Jones Potential
- `in.lj.miniMD` - LJ potential, generated atoms
- `in.lj.lammps` - LJ potential, LAMMPS format
- `in.lj-data.miniMD` - LJ potential, read from data file

### EAM Potential (More realistic, more expensive)
- `in.eam.miniMD` - EAM potential for metals (Cu)
- `in.eam.lammps` - EAM potential, LAMMPS format
- `in.eam-data.miniMD` - EAM potential, read from data file
- **Note**: Requires `Cu_u6.eam` file in the same directory

## Troubleshooting

### Issue: "mpi.h: No such file or directory"
**Solution**: Load MPI module or use MPI compiler wrapper:
```bash
module load OpenMPI
cmake .. -DKokkos_ROOT=~/kokkos-install -DCMAKE_CXX_COMPILER=mpicxx
```

### Issue: "Kokkos::OpenMP::initialize WARNING"
**Solution**: Set OpenMP environment variables:
```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

### Issue: Slow performance
**Check:**
1. Built with `-O3` optimization (Release build)
2. Using correct architecture flag in Kokkos build
3. OpenMP threads set appropriately (`OMP_NUM_THREADS`)
4. Not oversubscribing cores (threads × ranks ≤ physical cores)

### Issue: perf: Permission denied
**Solution**: May need to adjust perf paranoid level:
```bash
# Check current setting
cat /proc/sys/kernel/perf_event_paranoid

# If it's 3 or higher, contact system administrator
# Or use sudo (if you have permissions):
sudo sysctl -w kernel.perf_event_paranoid=1
```

## Performance Engineering Ideas

For your course project, consider these optimization strategies:

### 1. **Vectorization**
- Add SIMD directives to force loops
- Align data structures
- Use compiler reports: `-fopt-info-vec`

### 2. **Cache Optimization**
- Improve data locality in neighbor lists
- Reorder atom data by spatial location
- Tile computations

### 3. **Thread Parallelism**
- Experiment with different OpenMP schedules
- Try different Kokkos execution policies
- Profile thread scaling

### 4. **Memory Optimization**
- Reduce memory allocations in hot paths
- Use memory pools
- Optimize data layout (AoS vs SoA)

### 5. **Algorithm Changes**
- Experiment with neighbor list cutoffs
- Try different binning strategies
- Optimize force cutoff calculations

## Quick Reference Commands

```bash
# Build everything from scratch
cd ~
module load GCC CMake OpenMPI
git clone --depth 1 --branch 4.4.01 https://github.com/kokkos/kokkos.git kokkos-source
mkdir -p kokkos-build kokkos-install
cd kokkos-build
cmake ../kokkos-source -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/kokkos-install \
  -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_ARMV8_THUNDERX2=ON -DCMAKE_CXX_COMPILER=g++
make -j 8 && make install

cd /home/akaushik/miniMD/kokkos
mkdir -p build && cd build
cmake .. -DKokkos_ROOT=~/kokkos-install -DCMAKE_CXX_COMPILER=mpicxx
make -j 8

# Run and profile
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
cd /home/akaushik/miniMD/kokkos
perf record -g ./build/miniMD -i in.lj.miniMD -s 40 -n 1000
perf report
```

## Additional Resources

- **Kokkos Documentation**: https://kokkos.org/kokkos-core-wiki/
- **miniMD on GitHub**: https://github.com/Mantevo/miniMD
- **LAMMPS Documentation**: https://docs.lammps.org/ (for understanding MD concepts)
- **perf Tutorial**: https://perf.wiki.kernel.org/index.php/Tutorial
- **Intel VTune**: Alternative profiler (if available)
- **ARM Forge**: ARM-specific profiling tools

## Contact Information

For questions about this build:
- Kokkos issues: https://github.com/kokkos/kokkos/issues
- miniMD issues: https://github.com/Mantevo/miniMD/issues

---
**Build Date**: October 13, 2025  
**Built By**: akaushik  
**System**: ARM Kunpeng-920 HPC Cluster
