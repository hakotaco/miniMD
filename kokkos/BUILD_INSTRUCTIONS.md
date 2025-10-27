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

## Build Instructions

### Step 1: Load Required Modules

Load the required modules:

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

