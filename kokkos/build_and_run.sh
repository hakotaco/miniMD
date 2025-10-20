#!/bin/bash
#
# Automated build script for miniMD-Kokkos variant
# See BUILD_INSTRUCTIONS.md for detailed explanation
#

set -e  # Exit on error

echo "=========================================="
echo "miniMD-Kokkos Build Script"
echo "=========================================="
echo ""

# Step 1: Load required modules
echo "[1/7] Loading required modules..."
module load GCC
module load CMake
module load OpenMPI
echo "  ✓ Modules loaded"
echo ""

# Step 2: Check prerequisites
echo "[2/7] Checking prerequisites..."
echo "  GCC version: $(g++ --version | head -1)"
echo "  CMake version: $(cmake --version | head -1)"
echo "  MPI compiler: $(which mpicxx)"
echo "  Architecture: $(uname -m)"
echo ""

# Step 3: Install Kokkos (if not already installed)
KOKKOS_INSTALL_DIR="$HOME/kokkos-install"
if [ ! -d "$KOKKOS_INSTALL_DIR" ]; then
    echo "[3/7] Installing Kokkos..."
    cd ~
    
    # Clone Kokkos
    if [ ! -d "kokkos-source" ]; then
        echo "  Cloning Kokkos 4.4.01..."
        git clone --depth 1 --branch 4.4.01 https://github.com/kokkos/kokkos.git kokkos-source
    fi
    
    # Build Kokkos
    mkdir -p kokkos-build
    cd kokkos-build
    
    echo "  Configuring Kokkos..."
    cmake ../kokkos-source \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ENABLE_SERIAL=ON \
        -DKokkos_ARCH_ARMV8_THUNDERX2=ON \
        -DCMAKE_CXX_COMPILER=g++ > /dev/null
    
    echo "  Building Kokkos (this may take a few minutes)..."
    make -j 8 > /dev/null
    make install > /dev/null
    
    echo "  ✓ Kokkos installed to $KOKKOS_INSTALL_DIR"
else
    echo "[3/7] Kokkos already installed at $KOKKOS_INSTALL_DIR"
    echo "  ✓ Skipping Kokkos build"
fi
echo ""

# Step 4: Build miniMD
MINIMD_DIR="/home/akaushik/miniMD/kokkos"
echo "[4/7] Building miniMD..."
cd $MINIMD_DIR

# Clean previous build if requested
if [ "$1" = "clean" ]; then
    echo "  Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

echo "  Configuring miniMD..."
cmake .. \
    -DKokkos_ROOT=$KOKKOS_INSTALL_DIR \
    -DCMAKE_CXX_COMPILER=mpicxx > /dev/null

echo "  Compiling miniMD..."
make -j 8

echo "  ✓ miniMD built successfully"
echo "  Executable: $MINIMD_DIR/build/miniMD"
echo ""

# Step 5: Verify build
echo "[5/7] Verifying build..."
if [ -x "$MINIMD_DIR/build/miniMD" ]; then
    echo "  ✓ Executable exists and is executable"
    echo "  Size: $(du -h $MINIMD_DIR/build/miniMD | cut -f1)"
else
    echo "  ✗ Build failed - executable not found"
    exit 1
fi
echo ""

# Step 6: Set environment for running
echo "[6/7] Setting up runtime environment..."
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
echo "  ✓ OpenMP environment configured"
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "    OMP_PROC_BIND=$OMP_PROC_BIND"
echo ""

# Step 7: Run test
echo "[7/7] Running test case..."
cd $MINIMD_DIR
echo "  Command: ./build/miniMD -i in.lj.miniMD -s 20 -n 100"
echo ""
echo "========== Test Output =========="
./build/miniMD -i in.lj.miniMD -s 20 -n 100
echo "================================="
echo ""

echo "✓ Build and test completed successfully!"
echo ""
echo "To run with profiling:"
echo "  cd $MINIMD_DIR"
echo "  perf stat ./build/miniMD -i in.lj.miniMD -s 30 -n 1000"
echo "  perf record -g ./build/miniMD -i in.lj.miniMD -s 40 -n 1000"
echo "  perf report"
echo ""
echo "See BUILD_INSTRUCTIONS.md for more details."
