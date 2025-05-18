# Variables C API
CLA_BUILD_PATH := build-cla
CLA_SRC_PATH := cla
TOOLCHAIN_WIN := toolchain/win-mingw.cmake
CLA_BUILD_WIN_PATH := build-cla-win
CLA_TEST_TARGET := $(CLA_BUILD_PATH)/test_suite
CLA_TEST_MEM_LEAK_TARGET := $(CLA_BUILD_PATH)/memory_leak
CLA_TEST_MEM_STABLE_TARGET := $(CLA_BUILD_PATH)/memory_stability
CLA_VALGRIND_SUPP := valgrind_cudart.supp

# Variable Python API
PYCLA_DIST := dist
PYCLA_TEST_DIR := test/python
PYCLA_CLA_BIN := pycla/bin
PYCLA_BUILD := build

# Utility variables
CUDA_COMPUTE_SANITIZER := /opt/cuda/extras/compute-sanitizer/compute-sanitizer

# Run all steps to build cla and pycla
all: prepare-cla compile-cla copy-cla-bin-pycla 

# Create new target and run tests
test: all test-cla test-pycla

# Create release
release: test pack-release-cla pack-release-pycla

# Clean any build files
clean:
	@echo "[Makefile] Clean project files..."
	@rm -rf $(CLA_BUILD_PATH) $(CLA_BUILD_WIN_PATH) $(PYCLA_DIST)
	@rm -rf $(PYCLA_DIST) $(PYCLA_BUILD)
	@rm -f $(PYCLA_CLA_BIN)/libcla.so*

# Generate cla project files with Ninja
prepare-cla:
	@echo "[Makefile] Generating project files..."
	@cmake -G "Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -B $(CLA_BUILD_PATH) -S $(CLA_SRC_PATH)

# Build all targets from cla generated build files
compile-cla:
	@echo "[Makefile] Building all targets..."
	@cmake --build $(CLA_BUILD_PATH)
 
# Run cla tests
test-cla:
	@echo "[Makefile] Running test target..."
	@./$(CLA_TEST_TARGET)

test-cla-memory-leak:
	@echo "[Makefile] Running memory leak tests with Valgrind/compute_sanitizer..."
	@for e in vector matrix; do for d in CPU GPU; do valgrind --leak-check=yes --suppressions=$(CLA_VALGRIND_SUPP) $(CLA_TEST_MEM_LEAK_TARGET) $$e $$d; $(CUDA_COMPUTE_SANITIZER) $(CLA_TEST_MEM_LEAK_TARGET) $$e $$d; done; done

test-cla-memory-stability:
	@echo "[Makefile] Starting interactive memory stability test"
	@./$(CLA_TEST_MEM_STABLE_TARGET)

# Pack into release
pack-release-cla:
	@echo "[Makefile] Creating release for CLA..."
	@echo "[Makefile] Parsing release version for CLA..."
	@awk '/project\(cla/,/CUDA C\)/' $(CLA_SRC_PATH)/CMakeLists.txt | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' > make_cla_version
	@echo "[Makefile] Packing Linux cla build..."
	@rm -f cla-linux-`cat make_cla_version`.zip
	@cp $(CLA_BUILD_PATH)/libcla.so* .
	@zip cla-linux-`cat make_cla_version`.zip $(CLA_SRC_PATH)/include/*.h libcla.so*
	@rm libcla.so*
	@echo "[Makefile] Packed Linux cla build."
	@echo "[Makefile] Cleaning up..."
	@rm make_cla_version

pack-release-pycla:
	@echo "[Makefile] Creating release for pycla..."
	@python -m build -w -o $(PYCLA_DIST) 

# Copy named library to pycla
copy-cla-bin-pycla:
	@echo "[Makefile] Copying CLA shared library to pycla..."
	@cp $(CLA_BUILD_PATH)/libcla.so* $(PYCLA_CLA_BIN)

# Test pycla
test-pycla:
	@echo "[Makefile] Testing pycla..."
	@pytest -x $(PYCLA_TEST_DIR)
	@echo "[Makefile] Testing older Python versions..."
	@python ./tools/ensure_compatibility.py

# Publish pycla
publish-pycla:
	@echo "[Makefile] Publish pycla to PyPI..."
	@twine upload $(PYCLA_DIST)/*
