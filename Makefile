# Variables C API
CLA_BUILD_PATH := build-cla
CLA_SRC_PATH := cla
TOOLCHAIN_WIN := toolchain/win-mingw.cmake
CLA_BUILD_WIN_PATH := build-cla-win
CLA_TEST_TARGET := $(CLA_BUILD_PATH)/test_suite

# Variable Python API
PYCLA_DIST := dist

# Run all steps to build cla and pycla (TODO)
all: prepare-cla compile-cla

# Create new target and run tests
test: all test-cla

# Clean any build files
clean:
	@echo "[Makefile] Clean project files..."
	@rm -rf ${CLA_BUILD_PATH} ${CLA_BUILD_WIN_PATH} ${PYCLA_DIST}

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
	@chmod +x $(CLA_TEST_TARGET)
	@./$(CLA_TEST_TARGET)

# Compile for windows
compile-windows:
	@echo "[Makefile] Compile for Windows..."
	@cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=`pwd`/$(TOOLCHAIN_WIN) -B $(CLA_BUILD_WIN_PATH) -S $(CLA_SRC_PATH)
	@cmake --build $(CLA_BUILD_WIN_PATH)

