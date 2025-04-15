# CMake policy
cmake_minimum_required(VERSION 3.20)

# Setting project name, version and language
project(cla
        VERSION 0.2.1
        LANGUAGES CUDA C)

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Find CUDA headers (some packages install to /opt)
include_directories(SYSTEM "/opt/cuda/include")

# Configure CUDA architectures and flags
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Add library cla
add_library(${CMAKE_PROJECT_NAME} SHARED)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY C_STANDARD 17)

# Add all subdirectories that compose this library
add_subdirectory(include)
add_subdirectory(cuda)
add_subdirectory(matrix)
add_subdirectory(vector)
add_subdirectory(utils)

# Add external libraries
find_library(math m)

# Link external libraries to cla
target_link_libraries(${CMAKE_PROJECT_NAME} ${math} CUDA::cudart)

# Add test targets
add_executable(test_suite ../test/c/suite.c)
add_executable(memory_leak ../test/c/memory_leak.c)
add_executable(memory_stability ../test/c/memory_stability.c)

# Link library to executable
target_link_libraries(test_suite PRIVATE ${CMAKE_PROJECT_NAME})
target_link_libraries(memory_leak PRIVATE ${CMAKE_PROJECT_NAME})
target_link_libraries(memory_stability PRIVATE ${CMAKE_PROJECT_NAME})
