# CMake policy
cmake_minimum_required(VERSION 3.20)

# Setting project name, version and language
project(cla
        VERSION 0.1
        LANGUAGES CUDA C)

# Add library cla
add_library(${CMAKE_PROJECT_NAME} SHARED)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY C_STANDARD 17)

# Add all subdirectories that compose this library
add_subdirectory(cuda)
add_subdirectory(matrix)
add_subdirectory(vector)
add_subdirectory(include)

# Add external libraries
find_library(math m)

# Link external libraries to cla
target_link_libraries(${CMAKE_PROJECT_NAME} ${math})

# Add test_suite target
add_executable(test_suite ../test/c/suite.c)

# Link library to executable
target_link_libraries(test_suite PRIVATE ${CMAKE_PROJECT_NAME})
