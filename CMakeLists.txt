cmake_minimum_required(VERSION 3.10)
project(matrix_module C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

if (NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "Wrong compiler detected! Expected GCC, but got ${CMAKE_C_COMPILER_ID}. Run CMake with: cmake -G Ninja -DCMAKE_C_COMPILER=gcc")
endif()

# Enable OpenMP (for GCC)
find_package(OpenMP REQUIRED)

# Define source and include directories
set(SRC_DIR src)
set(INCLUDE_DIR include)
set(TEST_DIR tests)

# Define source files
set(SOURCES
    ${SRC_DIR}/matrix_ops.c
    ${SRC_DIR}/qr.c
    ${SRC_DIR}/lsqr.c
    ${SRC_DIR}/cg.c
)

# Define test source files
set(TEST_SOURCES
    ${TEST_DIR}/test_matrix_ops.c
    ${TEST_DIR}/test_qr.c
    ${TEST_DIR}/test_lsqr.c
    ${TEST_DIR}/test_cg.c
)

# Create the main library
add_library(matrix_module ${SOURCES})

# Include directories for the library
target_include_directories(matrix_module PUBLIC ${INCLUDE_DIR})

# Ensure OpenMP is used for GCC
if(OpenMP_C_FOUND)
    target_link_libraries(matrix_module OpenMP::OpenMP_C)
endif()

# Create test executables and link them with matrix_module
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE matrix_module)
    if(OpenMP_C_FOUND)
        target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_C)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Enable testing
enable_testing()