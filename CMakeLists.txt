cmake_minimum_required(VERSION 3.10)
project(matrix_module C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

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

# Include directories
include_directories(${INCLUDE_DIR})

# Create the main library
add_library(matrix_module ${SOURCES})

# Ensure OpenMP is used for GCC
if(OpenMP_C_FOUND)
    target_link_libraries(matrix_module OpenMP::OpenMP_C)
endif()

# Create test executables
add_executable(test_matrix_ops ${TEST_DIR}/test_matrix_ops.c)
add_executable(test_qr ${TEST_DIR}/test_qr.c)
add_executable(test_lsqr ${TEST_DIR}/test_lsqr.c)
add_executable(test_cg ${TEST_DIR}/test_cg.c)

# Link test executables with the library
target_link_libraries(test_matrix_ops matrix_module OpenMP::OpenMP_C)
target_link_libraries(test_qr matrix_module OpenMP::OpenMP_C)
target_link_libraries(test_lsqr matrix_module OpenMP::OpenMP_C)
target_link_libraries(test_cg matrix_module OpenMP::OpenMP_C)

# Enable testing
enable_testing()
add_test(NAME test_matrix_ops COMMAND test_matrix_ops)
add_test(NAME test_qr COMMAND test_qr)
add_test(NAME test_lsqr COMMAND test_lsqr)
add_test(NAME test_cg COMMAND test_cg)
