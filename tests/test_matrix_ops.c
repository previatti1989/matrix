#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-6

// Function to check if two matrices are approximately equal
int matrices_are_equal(const double* A, const double* B, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; i++) {
        if (fabs(A[i] - B[i]) > EPSILON) {
            return 0;
        }
    }
    return 1;
}

void test_matrix_multiply() {
    double A[6] = { 1, 2, 3, 4, 5, 6 };
    double B[6] = { 7, 8, 9, 10, 11, 12 };
    double C[4];
    double expected[4] = { 58, 64, 139, 154 };

    matrix_multiply(A, B, C, 2, 3, 2);

    if (matrices_are_equal(C, expected, 2, 2)) {
        printf("Matrix multiplication test passed.\n");
    }
    else {
        printf("Matrix multiplication test failed.\n");
    }
}

void test_matrix_transpose() {
    double A[6] = { 1, 2, 3, 4, 5, 6 };
    double B[6];
    double expected[6] = { 1, 4, 2, 5, 3, 6 };

    matrix_transpose(A, B, 2, 3);

    if (matrices_are_equal(B, expected, 3, 2)) {
        printf("Matrix transpose test passed.\n");
    }
    else {
        printf("Matrix transpose test failed.\n");
    }
}

void test_compute_gradient() {
    double J[6] = { 1, 2, 3, 4, 5, 6 };
    double f[2] = { 1, 2 };
    double grad[3];
    double expected[3] = { 9, 12, 15 };

    compute_gradient(J, f, grad, 2, 3);

    if (matrices_are_equal(grad, expected, 1, 3)) {
        printf("Gradient computation test passed.\n");
    }
    else {
        printf("Gradient computation test failed.\n");
    }
}

int main() {
    test_matrix_multiply();
    test_matrix_transpose();
    test_compute_gradient();
    return 0;
}