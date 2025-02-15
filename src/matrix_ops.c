#include "matrix_ops.h"
#include <stdlib.h>
#include <omp.h>

// Matrix multiplication: C = A * B
void matrix_multiply(const double *A, const double *B, double *C, size_t rowsA, size_t colsA, size_t colsB) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// Matrix transpose: B = A^T
void matrix_transpose(const double *A, double *B, size_t rows, size_t cols) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

// Gradient computation: grad = J^T * f
void compute_gradient(const double *J, const double *f, double *grad, size_t rows, size_t cols) {
    #pragma omp parallel for
    for (size_t j = 0; j < cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; i++) {
            sum += J[i * cols + j] * f[i];
        }
        grad[j] = sum;
    }
}