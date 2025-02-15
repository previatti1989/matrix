#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stddef.h>

// Matrix multiplication: C = A * B
void matrix_multiply(const double *A, const double *B, double *C, size_t rowsA, size_t colsA, size_t colsB);

// Matrix transpose: B = A^T
void matrix_transpose(const double *A, double *B, size_t rows, size_t cols);

// Gradient computation: grad = J^T * f
void compute_gradient(const double *J, const double *f, double *grad, size_t rows, size_t cols);

#endif // MATRIX_OPS_H
