#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stddef.h>

// Structure for a sparse or dense matrix
typedef struct {
    size_t rows;
    size_t cols;
    double* values; // Stores matrix data in row-major order
} FEMMatrix;

// Structure for a vector
typedef struct {
    size_t size;
    double* values;
} FEMVector;

void matrix_multiply(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C);
void matrix_transpose(const FEMMatrix* A, FEMMatrix* B);
void compute_gradient(const FEMMatrix* J, const FEMVector* f, FEMVector* grad);
void initialize_matrix(FEMMatrix* A, size_t rows, size_t cols);
void free_matrix(FEMMatrix* A);
void initialize_vector(FEMVector* v, size_t size);
double vector_norm(const FEMVector* v);
void set_vector_zero(FEMVector* v);
void copy_vector(FEMVector* dest, const FEMVector* src);
void vector_scale(FEMVector* v, double scalar);
void free_vector(FEMVector* v);
void matvec_mult(const FEMMatrix* A, const FEMVector* v, FEMVector* result);
void matvec_mult_transpose(const FEMMatrix* A, const FEMVector* v, FEMVector* result);
void matrix_multiply_transpose(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C);
void print_matrix(const FEMMatrix* M, const char* name);
void print_vector(const FEMVector* v, const char* name);

#endif // MATRIX_OPS_H
