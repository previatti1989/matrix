#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stddef.h>

// sparse matrix
typedef struct {
    int rows;           // Number of rows
    int cols;           // Number of columns
    int nnz;            // Number of nonzero elements
    double* values;     // Array of nonzero values
    int* col_idx;       // Column indices for each value
    int* row_ptr;       // Row pointer array (size: rows + 1)
} FEMMatrix_CSR;

// dense matrix
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

void initialize_matrix(FEMMatrix* A, size_t rows, size_t cols);
void initialize_vector(FEMVector* v, int size);
void free_matrix(FEMMatrix* A);
void free_vector(FEMVector* v);
void set_vector_zero(FEMVector* v);
void copy_vector(FEMVector* dest, const FEMVector* src);
void print_matrix(const FEMMatrix* M, const char* name);
void print_vector(const FEMVector* v, const char* name);
double vector_norm(const FEMVector* v);
void vector_scale(FEMVector* v, double scalar);
double dot_product(const FEMVector* a, const FEMVector* b);

void matrix_multiply(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C);
void matrix_transpose(const FEMMatrix* A, FEMMatrix* B);
void compute_gradient(const FEMMatrix* J, const FEMVector* f, FEMVector* grad);

void matvec_mult(const FEMMatrix* A, const FEMVector* v, FEMVector* result);
void matvec_mult_transpose(const FEMMatrix* A, const FEMVector* v, FEMVector* result);
void matrix_multiply_transpose(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C);

void convert_to_csr(const FEMMatrix* A, FEMMatrix_CSR* A_csr);
void csr_matvec_mult_transpose(const FEMMatrix_CSR* A, const FEMVector* x, FEMVector* result);
void csr_matvec_mult(const FEMMatrix_CSR* A, const FEMVector* x, FEMVector* result);


#endif // MATRIX_OPS_H
