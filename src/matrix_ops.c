#include "matrix_ops.h"
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void initialize_vector(FEMVector* v, int size) {
    if (size <= 0) {
        fprintf(stderr, "ERROR: Invalid vector size %d in initialize_vector().\n", size);
        v->values = NULL;
        return;
    }
    v->size = size;
    v->values = (double*)calloc(size, sizeof(double));
    if (!v->values) {
        fprintf(stderr, "ERROR: Memory allocation failed in initialize_vector().\n");
    }
}

void initialize_matrix(FEMMatrix* A, size_t rows, size_t cols) {
    if (!A) {
        fprintf(stderr, "ERROR: Null pointer passed to initialize_matrix().\n");
        return;
    }

    A->rows = rows;
    A->cols = cols;
    A->values = (double*)calloc(rows * cols, sizeof(double));

    if (!A->values) {
        fprintf(stderr, "ERROR: Memory allocation failed in initialize_matrix().\n");
    }
}

void free_vector(FEMVector* v) {
    if (!v) return;
    free(v->values);
    v->values = NULL;
}

void free_matrix(FEMMatrix* A) {
    if (!A || !A->values) {
        return;
    }

    free(A->values);
    A->values = NULL;
}

void copy_vector(FEMVector* dest, const FEMVector* src) {
    if (!src || !src->values) {
        fprintf(stderr, "ERROR: Source vector is NULL in copy_vector().\n");
        return;
    }

    if (dest->size != src->size) {
        free(dest->values); // Free old memory
        dest->values = (double*)malloc(src->size * sizeof(double));
        if (!dest->values) {
            fprintf(stderr, "ERROR: Memory allocation failed in copy_vector().\n");
            return;
        }
        dest->size = src->size;
    }
    memcpy(dest->values, src->values, src->size * sizeof(double));
}

void set_vector_zero(FEMVector* v) {
    if (!v || !v->values) {
        fprintf(stderr, "ERROR: Null pointer in set_vector_zero().\n");
        return;
    }
    memset(v->values, 0, v->size * sizeof(double));
}

void print_matrix(const FEMMatrix* M, const char* name) {
    if (!M || !M->values) {
        fprintf(stderr, "ERROR: Null pointer passed to print_matrix().\n");
        return;
    }
    if (!name) {
        fprintf(stderr, "ERROR: Null pointer passed as name to print_vector().\n");
        return;
    }
    if (M->rows == 0 || M->cols == 0) {
        printf("\n%s: (empty matrix)\n", name);
        return;
    }

    printf("\n%s:\n", name);
    for (size_t i = 0; i < M->rows; i++) {
        for (size_t j = 0; j < M->cols; j++) {
            printf("%8.3f ", M->values[i * M->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(const FEMVector* v, const char* name) {
    if (!v || !v->values) {
        fprintf(stderr, "ERROR: Null pointer passed to print_vector().\n");
        return;
    }
    if (!name) {
        fprintf(stderr, "ERROR: Null pointer passed as name to print_vector().\n");
        return;
    }
    if (v->size == 0) {
        printf("\n%s: (empty vector)\n", name);
        return;
    }


    printf("\n%s:\n", name);
    for (size_t i = 0; i < v->size; i++) {
        printf("%8.3f\n", v->values[i]);
    }
    printf("\n");
}

double vector_norm(const FEMVector* v) {
    if (!v || !v->values || v->size == 0) {
        fprintf(stderr, "ERROR: Invalid vector in vector_norm().\n");
        return 0.0;
    }

    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v->size; i++) {
        sum += v->values[i] * v->values[i];
    }
    return sqrt(sum);
}

void vector_scale(FEMVector* v, double scalar) {
    if (!v || !v->values) {
        fprintf(stderr, "ERROR: Null vector in vector_scale().\n");
        return;
    }

#pragma omp parallel for
    for (size_t i = 0; i < v->size; i++) {
        v->values[i] *= scalar;
    }
}

double dot_product(const FEMVector* a, const FEMVector* b) {
    if (a->size != b->size) {
        printf("ERROR: dot_product called with mismatched vector sizes!\n");
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < a->size; i++) {
        sum += a->values[i] * b->values[i];
    }
    return sum;
}

void matrix_multiply(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C) {
    if (!A || !B || !C) {
        fprintf(stderr, "ERROR: Null pointer passed to matrix_multiply().\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "ERROR: Dimension mismatch in matrix_multiply().\n");
        return;
    }
    if (!C->values) {
        fprintf(stderr, "ERROR: Output matrix C is not allocated!\n");
        return;
    }

    memset(C->values, 0, C->rows * C->cols * sizeof(double));

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->values[i * A->cols + k] * B->values[k * B->cols + j];
            }
            C->values[i * C->cols + j] = sum;
        }
    }
}

void matrix_transpose(const FEMMatrix* A, FEMMatrix* B) {
    if (!A || !B) {
        fprintf(stderr, "ERROR: Null pointer passed to matrix_transpose().\n");
        return;
    }
    if (A->rows != B->cols || A->cols != B->rows) {
        fprintf(stderr, "ERROR: Dimension mismatch in matrix_transpose().\n");
        return;
    }
    if (!B->values) {
        fprintf(stderr, "ERROR: Output matrix B is not allocated!\n");
        return;
    }

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            B->values[j * A->rows + i] = A->values[i * A->cols + j];
        }
    }
}

void compute_gradient(const FEMMatrix* J, const FEMVector* f, FEMVector* grad) {
    if (J->rows != f->size || J->cols != grad->size) {
        fprintf(stderr, "ERROR: Dimension mismatch in compute_gradient().\n");
        return;
    }
    if (!grad->values) {
        fprintf(stderr, "ERROR: Output gradient vector is not allocated!\n");
        return;
    }

#pragma omp parallel for
    for (size_t j = 0; j < J->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < J->rows; i++) {
            sum += J->values[i * J->cols + j] * f->values[i];
        }
        grad->values[j] = sum;
    }
}

void matvec_mult(const FEMMatrix* A, const FEMVector* v, FEMVector* result) {
    if (!A || !v || !result) {
        fprintf(stderr, "ERROR: Null pointer passed to matvec_mult().\n");
        return;
    }
    if (A->cols != v->size || A->rows != result->size) {
        fprintf(stderr, "ERROR: Dimension mismatch in matvec_mult().\n");
        return;
    }
    if (!result->values) {
        fprintf(stderr, "ERROR: Output vector is not allocated!\n");
        return;
    }
    if (!A->values || !v->values) {
        fprintf(stderr, "ERROR: Input matrix or vector is not allocated!\n");
        return;
    }

#pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; j++) {
            double product = A->values[i * A->cols + j] * v->values[j];
            sum += product;
        }
        result->values[i] = sum;
    }
}

void matvec_mult_transpose(const FEMMatrix* A, const FEMVector* v, FEMVector* result) {
    if (!A || !v || !result) {
        fprintf(stderr, "ERROR: Null pointer passed to matvec_mult_transpose().\n");
        return;
    }
    if (A->rows != v->size || A->cols != result->size) {
        fprintf(stderr, "ERROR: Dimension mismatch in matvec_mult_transpose().\n");
        return;
    }
    if (!result->values) {
        printf("ERROR: matvec_mult_transpose - result vector is not allocated!\n");
        return;
    }
    if (!A->values || !v->values) {
        fprintf(stderr, "ERROR: Input matrix or vector is not allocated!\n");
        return;
    }

#pragma omp parallel for
    for (size_t j = 0; j < A->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < A->rows; i++) {
            sum += A->values[i * A->cols + j] * v->values[i];
        }
        result->values[j] = sum;
    }

}

void matrix_multiply_transpose(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C) {
    if (!A || !B || !C) {
        fprintf(stderr, "ERROR: Null pointer passed to matrix_multiply_transpose().\n");
        return;
    }

    if (A->cols != B->cols || A->rows != C->rows || B->rows != C->cols) {
        fprintf(stderr, "ERROR: Dimension mismatch! Expected C(%zu x %zu) but got (%zu x %zu)\n",
            A->rows, B->rows, C->rows, C->cols);
        return;
    }

    if (!C->values) {
        fprintf(stderr, "ERROR: matrix_multiply_transpose - output matrix C is not allocated!\n");
        return;
    }

    memset(C->values, 0, C->rows * C->cols * sizeof(double));

#pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->rows; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->values[i * A->cols + k] * B->values[j * B->cols + k];
            }
            C->values[i * C->cols + j] = sum;
        }
    }
}

void convert_to_csr(const FEMMatrix* A, FEMMatrix_CSR* A_csr) {
    int nnz = 0;

    // Count nonzero elements
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (fabs(A->values[i * A->cols + j]) > 1e-12) {
                nnz++;
            }
        }
    }

    // Allocate CSR structure
    A_csr->rows = A->rows;
    A_csr->cols = A->cols;
    A_csr->nnz = nnz;
    A_csr->values = (double*)malloc(nnz * sizeof(double));
    A_csr->col_idx = (int*)malloc(nnz * sizeof(int));
    A_csr->row_ptr = (int*)malloc((A->rows + 1) * sizeof(int));

    int index = 0;
    A_csr->row_ptr[0] = 0;

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            double val = A->values[i * A->cols + j];
            if (fabs(val) > 1e-12) {
                A_csr->values[index] = val;
                A_csr->col_idx[index] = j;
                index++;
            }
        }
        A_csr->row_ptr[i + 1] = index;
    }
}

void csr_matvec_mult_transpose(const FEMMatrix_CSR* A, const FEMVector* x, FEMVector* result) {
    //  Check for NULL pointers
    if (!A || !x || !result) {
        printf("[ERROR] csr_matvec_mult_transpose: NULL pointer detected.\n");
        return;
    }

    //  Check for dimension mismatch
    if (A->rows != x->size) {
        printf("[ERROR] csr_matvec_mult_transpose: Dimension mismatch. Matrix rows = %d, Vector size = %d\n", A->rows, x->size);
        return;
    }
    if (A->cols != result->size) {
        printf("[ERROR] csr_matvec_mult_transpose: Output vector has incorrect size. Expected %d, got %d\n", A->cols, result->size);
        return;
    }

    //  Ensure nonzero elements are valid
    if (A->nnz <= 0) {
        printf("[WARNING] csr_matvec_mult_transpose: No nonzero elements in the matrix.\n");
        return;
    }

    // Initialize result vector to zero
    for (int i = 0; i < A->cols; i++) {
        result->values[i] = 0.0;
    }

    int invalid_flag = 0;
    //  Perform sparse transpose multiplication
    for (int i = 0; i < A->rows; i++) {
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            int col = A->col_idx[j];
            if (col >= A->cols || col < 0) {
                printf("[ERROR] csr_matvec_mult_transpose: Invalid column index %d at row %d\n", col, i);
                invalid_flag = 1;
            }
            result->values[col] += A->values[j] * x->values[i];
        }
    }
    if (invalid_flag) {
        printf("[ERROR] csr_matvec_mult: Invalid column index detected.\n");
        return;
    }
}

void csr_matvec_mult(const FEMMatrix_CSR* A, const FEMVector* x, FEMVector* result) {
    //  Check for NULL pointers
    if (!A || !x || !result) {
        printf("[ERROR] csr_matvec_mult: NULL pointer detected.\n");
        return;
    }

    //  Check for dimension mismatch
    if (A->cols != x->size) {
        printf("[ERROR] csr_matvec_mult: Dimension mismatch. Matrix cols = %d, Vector size = %d\n", A->cols, x->size);
        return;
    }
    if (A->rows != result->size) {
        printf("[ERROR] csr_matvec_mult: Output vector has incorrect size. Expected %d, got %d\n", A->rows, result->size);
        return;
    }

    //  Ensure nonzero elements are valid
    if (A->nnz <= 0) {
        printf("[WARNING] csr_matvec_mult: No nonzero elements in the matrix.\n");
        return;
    }

    int invalid_flag = 0;
    for (int i = 0; i < A->rows; i++) {
        result->values[i] = 0.0; // Reset the output vector

        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            int col = A->col_idx[j];
            if (col >= A->cols || col < 0) {
                printf("[ERROR] csr_matvec_mult: Invalid column index %d at row %d\n", col, i);
                invalid_flag = 1;
            }
            result->values[i] += A->values[j] * x->values[col];
        }
    }
    if (invalid_flag) {
        printf("[ERROR] csr_matvec_mult: Invalid column index detected.\n");
        return;
    }
}

