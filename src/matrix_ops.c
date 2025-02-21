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
    if (!v || !v->values || v->size < 0) {
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
    if (A->cols != v->size || A->rows != result->size) {
        fprintf(stderr, "ERROR: Dimension mismatch in matvec_mult().\n");
        return;
    }
    if (!result->values) {
        fprintf(stderr, "ERROR: Output vector is not allocated!\n");
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




