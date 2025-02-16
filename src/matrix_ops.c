#include "matrix_ops.h"
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void matrix_multiply(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C) {
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return;

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
    if (A->rows != B->cols || A->cols != B->rows) return;

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            B->values[j * A->rows + i] = A->values[i * A->cols + j];
        }
    }
}

void compute_gradient(const FEMMatrix* J, const FEMVector* f, FEMVector* grad) {
    if (J->rows != f->size || J->cols != grad->size) return;

#pragma omp parallel for
    for (size_t j = 0; j < J->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < J->rows; i++) {
            sum += J->values[i * J->cols + j] * f->values[i];
        }
        grad->values[j] = sum;
    }
}

void initialize_vector(FEMVector* v, size_t size) {
    v->size = size;
    v->values = (double*)calloc(size, sizeof(double));
}

double vector_norm(const FEMVector* v) {
    double sum = 0.0;
    for (size_t i = 0; i < v->size; i++) {
        sum += v->values[i] * v->values[i];
    }
    return sqrt(sum);
}

void set_vector_zero(FEMVector* v) {
    memset(v->values, 0, v->size * sizeof(double));
}

void copy_vector(FEMVector* dest, const FEMVector* src) {
    if (dest->size != src->size) {
        free(dest->values); // Free old memory
        dest->values = (double*)malloc(src->size * sizeof(double));
        dest->size = src->size;
    }
    memcpy(dest->values, src->values, src->size * sizeof(double));
}

void vector_scale(FEMVector* v, double scalar) {
    for (size_t i = 0; i < v->size; i++) {
        v->values[i] *= scalar;
    }
}

void free_vector(FEMVector* v) {
    free(v->values);
    v->values = NULL;
}

void matvec_mult(const FEMMatrix* A, const FEMVector* v, FEMVector* result) {
    if (A->cols != v->size || A->rows != result->size) return;

#pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; j++) {
            sum += A->values[i * A->cols + j] * v->values[j];
        }
        result->values[i] = sum;
    }
}

void matvec_mult_transpose(const FEMMatrix* A, const FEMVector* v, FEMVector* result) {
    if (A->rows != v->size || A->cols != result->size) return;

#pragma omp parallel for
    for (size_t j = 0; j < A->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < A->rows; i++) {
            sum += A->values[i * A->cols + j] * v->values[i];
        }
        result->values[j] = sum;
    }
}

void initialize_matrix(FEMMatrix* A, size_t rows, size_t cols) {
    A->rows = rows;
    A->cols = cols;
    A->values = (double*)calloc(rows * cols, sizeof(double));
}

void free_matrix(FEMMatrix* A) {
    free(A->values);
    A->values = NULL;
}

void matrix_multiply_transpose(const FEMMatrix* A, const FEMMatrix* B, FEMMatrix* C) {
    if (A->cols != B->cols || A->rows != C->rows || B->rows != C->cols) return;

#pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->rows; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->values[k * A->cols + i] * B->values[k * B->cols + j];
            }
            C->values[i * C->cols + j] = sum;
        }
    }
}

// Function to print a matrix
void print_matrix(const FEMMatrix* M, const char* name) {
    printf("\n%s:\n", name);
    for (size_t i = 0; i < M->rows; i++) {
        for (size_t j = 0; j < M->cols; j++) {
            printf("%8.3f ", M->values[i * M->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print a vector
void print_vector(const FEMVector* v, const char* name) {
    printf("\n%s:\n", name);
    for (size_t i = 0; i < v->size; i++) {
        printf("%8.3f\n", v->values[i]);
    }
    printf("\n");
}