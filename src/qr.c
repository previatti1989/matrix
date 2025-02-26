#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "solvers.h"
#include "matrix_ops.h"

// Parallelized Householder QR Decomposition
void householder_qr(const FEMMatrix* A, FEMMatrix* Q, FEMMatrix* R) {
    if (A->rows != Q->rows || A->cols != Q->cols || A->cols != R->cols || A->cols != R->rows) return;

    int m = A->rows;
    int n = A->cols;

    // Copy A into R
    initialize_matrix(R, m, n);
#pragma omp parallel for
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            R->values[i * n + j] = A->values[i * n + j];

    // Initialize Q as identity matrix
    initialize_matrix(Q, m, m);
#pragma omp parallel for
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < m; j++)
            Q->values[i * m + j] = (i == j) ? 1.0 : 0.0;

    // Householder transformation
    FEMVector v;
    initialize_vector(&v, m);  // Householder vector

    for (size_t k = 0; k < n; k++) {
        double norm_x = 0.0;

        // Compute the norm of column k of R
#pragma omp parallel for reduction(+:norm_x)
        for (size_t i = k; i < m; i++)
            norm_x += R->values[i * n + k] * R->values[i * n + k];
        norm_x = sqrt(norm_x);

        double alpha = -sign(R->values[k * n + k]) * norm_x;

        // Compute Householder vector v
        v.values[k] = R->values[k * n + k] - alpha;
        for (size_t i = k + 1; i < m; i++)
            v.values[i] = R->values[i * n + k];

        // Compute squared norm of v
        double v_norm_squared = 0.0;
#pragma omp parallel for reduction(+:v_norm_squared)
        for (size_t i = k; i < m; i++)
            v_norm_squared += v.values[i] * v.values[i];

        double beta = 2.0 / v_norm_squared;

        // Apply Householder transformation to R
        for (size_t j = k; j < n; j++) {
            double dot = 0.0;
#pragma omp parallel for reduction(+:dot)
            for (size_t i = k; i < m; i++)
                dot += v.values[i] * R->values[i * n + j];

#pragma omp parallel for
            for (size_t i = k; i < m; i++)
                R->values[i * n + j] -= beta * dot * v.values[i];
        }

        // Apply Householder transformation to Q
        for (size_t j = 0; j < m; j++) {
            double dot = 0.0;
#pragma omp parallel for reduction(+:dot)
            for (size_t i = k; i < m; i++)
                dot += v.values[i] * Q->values[i * m + j];

#pragma omp parallel for
            for (size_t i = k; i < m; i++)
                Q->values[i * m + j] -= beta * dot * v.values[i];
        }
    }

    // Transpose Q to make it orthogonal
    FEMMatrix QT;
    initialize_matrix(&QT, m, m);
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i + 1; j < m; j++) {
            double temp = Q->values[i * m + j];
            Q->values[i * m + j] = Q->values[j * m + i];
            Q->values[j * m + i] = temp;
        }
    }

    free_vector(&v);
}

// Solves Ax = b using QR decomposition
void qr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x) {
    // Step 1: Allocate space for Q and R
    FEMMatrix Q, R;
    initialize_matrix(&Q, A->rows, A->cols);
    initialize_matrix(&R, A->cols, A->cols);

    // Step 2: Compute Householder QR decomposition (A = QR)
    householder_qr(A, &Q, &R);

    // Step 3: Compute y = Q^T * b
    FEMVector y;
    initialize_vector(&y, Q.rows);
    matvec_mult_transpose(&Q, b, &y);

    // Step 4: Solve R * x = y using Back Substitution
    for (int i = R.cols - 1; i >= 0; i--) {
        double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
        for (size_t j = i + 1; j < R.cols; j++) {
            sum += R.values[i * R.cols + j] * x->values[j];
        }
        x->values[i] = (y.values[i] - sum) / R.values[i * R.cols + i];
    }

    // Step 5: Free allocated memory
    free_matrix(&Q);
    free_matrix(&R);
    free_vector(&y);
}
