#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "qr.h"

// Function to return the sign of a number
double sign(double x) {
    return (x >= 0) ? 1.0 : -1.0;
}

// Function to allocate a 2D matrix dynamically
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matrix[i] = (double*)malloc(cols * sizeof(double));
    return matrix;
}

// Function to free a dynamically allocated matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

// Parallelized Householder QR Decomposition
void householder_qr(double** A, double** Q, double** R, int m, int n) {
    double* v = (double*)malloc(m * sizeof(double));

    // Initialize R as a copy of A
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A[i][j];

    // Initialize Q as an identity matrix
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q[i][j] = (i == j) ? 1.0 : 0.0;

    // Householder transformation
    for (int k = 0; k < n; k++) {
        double norm_x = 0.0;

        // Compute the norm of column k of R
#pragma omp parallel for reduction(+:norm_x)
        for (int i = k; i < m; i++)
            norm_x += R[i][k] * R[i][k];
        norm_x = sqrt(norm_x);

        double alpha = -sign(R[k][k]) * norm_x;
        double beta; // = 1.0 / (norm_x * (norm_x + fabs(R[k][k]))); // Better stability

        // Compute Householder vector v
        v[k] = R[k][k] - alpha;
        for (int i = k + 1; i < m; i++)
            v[i] = R[i][k];

        // Compute squared norm of v
        double v_norm_squared = 0.0;
        for (int i = k; i < m; i++)
            v_norm_squared += v[i] * v[i];

        beta = 2.0 / v_norm_squared;  // Correct scaling factor

        // Apply Householder transformation to R
        for (int j = k; j < n; j++) {
            double dot = 0.0;
#pragma omp parallel for reduction(+:dot)
            for (int i = k; i < m; i++)
                dot += v[i] * R[i][j];

#pragma omp parallel for
            for (int i = k; i < m; i++)
                R[i][j] -= beta * dot * v[i];
        }

        // Apply Householder transformation to Q
        for (int j = 0; j < m; j++) {
            double dot = 0.0;
#pragma omp parallel for reduction(+:dot)
            for (int i = k; i < m; i++)
                dot += v[i] * Q[i][j];

#pragma omp parallel for
            for (int i = k; i < m; i++)
                Q[i][j] -= beta * dot * v[i];
        }
    }

    // Transpose Q to make it orthogonal
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            double temp = Q[i][j];
            Q[i][j] = Q[j][i];
            Q[j][i] = temp;
        }
    }

    free(v);
}

// Solves Ax = b using QR decomposition
void qr_solve(double** Q, double** R, double* b, double* x, int rows, int cols) {
    double* y = (double*)malloc(rows * sizeof(double));

    // Compute y = Q^T * b
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < rows; j++) {
            sum += Q[j][i] * b[j];  // Fixed indexing
        }
        y[i] = sum;
    }

    // Back substitution to solve R * x = y
    for (int i = cols - 1; i >= 0; i--) {
        double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < cols; j++) {
            sum += R[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / R[i][i];
    }

    free(y);
}
