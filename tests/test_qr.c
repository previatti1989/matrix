#include "qr.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-6  // Floating-point tolerance

// Function to check if QT Q is close to identity
int check_orthogonality(const FEMMatrix* Q) {
    if (Q->rows != Q->cols) return 0;

    FEMMatrix QTQ;
    initialize_matrix(&QTQ, Q->rows, Q->cols);
    
    matrix_multiply_transpose(Q, Q, &QTQ);  // Compute QT Q
    print_matrix(&QTQ, "Computed Q^T * Q (Should be Identity)");
    double max_diff = 0.0;
    for (size_t i = 0; i < Q->rows; i++) {
        for (size_t j = 0; j < Q->cols; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = fabs(QTQ.values[i * Q->cols + j] - expected);
            if (diff > max_diff) max_diff = diff;
        }
    }
    free_matrix(&QTQ);

    printf("Maximum deviation from identity: %e\n", max_diff);
    return (max_diff < TOLERANCE);
}

// Function to check if A = QR
int check_reconstruction(const FEMMatrix* A, const FEMMatrix* Q, const FEMMatrix* R) {
    if (A->rows != Q->rows || A->cols != R->cols) return 0;

    FEMMatrix QR;
    initialize_matrix(&QR, A->rows, A->cols);

    matrix_multiply(Q, R, &QR);  // Compute QR
    print_matrix(&QR, "Computed QR (Should match A)");
    double max_diff = 0.0;
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            double diff = fabs(A->values[i * A->cols + j] - QR.values[i * A->cols + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    free_matrix(&QR);

    printf("Maximum deviation from A: %e\n", max_diff);
    return (max_diff < TOLERANCE);
}

// Test QR Decomposition
void test_qr_decomposition() {
    size_t m = 3, n = 3;

    FEMMatrix A, Q, R;
    initialize_matrix(&A, m, n);
    initialize_matrix(&Q, m, m);
    initialize_matrix(&R, m, n);

    double A_values[3][3] = {
        {2, -1, 3},
        {1, 4, -2},
        {3, 1, 2}
    };

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            A.values[i * n + j] = A_values[i][j];

    householder_qr(&A, &Q, &R);

    print_matrix(&Q, "Computed Q");

    double expected_Q[3][3] = {
        {-0.5345,  0.4182,  0.7330},
        {-0.2673, -0.9050,  0.3304},
        {-0.8018, -0.0861, -0.5916}
    };

    printf("\nExpected Q (Precomputed from NumPy):\n");
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%8.4f ", expected_Q[i][j]);
        }
        printf("\n");
    }

    if (check_orthogonality(&Q))
        printf("Q is orthogonal.\n");
    else
        printf("Q is NOT orthogonal.\n");

    if (check_reconstruction(&A, &Q, &R))
        printf("QR decomposition is correct.\n");
    else
        printf("QR decomposition is incorrect.\n");

    free_matrix(&A);
    free_matrix(&Q);
    free_matrix(&R);
}

// Test QR Solver
void test_qr_solver() {
    printf("\nTesting QR Solver...\n");

    size_t m = 3, n = 3;

    FEMMatrix A, Q, R;
    initialize_matrix(&A, m, n);
    initialize_matrix(&Q, m, m);
    initialize_matrix(&R, m, n);

    FEMVector b, x;
    initialize_vector(&b, m);
    initialize_vector(&x, n);

    // Example system Ax = b
    double A_values[3][3] = {
        {2, -1, 3},
        {1, 4, -2},
        {3, 1, 2}
    };
    double b_values[3] = { 5, 3, 7 };

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            A.values[i * n + j] = A_values[i][j];

    for (size_t i = 0; i < m; i++)
        b.values[i] = b_values[i];

    // Perform QR decomposition
    householder_qr(&A, &Q, &R);

    // Solve Ax = b using QR
    qr_solve(&Q, &R, &b, &x);

    print_vector(&x, "Computed Solution x");

    // Check if Ax = b
    FEMVector Ax;
    initialize_vector(&Ax, m);
    matvec_mult(&A, &x, &Ax);

    print_vector(&Ax, "Computed Ax");
    print_vector(&b, "Expected b");

    double max_diff = 0.0;
    for (size_t i = 0; i < m; i++) {
        double diff = fabs(Ax.values[i] - b.values[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Maximum deviation from expected b: %e\n", max_diff);

    if (max_diff < TOLERANCE)
        printf("QR Solver test PASSED\n");
    else
        printf("QR Solver test FAILED\n");

    free_matrix(&A);
    free_matrix(&Q);
    free_matrix(&R);
    free_vector(&b);
    free_vector(&x);
    free_vector(&Ax);
}

// Main test execution
int main() {
    printf("\nRunning QR Decomposition Tests...\n");
    test_qr_decomposition();
    test_qr_solver();
    return 0;
}
