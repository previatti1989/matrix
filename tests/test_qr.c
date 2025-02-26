#include "solvers.h"
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

void test_qr_decomposition_case1() {
    size_t m = 5, n = 5;

    FEMMatrix A, Q, R;
    initialize_matrix(&A, m, n);
    initialize_matrix(&Q, m, m);
    initialize_matrix(&R, m, n);

    double A_values[5][5] = {
        {4, -2, 2, 1, 3},
        {1,  3, -1, 4, 2},
        {2,  1,  3, -2, 5},
        {3,  4,  1, 3, -1},
        {-1, 2,  4, 1, 3}
    };

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            A.values[i * n + j] = A_values[i][j];

    householder_qr(&A, &Q, &R);

    print_matrix(&Q, "Computed Q (Case 1)");

    double expected_Q[5][5] = {
        {-0.6901,  0.2277,  0.3992,  0.4675, -0.3228},
        {-0.1725, -0.8993, -0.1183,  0.3568, -0.1621},
        {-0.3450, -0.1915,  0.7804, -0.2607, -0.3867},
        {-0.5175, -0.0481, -0.4336, -0.6902,  0.2674},
        { 0.1725, -0.3240, -0.1174,  0.3557,  0.8721}
    };

    printf("\nExpected Q (Precomputed from NumPy - Case 1):\n");
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%8.4f ", expected_Q[i][j]);
        }
        printf("\n");
    }

    if (check_orthogonality(&Q))
        printf("Q is orthogonal (Case 1).\n");
    else
        printf("Q is NOT orthogonal (Case 1).\n");

    if (check_reconstruction(&A, &Q, &R))
        printf("QR decomposition is correct (Case 1).\n");
    else
        printf("QR decomposition is incorrect (Case 1).\n");

    free_matrix(&A);
    free_matrix(&Q);
    free_matrix(&R);
}

void test_qr_decomposition_case2() {
    size_t m = 3, n = 3;

    FEMMatrix A, Q, R;
    initialize_matrix(&A, m, n);
    initialize_matrix(&Q, m, m);
    initialize_matrix(&R, m, n);

    double A_values[3][3] = {
        {5, 3, 2},
        {1, 4, 7},
        {2, 1, 3}
    };

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            A.values[i * n + j] = A_values[i][j];

    householder_qr(&A, &Q, &R);

    print_matrix(&Q, "Computed Q (Case 2)");

    double expected_Q[3][3] = {
        {-0.8729,  0.4544,  0.1755},
        {-0.1746, -0.5436,  0.8208},
        {-0.3359, -0.7048, -0.6242}
    };

    printf("\nExpected Q (Precomputed from NumPy - Case 2):\n");
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%8.4f ", expected_Q[i][j]);
        }
        printf("\n");
    }

    if (check_orthogonality(&Q))
        printf("Q is orthogonal (Case 2).\n");
    else
        printf("Q is NOT orthogonal (Case 2).\n");

    if (check_reconstruction(&A, &Q, &R))
        printf("QR decomposition is correct (Case 2).\n");
    else
        printf("QR decomposition is incorrect (Case 2).\n");

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
    //householder_qr(&A, &Q, &R);

    // Solve Ax = b using QR
    qr_solver(&A, &b, &x);

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

void test_qr_solver_case3() {
    printf("\nTesting QR Solver - Case 3 (Hilbert Matrix)...\n");

    size_t m = 5, n = 5;

    FEMMatrix A, Q, R;
    initialize_matrix(&A, m, n);
    initialize_matrix(&Q, m, m);
    initialize_matrix(&R, m, n);

    FEMVector b, x;
    initialize_vector(&b, m);
    initialize_vector(&x, n);

    // Hilbert matrix A
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            A.values[i * n + j] = 1.0 / (i + j + 1);
        }
    }

    // Right-hand side b
    double b_values[5] = { 1, 2, 3, 4, 5 };
    for (size_t i = 0; i < m; i++)
        b.values[i] = b_values[i];

    // Perform QR decomposition
    //householder_qr(&A, &Q, &R);

    // Solve Ax = b using QR
    qr_solver(&A, &b, &x);

    print_vector(&x, "Computed Solution x (Case 3)");

    // Check if Ax = b
    FEMVector Ax;
    initialize_vector(&Ax, m);
    matvec_mult(&A, &x, &Ax);

    print_vector(&Ax, "Computed Ax (Case 3)");
    print_vector(&b, "Expected b (Case 3)");

    double max_diff = 0.0;
    for (size_t i = 0; i < m; i++) {
        double diff = fabs(Ax.values[i] - b.values[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Maximum deviation from expected b (Case 3): %e\n", max_diff);

    if (max_diff < TOLERANCE)
        printf("QR Solver test PASSED (Case 3)\n");
    else
        printf("QR Solver test FAILED (Case 3)\n");

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
    test_qr_decomposition_case1();
    test_qr_decomposition_case2();

    printf("\nRunning QR Solver Tests...\n");
    test_qr_solver();
    test_qr_solver_case3();

    return 0;
}
