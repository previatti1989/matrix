#include "qr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-6  // Floating-point tolerance

// Function to print matrices
void print_matrix(double** M, int rows, int cols, const char* name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%8.4f ", M[i][j]);
        printf("\n");
    }
}

// Function to check if Q^T * Q is close to identity
int check_orthogonality(double** Q, int m) {
    double** QTQ = allocate_matrix(m, m);

    // Compute Q^T * Q
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            QTQ[i][j] = 0.0;
            for (int k = 0; k < m; k++)
                QTQ[i][j] += Q[k][i] * Q[k][j];
        }
    }

    // Check if QTQ is approximately the identity matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(QTQ[i][j] - expected) > TOLERANCE) {
                free_matrix(QTQ, m);
                return 0;  // Not orthogonal
            }
        }
    }
    free_matrix(QTQ, m);
    return 1;  // Q is orthogonal
}

// Function to check if A = QR
int check_reconstruction(double** A, double** Q, double** R, int m, int n) {
    double** QR = allocate_matrix(m, n);

    // Compute QR
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            QR[i][j] = 0.0;
            for (int k = 0; k < m; k++)
                QR[i][j] += Q[i][k] * R[k][j];
        }
    }

    // Compare QR with original A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - QR[i][j]) > TOLERANCE) {
                free_matrix(QR, m);
                return 0;  // Decomposition is incorrect
            }
        }
    }
    free_matrix(QR, m);
    return 1;  // Decomposition is correct
}

// Test QR Decomposition
void test_qr_decomposition() {
    int m = 3, n = 3;
    double** A = allocate_matrix(m, n);
    double** Q = allocate_matrix(m, m);
    double** R = allocate_matrix(m, n);
    double** QTQ = allocate_matrix(m, m);

    // Expected Q (Precomputed from NumPy)
    double expected_Q[3][3] = {
        {-0.5345,  0.4182,  0.7330},
        {-0.2673, -0.9050,  0.3304},
        {-0.8018, -0.0861, -0.5916}
    };

    double A_values[3][3] = {
        {2, -1, 3},
        {1, 4, -2},
        {3, 1, 2}
    };

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = A_values[i][j];

    householder_qr(A, Q, R, m, n);

    // Compute QTQ = Q^T * Q
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            QTQ[i][j] = 0.0;
            for (int k = 0; k < m; k++)
                QTQ[i][j] += Q[k][i] * Q[k][j];  // Compute Q^T * Q
        }
    }

    //  Print Computed Q
    print_matrix(Q, m, m, "Computed Q");

    // Print Expected Q (from NumPy)
    printf("\nExpected Q (Precomputed from NumPy):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            printf("%8.4f ", expected_Q[i][j]);
        }
        printf("\n");
    }

    if (check_orthogonality(Q, m))
        printf(" Q is orthogonal \n");
    else
        printf(" Q is NOT orthogonal \n");

    if (check_reconstruction(A, Q, R, m, n))
        printf(" QR decomposition is correct \n");
    else
        printf(" QR decomposition is incorrect \n");

    print_matrix(QTQ, m, m, "Computed Q^T * Q");

    free_matrix(A, m);
    free_matrix(Q, m);
    free_matrix(R, m);
}

// Main test execution
int main() {
    printf("\n Running QR Decomposition Tests...\n");
    test_qr_decomposition();
    return 0;
}
