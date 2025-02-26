#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_ops.h"  // Your existing matrix functions
#include "solvers.h"          // QR decomposition for solving least squares

// Generates a tridiagonal test FEMMatrix
void generate_test_matrix(FEMMatrix* A) {
    int n = A->rows;
    for (int i = 0; i < n; i++) {
        A->values[i * A->cols + i] = 2.0;  // Diagonal
        if (i > 0) A->values[i * A->cols + (i - 1)] = -1.0;  // Lower diagonal
        if (i < n - 1) A->values[i * A->cols + (i + 1)] = -1.0;  // Upper diagonal
    }
}

// Computes residual norm ||Ax - b||
double compute_residual(FEMMatrix* A, FEMVector* x, FEMVector* b) {
    FEMVector Ax;
    initialize_vector(&Ax, A->rows);
    matvec_mult(A, x, &Ax);

    // Compute residual Ax - b
    for (int i = 0; i < A->rows; i++) {
        Ax.values[i] -= b->values[i];
    }

    double res = vector_norm(&Ax);
    free_vector(&Ax);
    return res;
}

// Main test function
int main() {
    int n = 5;  // Matrix size
    int max_iter = 1000;  // GMRES iterations
    double tol = 1e-6;
    int k_max = 5;

    // Initialize FEM structures
    FEMMatrix A;
    initialize_matrix(&A, n, n);

    FEMVector b, x;
    initialize_vector(&b, n);
    initialize_vector(&x, n);  // Initial guess x0 = 0 (already zeroed by calloc)

    // Generate test system
    generate_test_matrix(&A);
    for (int i = 0; i < n; i++) b.values[i] = i + 1;

    // Print test system
    print_matrix(&A, "Matrix A");
    print_vector(&b, "Vector b");

    // Run GMRES
    printf("Running GMRES test...\n");
    gmres_solver(&A, &b, &x, tol, max_iter, k_max);

    // Compute and print residual
    double res = compute_residual(&A, &x, &b);
    printf("Residual norm: %e\n", res);

    // Print solution
    print_vector(&x, "Computed solution x");

    // Cleanup
    free_matrix(&A);
    free_vector(&b);
    free_vector(&x);

    return 0;
}