#include "cg.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Generate a well-conditioned SPD matrix A, a known solution x_expected, and compute b = A * x_expected
void generate_spd_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected, int n) {
    initialize_matrix(A, n, n);
    initialize_vector(b, n);
    initialize_vector(x_expected, n);

    // Generate a symmetric positive definite matrix A
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double value = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random values in [-1,1]
            A->values[i * n + j] = value;
            A->values[j * n + i] = value; // Ensure symmetry
        }
        A->values[i * n + i] += n; // Enforce diagonal dominance
    }

    // Generate a known solution x_expected in range [-1, 1]
    for (int i = 0; i < n; i++) {
        x_expected->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Compute b = A * x_expected
    matvec_mult(A, x_expected, b);
}

int main() {
    int n = 4;  // Size of the system (square SPD matrix)
    double tol = 1e-10;
    int max_iter = 5;

    srand(42); // Set seed for reproducibility

    FEMMatrix A;
    FEMVector b, x, expected_x, Ax;

    initialize_vector(&x, n);
    initialize_vector(&expected_x, n);
    initialize_vector(&Ax, n);

    // Generate a well-conditioned symmetric positive definite matrix A
    generate_spd_system(&A, &b, &expected_x, n);

    // Print matrix A and vector b
    print_matrix(&A, "Matrix A (SPD)");
    print_vector(&b, "Vector b (computed from Ax)");

    // Run Conjugate Gradient Solver
    set_vector_zero(&x);
    cg_solver(&A, &b, &x, tol, max_iter);

    // Compute Ax to check if Ax = b
    matvec_mult(&A, &x, &Ax);

    // Print computed solution
    print_vector(&x, "Computed x (CG solution)");

    // Print expected solution
    print_vector(&expected_x, "Expected x");

    // Compute residual ||Ax - b||
    double residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        residual_norm += (Ax.values[i] - b.values[i]) * (Ax.values[i] - b.values[i]);
    }
    residual_norm = sqrt(residual_norm);

    printf("CG Residual ||Ax - b|| = %f\n", residual_norm);

    // Compute expected residual (should be close to 0)
    double expected_residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        expected_residual_norm += (b.values[i] - b.values[i]) * (b.values[i] - b.values[i]);
    }
    expected_residual_norm = sqrt(expected_residual_norm);

    printf("Expected Residual ||Ax - b|| = %e (should be close to 0)\n", expected_residual_norm);

    // **Final Pass/Fail Check**
    if (residual_norm < tol) {
        printf(" CG test PASSED \n");
    }
    else {
        printf(" CG test FAILED\n");
    }

    // Cleanup
    free_vector(&b);
    free_vector(&x);
    free_vector(&Ax);
    free_matrix(&A);
    free_vector(&expected_x);

    return 0;
}