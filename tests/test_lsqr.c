#include "solvers.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void generate_well_conditioned_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected, int m, int n) {
    initialize_matrix(A, m, n);
    initialize_vector(b, m);
    initialize_vector(x_expected, n);

    // Generate a well-conditioned rectangular matrix A (diagonal dominance)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double value = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random values in [-1, 1]
            A->values[i * n + j] = value;
        }
    }

    // Enforce diagonal dominance for stability (helps LSQR convergence)
    int min_dim = (m < n) ? m : n;  // Get min(m, n)
    for (int i = 0; i < min_dim; i++) {
        A->values[i * n + i] += n; // Strengthen diagonal elements
    }

    for (int i = 0; i < m; i++) {
        double row_norm = 0.0;
        for (int j = 0; j < n; j++) {
            row_norm += A->values[i * n + j] * A->values[i * n + j];
        }
        row_norm = sqrt(row_norm);
        if (row_norm > 1e-12) {  // Avoid division by zero
            for (int j = 0; j < n; j++) {
                A->values[i * n + j] /= row_norm;
            }
        }
    }

    // Generate a random expected solution x_expected in range [-1, 1]
    for (int i = 0; i < n; i++) {
        x_expected->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Compute b = A * x_expected
    matvec_mult(A, x_expected, b);
}

int main() {
    int m = 4; // Number of rows (overdetermined system)
    int n = 3; // Number of columns (unknowns)
    double tol = 1e-10;
    int max_iter = 5;

    srand(42);

    FEMMatrix A;
    FEMVector b, x, expected_x, Ax;

    initialize_vector(&x, n);
    initialize_vector(&expected_x, n);
    initialize_vector(&Ax, m);

    // Generate a rectangular system where A is (4x2)
    generate_well_conditioned_system(&A, &b, &expected_x, m, n);

    print_matrix(&A, "matrix A");
    print_vector(&b, "vector b computed from Ax");

    // Run LSQR Solver
    set_vector_zero(&x);
    lsqr_solver(&A, &b, &x, tol, max_iter);

    // Compute Ax to check if Ax = b
    matvec_mult(&A, &x, &Ax);

    // Print computed x
    print_vector(&x, "Computed x (LSQR solution)");

    // Print expected x
    print_vector(&expected_x, "Expected x");

    // Compute residual ||Ax - b||
    double residual_norm = 0.0;
    for (int i = 0; i < m; i++) {
        residual_norm += (Ax.values[i] - b.values[i]) * (Ax.values[i] - b.values[i]);
    }
    residual_norm = sqrt(residual_norm);

    printf("LSQR Residual ||Ax - b|| = %f\n", residual_norm);

    double expected_residual_norm = 0.0;
    for (int i = 0; i < m; i++) {
        expected_residual_norm += (b.values[i] - b.values[i]) * (b.values[i] - b.values[i]);  // Expected ||Ax - b|| = 0
    }
    expected_residual_norm = sqrt(expected_residual_norm);

    printf("Expected Residual ||Ax - b|| = %e (should be close to 0)\n", expected_residual_norm);

    // **Final Pass/Fail Check**
    if (residual_norm < tol) {
        printf(" LSQR test PASSED \n");
    }
    else {
        printf(" LSQR test FAILED\n");
    }

    // Cleanup
    free_vector(&b);
    free_vector(&x);
    free_vector(&Ax);
    free_matrix(&A);
    free_vector(&expected_x);

    return 0;
}