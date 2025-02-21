#include "lsqr.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function to create a simple test system Ax = b
void generate_test_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected, int size) {
    initialize_matrix(A, size, size);
    initialize_vector(b, size);
    initialize_vector(x_expected, size); // Store the expected solution

    srand(42); // Fixed seed for reproducibility

    // Generate a random expected solution x_expected in range [-1,1]
    for (int i = 0; i < size; i++) {
        x_expected->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Generate a well-conditioned SPD matrix A using A = M^T * M + diag_shift * I
    double* M = (double*)malloc(size * size * sizeof(double));  // Temporary matrix

    // Fill M with small random values to control conditioning
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            M[i * size + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random values in [-1,1]
        }
    }

    // Compute A = M^T * M to ensure positive semi-definiteness
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A->values[i * size + j] = 0.0;
            for (int k = 0; k < size; k++) {
                A->values[i * size + j] += M[k * size + i] * M[k * size + j];  // M^T * M
            }
        }
        A->values[i * size + i] += 5.0;  // Diagonal shift to ensure strict positive definiteness
    }

    free(M);

    // Compute expected b = A * x_expected
    matvec_mult(A, x_expected, b);
}

void generate_2x2_test_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected) {
    initialize_matrix(A, 2, 2);
    initialize_vector(b, 2);
    initialize_vector(x_expected, 2);

    // Define matrix A
    A->values[0] = 1.0;  A->values[1] = 2.0;
    A->values[2] = 3.0;  A->values[3] = 4.0;

    // Define expected solution x_expected
    x_expected->values[0] = 1.0;
    x_expected->values[1] = 2.0;

    // Compute b = A * x_expected
    matvec_mult(A, x_expected, b);
}

void generate_smooth_decay_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected) {
    // Initialize a well-conditioned 3x3 matrix
    initialize_matrix(A, 3, 3);
    initialize_vector(b, 3);
    initialize_vector(x_expected, 3);

    // Define A (smooth singular value decay)
    A->values[0] = 1.0;  A->values[1] = 0.5;  A->values[2] = 0.2;
    A->values[3] = 0.5;  A->values[4] = 1.0;  A->values[5] = 0.3;
    A->values[6] = 0.2;  A->values[7] = 0.3;  A->values[8] = 1.0;

    // Define expected solution
    x_expected->values[0] = 1.0;
    x_expected->values[1] = 2.0;
    x_expected->values[2] = 3.0;

    // Compute b = A * x_expected
    matvec_mult(A, x_expected, b);
}

int main() {
    int size = 3; // Small test case
    double tol = 1e-10;
    int max_iter = 100;

    FEMMatrix A; 
    FEMVector b, x, expected_x, Ax;

    initialize_vector(&x, size);
    initialize_vector(&expected_x, size);
    initialize_vector(&Ax, size);

    generate_smooth_decay_system(&A, &b, &expected_x);

    // Now Call LSQR Solver
    printf("\n=== Running LSQR Solver ===\n");
    set_vector_zero(&x);

    lsqr_solver(&A, &b, &x, tol, max_iter);

    matvec_mult(&A, &x, &Ax);

    // Print computed x
    print_vector(&x, "Computed x");

    // Print expected x
    print_vector(&expected_x, "Expected x");

    // Compute new residual ||Ax - b||
    double residual_norm = 0.0;
    for (int i = 0; i < size; i++) {
        residual_norm += (Ax.values[i] - b.values[i]) * (Ax.values[i] - b.values[i]);
    }
    residual_norm = sqrt(residual_norm);

    printf("New LSQR Residual ||Ax - b|| = %f\n", residual_norm);

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

    return 0;
}