#include "lsqr.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to create a simple test system Ax = b
void generate_test_system(FEMMatrix* A, FEMVector* b, FEMVector* x_expected, int size) {
    initialize_matrix(A, size, size);
    initialize_vector(b, size);
    initialize_vector(x_expected, size); // Store the expected solution

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A->values[i * size + j] = (i == j) ? 4.0 : 1.0;
        }
        x_expected->values[i] = 1.0;  // Expected solution x = [1, 1, ..., 1]
    }

    // Compute expected b = A * x_expected
    matvec_mult(A, x_expected, b);
}

int main() {
    int size = 5; // Small test case
    double tol = 1e-6;
    int max_iter = 1000;

    FEMMatrix A;
    FEMVector b, x, expected_x, Ax;
    generate_test_system(&A, &b, &expected_x, size);

    //print_matrix(&A, "Matrix A");
    //print_vector(&b, "Vector b computed from A x_expected");

    initialize_vector(&x, size);
    set_vector_zero(&x);

    initialize_vector(&Ax, size);

    printf("Testing LSQR solver...\n");
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
    free_matrix(&A);
    free_vector(&b);
    free_vector(&x);
    free_vector(&Ax);

    return 0;
}