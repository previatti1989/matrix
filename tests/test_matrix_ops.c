#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-6

void compare_matrices(const FEMMatrix* A, const FEMMatrix* B, const char* test_name) {
    int equal = 1;
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        if (fabs(A->values[i] - B->values[i]) > EPSILON) {
            equal = 0;
            break;
        }
    }
    if (equal) {
        printf("[PASS] %s\n", test_name);
    }
    else {
        printf("[FAIL] %s\n", test_name);
    }
}

void compare_vectors(const FEMVector* A, const FEMVector* B, const char* test_name) {
    int equal = 1;
    for (size_t i = 0; i < A->size; i++) {
        if (fabs(A->values[i] - B->values[i]) > EPSILON) {
            equal = 0;
            break;
        }
    }
    if (equal) {
        printf("[PASS] %s\n", test_name);
    }
    else {
        printf("[FAIL] %s\n", test_name);
    }
}

void test_matrix_multiply() {
    printf("\nTesting matrix_multiply...\n");

    FEMMatrix A = { 2, 3, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix C;
    initialize_matrix(&C, A.rows, B.cols);

    matrix_multiply(&A, &B, &C);
    print_matrix(&C, "Computed Result of AB");

    FEMMatrix expected = { 2, 2, (double[]) { 22, 28, 49, 64 } };
    print_matrix(&expected, "Expected Result of AB");

    compare_matrices(&C, &expected, "test_matrix_multiply");

    free_matrix(&C);
}

// Test matrix transpose
void test_matrix_transpose() {
    printf("\nTesting matrix_transpose...\n");

    FEMMatrix A = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B;
    initialize_matrix(&B, A.cols, A.rows);

    matrix_transpose(&A, &B);
    print_matrix(&B, "Computed Transpose A");

    FEMMatrix expected = { 2, 3, (double[]) { 1, 3, 5, 2, 4, 6 } };
    print_matrix(&expected, "Expected Transpose A");

    compare_matrices(&B, &expected, "test_matrix_transpose");

    free_matrix(&B);
}

// Test compute gradient (J^T * f)
void test_compute_gradient() {
    printf("\nTesting compute_gradient...\n");

    FEMMatrix J = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMVector f = { 3, (double[]) { 1, 1, 1 } };
    FEMVector grad;
    initialize_vector(&grad, J.cols);

    compute_gradient(&J, &f, &grad);
    print_vector(&grad, "Computed Gradient J^T * f");

    FEMVector expected = { 2, (double[]) { 9, 12 } };
    print_vector(&expected, "Expected Gradient J^T * f");

    compare_vectors(&grad, &expected, "test_compute_gradient");

    free_vector(&grad);
}

// Test vector norm
void test_vector_norm() {
    printf("\nTesting vector_norm...\n");

    FEMVector v = { 3, (double[]) { 1, 2, 2 } };
    double computed = vector_norm(&v);
    double expected = 3.0;

    printf("Computed Norm: %.3f\n", computed);
    printf("Expected Norm: %.3f\n", expected);

    if (fabs(computed - expected) < EPSILON)
        printf("[PASS] test_vector_norm\n");
    else
        printf("[FAIL] test_vector_norm\n");
}

// Test vector scaling
void test_vector_scale() {
    printf("\nTesting vector_scale...\n");

    FEMVector v;
    initialize_vector(&v, 3);
    v.values[0] = 1.0;
    v.values[1] = 2.0;
    v.values[2] = 3.0;

    vector_scale(&v, 2.0);
    print_vector(&v, "Computed Scaled Vector");

    FEMVector expected = { 3, (double[]) { 2, 4, 6 } };
    print_vector(&expected, "Expected Scaled Vector");

    compare_vectors(&v, &expected, "test_vector_scale");

    free_vector(&v);
}

// Test matrix-vector multiplication
void test_matvec_mult() {
    printf("\nTesting matvec_mult...\n");

    FEMMatrix A = { 3, 3, (double[]) { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
    FEMVector v = { 3, (double[]) { 1, 1, 1 } };
    FEMVector result;
    initialize_vector(&result, A.rows);

    matvec_mult(&A, &v, &result);
    print_vector(&result, "Computed A * v");

    FEMVector expected = { 3, (double[]) { 6, 15, 24 } };
    print_vector(&expected, "Expected A * v");

    compare_vectors(&result, &expected, "test_matvec_mult");

    free_vector(&result);
}


// Test matrix multiply transpose
void test_matrix_multiply_transpose() {
    printf("\nTesting matrix_multiply_transpose...\n");

    FEMMatrix A = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B = { 4, 2, (double[]) { 7, 8, 9, 10, 11, 12, 13, 14 } };
    FEMMatrix BT;
    initialize_matrix(&BT, B.cols, B.rows);
    matrix_transpose(&B, &BT);

    FEMMatrix C;
    initialize_matrix(&C, A.rows, BT.cols);
    matrix_multiply_transpose(&A, &B, &C);
    print_matrix(&C, "Computed A * B^T");

    FEMMatrix expected = {
        3, 4, (double[]) {
            23, 29, 35, 41,
            53, 67, 81, 95,
            83, 105, 127, 149
        }
    };
    print_matrix(&expected, "Expected A * A^T");

    compare_matrices(&C, &expected, "test_matrix_multiply_transpose");

    free_matrix(&BT);
    free_matrix(&C);
}

void test_matvec_mult_transpose() {
    printf("\nTesting matvec_mult_transpose...\n");

    FEMMatrix A = { 3, 3, (double[]) { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
    FEMVector v = { 3, (double[]) { 1, 1, 1 } };
    FEMVector result;
    initialize_vector(&result, A.cols);

    matvec_mult_transpose(&A, &v, &result);
    print_vector(&result, "Computed A^T * v");

    FEMVector expected = { 3, (double[]) { 12, 15, 18 } };
    print_vector(&expected, "Expected A^T * v");

    compare_vectors(&result, &expected, "test_matvec_mult_transpose");

    free_vector(&result);
}

// Run all test cases
int main() {
    test_matrix_multiply();
    test_matrix_transpose();
    test_compute_gradient();
    test_vector_norm();
    test_vector_scale();
    test_matvec_mult();
    test_matvec_mult_transpose();
    test_matrix_multiply_transpose();

    printf("\nAll tests completed.\n");
    return 0;
}