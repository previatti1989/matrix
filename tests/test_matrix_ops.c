#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-6

// Test matrix initialization and free
void test_initialize_free_matrix() {
    printf("\nTesting matrix initialization and free...\n");
    FEMMatrix A;
    initialize_matrix(&A, 3, 3);

    for (size_t i = 0; i < A.rows; i++)
        for (size_t j = 0; j < A.cols; j++)
            A.values[i * A.cols + j] = i + j + 1.0;

    print_matrix(&A, "Initialized Matrix A");

    free_matrix(&A);
    printf("Matrix freed successfully.\n");
}

// Test vector initialization and free
void test_initialize_free_vector() {
    printf("\nTesting vector initialization and free...\n");
    FEMVector v;
    initialize_vector(&v, 3);

    for (size_t i = 0; i < v.size; i++)
        v.values[i] = i + 1.0;

    print_vector(&v, "Initialized Vector v");

    free_vector(&v);
    printf("Vector freed successfully.\n");
}

// Test matrix multiplication
void test_matrix_multiply() {
    printf("Testing matrix multiplication...\n");

    FEMMatrix A = { 2, 3, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.values = (double*)calloc(C.rows * C.cols, sizeof(double));

    matrix_multiply(&A, &B, &C);
    print_matrix(&C, "Result of AB");

    free(C.values);
}

// Test matrix transpose
void test_matrix_transpose() {
    printf("Testing matrix transpose...\n");

    FEMMatrix A = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B;
    B.rows = A.cols;
    B.cols = A.rows;
    B.values = (double*)calloc(B.rows * B.cols, sizeof(double));

    matrix_transpose(&A, &B);
    print_matrix(&B, "Transpose A");

    free(B.values);
}

// Test compute gradient (J^T * f)
void test_compute_gradient() {
    printf("Testing compute gradient (J^T * f)...\n");

    FEMMatrix J = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMVector f = { 3, (double[]) { 1, 1, 1 } };
    FEMVector grad;
    grad.size = J.cols;
    grad.values = (double*)calloc(grad.size, sizeof(double));

    compute_gradient(&J, &f, &grad);
    print_vector(&grad, "computed gradient");

    free(grad.values);
}

// Test copy vector
void test_copy_vector() {
    printf("\nTesting copy vector...\n");

    FEMVector v;
    initialize_vector(&v, 3);
    v.values[0] = 1.0;
    v.values[1] = 2.0;
    v.values[2] = 3.0;

    FEMVector v_copy;
    initialize_vector(&v_copy, 3);
    copy_vector(&v_copy, &v);

    print_vector(&v, "Original Vector v");
    print_vector(&v_copy, "Copied Vector v_copy");

    free_vector(&v);
    free_vector(&v_copy);
}

// Test vector operations
void test_vector_operations() {
    printf("Testing vector operations...\n");

    FEMVector v;
    initialize_vector(&v, 3);
    v.values[0] = 1.0;
    v.values[1] = 2.0;
    v.values[2] = 3.0;

    printf("Original vector:\n");
    print_vector(&v, "original vector");

    double norm = vector_norm(&v);
    printf("Vector norm: %.3f\n\n", norm);

    vector_scale(&v, 2.0);
    printf("Vector after scaling by 2:\n");
    print_vector(&v, "scale by 2");

    set_vector_zero(&v);
    printf("Vector after setting to zero:\n");
    print_vector(&v, "set zero");

    free_vector(&v);
}

// Test matrix-vector multiplication
void test_matvec_mult() {
    printf("Testing matrix-vector multiplication...\n");

    FEMMatrix A = { 3, 3, (double[]) { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
    FEMVector v = { 3, (double[]) { 1, 2, 3 } };
    FEMVector result;
    result.size = A.rows;
    result.values = (double*)calloc(result.size, sizeof(double));

    matvec_mult(&A, &v, &result);
    print_vector(&result, "matrix vector multiplication");

    free(result.values);
}

// Test matrix-vector transpose multiplication (A^T * v)
void test_matvec_mult_transpose() {
    printf("Testing matrix-vector transpose multiplication...\n");

    FEMMatrix A = { 3, 3, (double[]) { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
    FEMVector v = { 3, (double[]) { 1, 1, 1 } };
    FEMVector result;
    result.size = A.cols;
    result.values = (double*)calloc(result.size, sizeof(double));

    matvec_mult_transpose(&A, &v, &result);
    print_vector(&result, "matrix vector transpose multiplication result");

    free(result.values);
}

// Test matrix multiply transpose
void test_matrix_multiply_transpose() {
    printf("\nTesting matrix multiply transpose...\n");

    FEMMatrix A = { 3, 2, (double[]) { 1, 2, 3, 4, 5, 6 } };
    FEMMatrix B;
    initialize_matrix(&B, A.cols, A.rows);
    matrix_transpose(&A, &B);

    FEMMatrix C;
    initialize_matrix(&C, A.cols, A.cols);
    matrix_multiply_transpose(&A, &B, &C);

    print_matrix(&C, "Result of A * A^T");

    free_matrix(&B);
    free_matrix(&C);
}

// Main function to run all tests
int main() {
    test_initialize_free_matrix();
    test_initialize_free_vector();
    test_matrix_multiply();
    test_matrix_transpose();
    test_compute_gradient();
    test_copy_vector();
    test_vector_operations();
    test_matvec_mult();
    test_matvec_mult_transpose();
    test_matrix_multiply_transpose();

    printf("\nAll tests completed successfully.\n");
    return 0;
}