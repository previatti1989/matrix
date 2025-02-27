#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "matrix_ops.h"  // Your existing matrix operations (dot product, norm, etc.)
#include "solvers.h"

// Arnoldi Process: Generates orthonormal Krylov subspace and Hessenberg matrix
void arnoldi_iteration(const FEMMatrix_CSR* A, FEMVector** V, FEMMatrix* H, int k) {
    int n = A->rows;
    FEMVector w;
    initialize_vector(&w, n);

    // Compute w = A * V[k]
    csr_matvec_mult(A, V[k], &w);

    // Gram-Schmidt orthogonalization
#pragma omp parallel for
    for (int j = 0; j <= k; j++) {
        H->values[j * H->cols + k] = dot_product(V[j], &w);
        for (int i = 0; i < n; i++) {
            w.values[i] -= H->values[j * H->cols + k] * V[j]->values[i];
        }
    }

    // Compute next Krylov basis vector
    H->values[(k + 1) * H->cols + k] = vector_norm(&w);
    if (H->values[(k + 1) * H->cols + k] > 1e-10) {  // Avoid division by zero
        vector_scale(&w, 1.0 / H->values[(k + 1) * H->cols + k]);
        copy_vector(V[k + 1], &w);
    }

    free_vector(&w);
}

// GMRES solver with Arnoldi and QR
void gmres_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter, int k_max) {
    int n = A->rows;
    FEMMatrix_CSR A_csr;
    convert_to_csr(A, &A_csr);
    
    FEMVector r;
    initialize_vector(&r, n);

    // Compute initial residual r0 = b - Ax
    csr_matvec_mult(&A_csr, x, &r);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r.values[i] = b->values[i] - r.values[i];
    }

    double beta = vector_norm(&r);
    if (beta < tol) {
        printf("Initial residual is below tolerance. Exiting.\n");
        free_vector(&r);
        free(A_csr.values);
        free(A_csr.col_idx);
        free(A_csr.row_ptr);
        return;
    }

    // Allocate Krylov basis vectors
    FEMVector* V[k_max + 1];
    for (int i = 0; i <= k_max; i++) {
        V[i] = malloc(sizeof(FEMVector));
        initialize_vector(V[i], n);
    }

    // Declare Hessenberg matrix and auxiliary vectors
    FEMMatrix H;
    FEMVector g, y;

    int iter = 0;
    while (iter < max_iter) {  // Outer loop for restarts
        // Free & reallocate Hessenberg matrix and vectors
        if (iter > 0) {  // Only free if they have been allocated before
            free_matrix(&H);
            free_vector(&g);
            free_vector(&y);
        }

        initialize_matrix(&H, k_max, k_max + 1);
        initialize_vector(&g, k_max + 1);
        initialize_vector(&y, k_max);

        // Normalize V[0] = r / ||r||
        vector_scale(&r, 1.0 / beta);
        copy_vector(V[0], &r);
        g.values[0] = beta;

        int k;
        for (k = 0; k < k_max && iter < max_iter; k++, iter++) {  // Arnoldi process (inner loop)
            arnoldi_iteration(&A_csr, V, &H, k);
        }

        // Solve the least squares problem Hy = g using QR
        // Create a reduced Hessenberg matrix (H_reduced) of size k x k
        FEMMatrix H_reduced;
        initialize_matrix(&H_reduced, k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                H_reduced.values[i * k + j] = H.values[i * H.cols + j];  // Copy only k x k block
            }
        }
        FEMVector g_reduced;
        initialize_vector(&g_reduced, k);
        for (int i = 0; i < k; i++) {
            g_reduced.values[i] = g.values[i];  // Copy only first k elements
        }
        qr_solver(&H_reduced, &g_reduced, &y);

        // Update solution: x = x + V * y
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                x->values[i] += V[j]->values[i] * y.values[j];
            }
        }

        // Compute new residual
        csr_matvec_mult(&A_csr, x, &r);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            r.values[i] = b->values[i] - r.values[i];
        }
        beta = vector_norm(&r);

        printf("Iteration %d, Residual: %e\n", iter, beta);
        if (beta < tol) {
            printf("GMRES converged after %d iterations.\n", iter);
            break;
        }
    }

    // Final cleanup
    for (int i = 0; i <= k_max; i++) {
        free_vector(V[i]);
        free(V[i]);
    }
    free_matrix(&H);
    free_vector(&g);
    free_vector(&y);
    free_vector(&r);

    free(A_csr.values);
    free(A_csr.col_idx);
    free(A_csr.row_ptr);
}