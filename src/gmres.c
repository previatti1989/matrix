#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "matrix_ops.h"  // Your existing matrix operations (dot product, norm, etc.)
#include "qr.h"          // Your QR decomposition solver
#include "gmres.h"

// GMRES solver
void gmres_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, int k_max) {
    int n = A->rows;

    FEMVector* V[k_max + 1];
    FEMMatrix H;
    FEMVector g, y;

    initialize_matrix(&H, k_max, k_max + 1);
    initialize_vector(&g, k_max + 1);
    initialize_vector(&y, k_max);

    for (int i = 0; i <= k_max; i++) {
        V[i] = malloc(sizeof(FEMVector));
        initialize_vector(V[i], n);
    }

    // Compute initial residual r0 = b - Ax
    FEMVector r;
    initialize_vector(&r, n);
    matvec_mult(A, x, &r);
    for (int i = 0; i < n; i++) {
        r.values[i] = b->values[i] - r.values[i];
    }

    double beta = vector_norm(&r);
    if (beta < TOLERANCE) {
        printf("Initial residual is below tolerance. Exiting.\n");
        goto cleanup;
    }

    // Normalize V[0] = r / ||r||
    vector_scale(&r, 1.0 / beta);
    copy_vector(V[0], &r);
    g.values[0] = beta;

    int k;
    for (k = 0; k < k_max; k++) {
        FEMVector w;
        initialize_vector(&w, n);

        // Arnoldi iteration: w = A * V[k]
        matvec_mult(A, V[k], &w);

        // Gram-Schmidt orthogonalization
        for (int j = 0; j <= k; j++) {
            H.values[j * H.cols + k] = dot_product(V[j], &w);
            for (int i = 0; i < n; i++) {
                w.values[i] -= H.values[j * H.cols + k] * V[j]->values[i];
            }
        }

        H.values[(k + 1) * H.cols + k] = vector_norm(&w);
        if (H.values[(k + 1) * H.cols + k] < TOLERANCE) {
            free_vector(&w);
            break;
        }

        // Normalize next basis vector
        vector_scale(&w, 1.0 / H.values[(k + 1) * H.cols + k]);
        copy_vector(V[k + 1], &w);

        free_vector(&w);
    }

    // Solve least squares problem Hy = g using QR decomposition
    qr_solver(&H, &g, &y);

    // Compute final solution x = x + V * y
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            x->values[i] += V[j]->values[i] * y.values[j];
        }
    }

cleanup:
    for (int i = 0; i <= k_max; i++) {
        free_vector(V[i]);
        free(V[i]);
    }
    free_matrix(&H);
    free_vector(&g);
    free_vector(&y);
    free_vector(&r);
}

