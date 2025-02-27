#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "matrix_ops.h"
#include "solvers.h"

void cg_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter) {
    int n = A->rows;
    
    FEMMatrix_CSR A_csr;
    convert_to_csr(A, &A_csr);

    // initialize vectors
    FEMVector r, p, Ap;

    initialize_vector(&r, n);
    initialize_vector(&p, n);
    initialize_vector(&Ap, n);

    csr_matvec_mult(&A_csr, x, &r);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r.values[i] = b->values[i] - r.values[i];
    }

    copy_vector(&p, &r);
    double rs_old = dot_product(&r, &r);
    double rs_new;

    for (int k = 0; k < max_iter; k++) {
        csr_matvec_mult(&A_csr, &p, &Ap); // Compute Ap_k

        double alpha = rs_old / dot_product(&p, &Ap);

        // Update x and r
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x->values[i] += alpha * p.values[i];
            r.values[i] -= alpha * Ap.values[i];
        }

        rs_new = dot_product(&r, &r);

        // Check for convergence
        if (sqrt(rs_new) < tol) {
            printf("Converged in %d iterations\n", k + 1);
            break;
        }

        double beta = rs_new / rs_old;

        // Update search direction p
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p.values[i] = r.values[i] + beta * p.values[i];
        }

        rs_old = rs_new;
    }

}

