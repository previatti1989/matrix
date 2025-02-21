
#include "lsqr.h"
#include "matrix_ops.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void lsqr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter) {
    int n = A->rows;

    // initialize vectors
    FEMVector v, u, v_old, u_old, w, x_old;
    initialize_vector(&v, n);
    initialize_vector(&u, n);
    initialize_vector(&w, n);
    initialize_vector(&x_old, n);
    initialize_vector(&v_old, n);
    initialize_vector(&u_old, n);

    double beta, alpha, rho, rho_old, c, s, theta, phi;

    // initialization ////
    // u = b/beta ////
    // compute initial norm of b
    double b_norm = vector_norm(b);
    if (b_norm < tol) {
        printf("DEBUG: b is zero! Exiting LSQR early.\n");
        set_vector_zero(x);
        return;
    }
    copy_vector(&u, b);
    beta = vector_norm(&u);
    if (beta < tol) {
        printf("DEBUG: Beta too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&u, 1.0 / beta);
    print_vector(&u, "first u");
    ////
    // v = ATu/alpha ////
    // v = ATu
    matvec_mult_transpose(A, &u, &v);
    alpha = vector_norm(&v);
    if (alpha < tol) {
        printf("DEBUG: Alpha too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&v, 1.0 / alpha);
    print_vector(&v, "first v");
    ////
    // initialize variables
    copy_vector(&w, &v);
    set_vector_zero(x);
    phi = beta;
    theta = 0;
    rho_old = alpha;
    ////

    for (int iter = 0; iter < max_iter; iter++) {
        printf("\n=== Iteration %d ===\n", iter);
        fflush(stdout); // Ensure immediate printing       

        // bidiagonalization ////
        copy_vector(&u_old, &u);
        copy_vector(&v_old, &v);
        // u = Av-alpha u/beta ////
        // compute new u = Av
        matvec_mult(A, &v, &u);
        // compute u_k+1 = Av_k - alpha_k u_k
        for (int i = 0; i < n; i++) {
            u.values[i] -= alpha* u_old.values[i]; // Prevents u from going to zero
        }
        // Compute new beta
        beta = vector_norm(&u);
        if (beta < tol) {
            printf("DEBUG: beta too small, Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            break;
        }
        vector_scale(&u, 1.0 / beta);
        print_vector(&u, "Normalized u");
        ////
        // v = ATu-beta v/alpha //
        // compute new v = ATu
        matvec_mult_transpose(A, &u, &v);
        for (int i = 0; i < n; i++) {
            v.values[i]  -= beta * v_old.values[i];
        }
        alpha = vector_norm(&v);
        if (alpha < tol) {
            printf("DEBUG: alpha too small, Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            break;
        }
        vector_scale(&v, 1.0 / alpha);
        print_vector(&v, "Normalized v");
        ////
        
        // implicit QR factorization ////
        rho = sqrt(fmax(tol, rho_old * rho_old + beta * beta));
        if (rho <= tol || isnan(rho)) {
            printf("DEBUG: rho became zero or NaN! Stopping LSQR.\n");
            break;
        }
        c = rho_old / rho;
        s = beta / rho;
        theta = s * alpha;
        rho_old = rho;
        ////
        // 
        // update x //
        double factor = phi / rho;
        for (int i = 0; i < n; i++) {
            x->values[i] += factor * w.values[i];
        }
        print_vector(x, "updated x");
        ////
        // update w //
        for (int i = 0; i < n; i++) {
            w.values[i] = v.values[i] - theta * w.values[i];  
        }
        print_vector(&w, "Updated w");
        ////

        phi = c * phi;

        if (iter % 1 == 0) {
            FEMVector Ax;
            initialize_vector(&Ax, b->size);
            matvec_mult(A, x, &Ax);

            double residual_norm = 0.0;
            for (size_t i = 0; i < b->size; i++) {
                double diff = Ax.values[i] - b->values[i];
                residual_norm += diff * diff;
            }
            residual_norm = sqrt(residual_norm);
            free_vector(&Ax);
            printf("Iter %d: ||Ax - b|| = %f, factor = %f, alpha = %f, beta = %f, rho = %f, c = %f, s = %f, phi = %f, theta = %f\n",
                iter, residual_norm, factor, alpha, beta, rho, c, s, phi, theta);
            print_vector(x, "Current x");
        }

        if (iter > 0 && phi < tol * b_norm) {
            printf("LSQR Converged: ||Ax - b|| = %f < tol * ||b|| = %f\n", iter, phi, tol * b_norm);
            break;
        }

    }

    // free memory
    free_vector(&u);
    free_vector(&u_old);
    free_vector(&v);
    free_vector(&v_old);
    free_vector(&w);
    free_vector(&x_old);
}
