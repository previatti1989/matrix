#include "lsqr.h"
#include "matrix_ops.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void lsqr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter) {
    int m = A->rows;
    int n = A->cols;

    // initialize vectors
    FEMVector v, u, v_old, u_old, w;
    initialize_vector(&v, n);
    initialize_vector(&u, m);
    initialize_vector(&w, n);
    initialize_vector(&v_old, n);
    initialize_vector(&u_old, m);

    double alpha, beta, theta, c, s, rho, phi, barrho, barphi;
    int exit_flag = 0;

    // initialization 
    double b_norm = vector_norm(b);
    if (b_norm < tol) {
        printf("DEBUG: b is zero! Exiting LSQR early.\n");
        set_vector_zero(x);
        return;
    }
    // u = b/beta
    copy_vector(&u, b);
    beta = vector_norm(&u);
    if (beta < tol) {
        printf("DEBUG: Beta too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&u, 1.0 / beta);
    print_vector(&u, "first u");

    // v = ATu/alpha 
    matvec_mult_transpose(A, &u, &v);
    alpha = vector_norm(&v);
    if (alpha < tol) {
        printf("DEBUG: Alpha too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&v, 1.0 / alpha);
    print_vector(&v, "first v");

    // initialize variables
    copy_vector(&w, &v);
    set_vector_zero(x);
    barrho = alpha;
    barphi = beta;
    //theta = 0.0;
    printf("alpha = %f, beta = %f\n", alpha, beta);

    /*double factor = beta / alpha;
    for (int i = 0; i < n; i++) {
        x->values[i] += factor * w.values[i];
    }
    print_vector(x, "updated x");*/

    for (int iter = 0; iter < max_iter; iter++) {
        printf("\n=== Iteration %d ===\n", iter);
        fflush(stdout); // Ensure immediate printing       

        // bidiagonalization

        // u = Av-alpha*u_old
        copy_vector(&u_old, &u);
        matvec_mult(A, &v, &u);
#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            u.values[i] -= alpha* u_old.values[i]; // Prevents u from going to zero
        }
        beta = vector_norm(&u);
        printf("beta = %f\n", beta);
        if (beta < tol) {
            printf("DEBUG: beta too small, Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            exit_flag = 1;
        }
        else {
            vector_scale(&u, 1.0 / beta);
            print_vector(&u, "normalized u");
        }
        
        // v = ATu-beta*v_old
        copy_vector(&v_old, &v);
        matvec_mult_transpose(A, &u, &v);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            v.values[i]  -= beta * v_old.values[i];
        }
        alpha = vector_norm(&v);
        printf("alpha = %f\n", alpha);
        if (alpha < tol) {
            printf("DEBUG: alpha too small, Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            exit_flag = 1;
        }
        else {
            vector_scale(&v, 1.0 / alpha);
            print_vector(&v, "normalized v");
        }

        // implicit QR factorization
        rho = sqrt(barrho * barrho + beta * beta);
        c = barrho / rho;
        s = beta / rho;
        theta = s * alpha;
        barrho = -c * alpha;
        phi = c * barphi;
        barphi = s * barphi;

        printf("updates\n");
        printf("iter %d: rho = %f,  c = %f, s = %f, theta = %f, barrho = %f, phi = %f, barphi = %f\n",
            iter, rho, c, s, theta, barrho, phi, barphi);;

        // update x and w
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x->values[i] += (phi / rho) * w.values[i];
        }
        print_vector(x, "updated x");
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            w.values[i] = v.values[i] - (theta / rho) * w.values[i];
        }
        //double norm_w = vector_norm(&w);
        //vector_scale(&w, 1.0 / norm_w);
        print_vector(&w, "updated w");

        // compute residual
        FEMVector Ax;
        initialize_vector(&Ax, b->size);
        matvec_mult(A, x, &Ax);

        double residual_norm = 0.0;
#pragma omp parallel for reduction(+:residual_norm)
        for (size_t i = 0; i < b->size; i++) {
            double diff = Ax.values[i] - b->values[i];
            residual_norm += diff * diff;
        }
        residual_norm = sqrt(residual_norm);
        free_vector(&Ax);

        printf("iter %d: ||Ax - b|| = %f\n",
            iter, residual_norm);

        // check convergence
        if (exit_flag || residual_norm < tol +  tol * b_norm) {
            printf("LSQR finished, iter = %d, residual = %e, exit_flag = %d\n", iter, residual_norm, exit_flag);
            break;
        }

    }

    // free memory
    free_vector(&u);
    free_vector(&u_old);
    free_vector(&v);
    free_vector(&v_old);
    free_vector(&w);
}
