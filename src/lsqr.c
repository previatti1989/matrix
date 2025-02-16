
#include "lsqr.h"
#include "matrix_ops.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void lsqr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter) {
    int n = A->rows;
    FEMVector v, u, v_old, u_old, w, x_old;
    initialize_vector(&v, n);
    initialize_vector(&u, n);
    initialize_vector(&w, n);
    initialize_vector(&x_old, n);
    initialize_vector(&v_old, n);
    initialize_vector(&u_old, n);

    double beta, alpha, rho, rho_old, c, s, theta, phi;
    double b_norm = vector_norm(b);

    if (b_norm == 0) {
        printf("DEBUG: b is zero! Exiting LSQR early.\n");
        set_vector_zero(x);
        return;
    }
    copy_vector(&u, b);
    print_vector(b, "Initial b");
    beta = vector_norm(&u);
    printf("Initial beta = %f\n", beta);

    if (beta < tol) {
        printf("DEBUG: Beta too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&u, 1.0 / beta);

    matvec_mult_transpose(A, &u, &v);
    alpha = vector_norm(&v);
    if (alpha < tol) {
        printf("DEBUG: Alpha too small, exiting LSQR.\n");
        set_vector_zero(x);
        return;
    }
    vector_scale(&v, 1.0 / alpha);

    set_vector_zero(x);
    phi = beta;
    theta = 0;
    rho_old = alpha;

    for (int iter = 0; iter < max_iter; iter++) {
        printf("\n=== Iteration %d ===\n", iter);
        fflush(stdout); // Ensure immediate printing

        print_vector(x, "Previous x");

        if (iter > 0 && phi < tol * b_norm) {
            printf("LSQR Converged: ||Ax - b|| = %f < tol * ||b|| = %f\n", phi, tol * b_norm);
            break;
        }

        print_vector(&u, "Vector u before update");
        print_vector(&v, "Vector v before update");

        copy_vector(&u_old, &u);
        copy_vector(&v_old, &v);

        print_vector(&u_old, "u_old before matvec_mult");
        print_vector(&u, "u before matvec_mult(A, v, u)");

        matvec_mult(A, &v, &u);
        print_vector(&u, "u after matvec_mult(A, v, u)");

        FEMVector u_temp;
        initialize_vector(&u_temp, n);
        copy_vector(&u_temp, &u);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            u.values[i] -= u_old.values[i]; // Prevents u from going to zero
        }
        free_vector(&u_temp);

        print_vector(&u, "u after subtracting alpha * u_old");

        double debug_norm = 0.0;
        for (int i = 0; i < n; i++) {
            debug_norm += u.values[i] * u.values[i];
        }
        debug_norm = sqrt(debug_norm);
        printf("DEBUG: Manually computed norm of u = %f\n", debug_norm);

        beta = vector_norm(&u);
        printf("Iteration %d: beta = %f\n", iter, beta);
        if (beta < tol) {
            printf("DEBUG: Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            break;
        }
        
        vector_scale(&u, 1.0 / beta);
        print_vector(&u, "Normalized u");

        matvec_mult_transpose(A, &u, &v);
        print_vector(&v, "Vector v after ATu");

        print_vector(&v_old, "v_old before subtraction");
        print_vector(&v, "v before subtracting beta * v_old");

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            v.values[i] -= beta * v_old.values[i];
        }

        print_vector(&v, "v after subtracting beta * v_old");

        alpha = vector_norm(&v);
        if (alpha < tol) {
            printf("DEBUG: Stopping at iteration %d, alpha = %f, beta = %f, phi = %f\n", iter, alpha, beta, phi);
            break;
        }
        printf("Iteration %d: alpha = %f\n", iter, alpha);
        vector_scale(&v, 1.0 / alpha);
        print_vector(&v, "Normalized v");

        double uv_dot = 0.0;
#pragma omp parallel for reduction(+:uv_dot)
        for (int i = 0; i < n; i++) {
            uv_dot += u.values[i] * v.values[i];
        }
        if (fabs(uv_dot) > tol) {
            printf("Iteration %d: WARNING - Re-orthogonalizing (uv = %f)\n", iter, uv_dot);
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                v.values[i] -= uv_dot * u.values[i];
            }
            double v_norm = vector_norm(&v);
            vector_scale(&v, 1.0 / v_norm);
        }

        rho = sqrt(rho_old * rho_old + beta * beta);
        printf("Iteration %d: rho = %f\n", iter, rho);
        if (rho == 0 || isnan(rho)) {
            printf("DEBUG: rho became zero or NaN! Stopping LSQR.\n");
            break;
        }
        c = rho_old / rho;
        s = beta / rho;
        theta = s * alpha;
        if (rho == 0 || isnan(rho)) {
            printf("DEBUG: rho became zero or NaN! Stopping LSQR.\n");
            break;
        }
        rho_old = rho;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            x->values[i] += (phi / rho) * v.values[i];
        }

        phi *= c;

        printf("Iteration %d: alpha = %f, beta = %f\n", iter, alpha, beta);
        printf("Iteration %d: rho = %f, c = %f, s = %f, theta = %f\n", iter, rho, c, s, theta);
        print_vector(&u, "Updated u");
        print_vector(&v, "Updated v");
        print_vector(x, "Updated x");

    }

    free_vector(&u);
    free_vector(&u_old);
    free_vector(&v);
    free_vector(&v_old);
    free_vector(&w);
    free_vector(&x_old);
}
