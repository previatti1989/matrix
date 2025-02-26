#ifndef SOLVERS_H
#define SOLVERS_H

#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Solves Ax = b using CG
void cg_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter);

// GMRES solver function prototype
void gmres_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter, int k_max);

// LSQR Solver function prototype
void lsqr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter);

static inline double sign(double x) {
    return (x >= 0) ? 1.0 : -1.0;
}

// Parallelized Householder QR Decomposition
void householder_qr(const FEMMatrix* A, FEMMatrix* Q, FEMMatrix* R);

// Solves Ax = b using QR decomposition
void qr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x);

#endif // SOLVERS_H