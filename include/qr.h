#ifndef QR_H
#define QR_H

#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TOLERANCE 1e-6  // Floating-point tolerance

static inline double sign(double x) {
    return (x >= 0) ? 1.0 : -1.0;
}

// Parallelized Householder QR Decomposition
void householder_qr(const FEMMatrix* A, FEMMatrix* Q, FEMMatrix* R);

// Solves Ax = b using QR decomposition
void qr_solve(const FEMMatrix* Q, const FEMMatrix* R, const FEMVector* b, FEMVector* x);

#endif // QR_H
