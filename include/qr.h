#ifndef QR_H
#define QR_H

#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static inline double sign(double x) {
    return (x >= 0) ? 1.0 : -1.0;
}

// Parallelized Householder QR Decomposition
void householder_qr(const FEMMatrix* A, FEMMatrix* Q, FEMMatrix* R);

// Solves Ax = b using QR decomposition
void qr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x);

#endif // QR_H
