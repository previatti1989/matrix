#ifndef GMRES_H
#define GMRES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_ops.h"  // Matrix operations (dot product, norm, etc.)
#include "qr.h"          // QR solver for least squares

#define MAX_ITER 1000
#define TOLERANCE 1e-6

// GMRES solver function prototype
void gmres_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, int k_max);

#endif // GMRES_H