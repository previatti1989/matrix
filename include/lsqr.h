#ifndef LSQR_H
#define LSQR_H

#include "matrix_ops.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// LSQR Solver function prototype
void lsqr_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter);

#endif // LSQR_H