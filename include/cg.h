#ifndef CG_H
#define CG_H

#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Solves Ax = b using CG
void cg_solver(const FEMMatrix* A, const FEMVector* b, FEMVector* x, double tol, int max_iter);

#endif // CG_H