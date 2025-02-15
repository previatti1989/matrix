#ifndef QR_H
#define QR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TOLERANCE 1e-6  // Floating-point tolerance

// Function to return the sign of a number
double sign(double x);

// Function to compute the 2-norm of a vector (parallelized)
double norm(double* x, int n);

// Function to allocate a 2D matrix dynamically
double** allocate_matrix(int rows, int cols);

// Function to free a 2D matrix
void free_matrix(double** matrix, int rows);

// Parallelized Householder QR Decomposition
void householder_qr(double** A, double** Q, double** R, int m, int n);

// Solves Ax = b using QR decomposition
void qr_solve(double** Q, double** R, double* b, double* x, int rows, int cols);

#endif // QR_H
