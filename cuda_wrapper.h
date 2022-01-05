#if !defined(__WRAPPER__)
#define __WRAPPER__

#include "pointer_2d_matrix.h"

// Matrices are stored in row-major order:
// // M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
     int width;
     int height;
     float* elements;
} cu_Matrix;

int get_grid_Potential(
	double* phi_grid ,
	double* bvec);

int get_grid_Efield(
	double* phi_grid ,
	double** Gmtx);

#endif 
