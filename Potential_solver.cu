#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_wrapper.h"
#include "pointer_2d_matrix.h"

// Thread block size
#define BLOCK_SIZE  10  // number of threads in a direction of the block
#define M_WIDTH     100 // number of columns
#define M_HEIGHT    1 // number of rows

/*
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} cu_Matrix;
*/

// Forward declaration of the matrix multiplication kernel
__global__ void GaussSiedel_solver_Kernel(const cu_Matrix, const cu_Matrix, cu_Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void Potential_solver(const cu_Matrix A, const cu_Matrix B, cu_Matrix C)
{
    // Load A and B to device memory
    cu_Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    cu_Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    cu_Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);
    
    // Invoke kernel
    // dim3 dimGrid(32, 32);
    // dim3 dimBlock(38,26);
    
    ////////////////// Benchmarking  ///////////////
    
    // cudaEvent_t start, stop; 
	// cudaEventCreate(&start); 
	// cudaEventCreate(&stop); 
	// cudaEventRecord(start, 0); 
	 
	// /// your kernel call here 
    GaussSiedel_solver_Kernel<<<A.width/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C);
	 
	// cudaEventRecord(stop, 0); 

	// cudaEventSynchronize(stop); 
	 
	// float elapseTime; 
	// cudaEventElapsedTime(&elapseTime, start, stop); 
	// cout << "Time to run the kernel: "<< elapseTime << " ms "<< endl;
	
    ////////////////// Benchmarking  ///////////////


    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void GaussSiedel_solver_Kernel(cu_Matrix A, cu_Matrix B, cu_Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // for (int iter = 0 ; iter < 10000 ; iter++)
    // {
    //     if( (col>0) && (col<(A.width-1)) )
    //         Cvalue = 0.5*(A.elements[col+1]+A.elements[col-1]-B.elements[col]);

    //     if( col == 0 )
    //         Cvalue = 0.5*(A.elements[1]+A.elements[A.width-1]-B.elements[0]);
        
    //     if( col == (A.width-1) )
    //         Cvalue = 0.5*(A.elements[0]+A.elements[A.width-2]-B.elements[A.width-1]);

    //     __syncthreads();

    //     A.elements[col] = 0.8*Cvalue + (1.0-0.8)*A.elements[col];
    // }


    if( (col>0) && (col<(A.width-1)) )
    {
        for (int iter = 0 ; iter < 10000 ; iter++)
        {
            Cvalue = 0.5*(A.elements[col+1]+A.elements[col-1]-B.elements[col]);
            A.elements[col] = 0.8*Cvalue + (1.0-0.8)*A.elements[col];
            __syncthreads();
        }
    }

    if( col == 0 )
    {
        for (int iter = 0 ; iter < 10000 ; iter++)
        {
            Cvalue = 0.5*(A.elements[1]+A.elements[A.width-1]-B.elements[0]);
            A.elements[col] = 0.8*Cvalue + (1.0-0.8)*A.elements[col];
            __syncthreads();
        }
    }
    
    if( col == (A.width-1) )
    {
        for (int iter = 0 ; iter < 10000 ; iter++)
        {
            Cvalue = 0.5*(A.elements[0]+A.elements[A.width-2]-B.elements[A.width-1]);
            A.elements[col] = 0.8*Cvalue + (1.0-0.8)*A.elements[col];
            __syncthreads();
        }
    }


    C.elements[col] = Cvalue;
}

int get_grid_Potential(
	double* phi_grid,
	double* bvec){
	
    float *A, *B, *C;
    int i;
    cu_Matrix M_A, M_B, M_C; 

    A = (float*)malloc(M_WIDTH*sizeof(float));
    B = (float*)malloc(M_WIDTH*sizeof(float));
    C = (float*)malloc(M_WIDTH*sizeof(float));  

    srand((unsigned)time( NULL ));

    // initialize A[] and B[]
    for(i = 0; i < M_WIDTH; i++)
    {
        A[i] = (float)phi_grid[i];
        B[i] = (float)bvec[i];
        C[i] = 0.0;
    }
    
    M_A.width = M_WIDTH; M_A.height = M_HEIGHT;
    M_A.elements = A; 
    M_B.width = M_WIDTH; M_B.height = M_HEIGHT;
    M_B.elements = B; 
    M_C.width = M_WIDTH; M_C.height = M_HEIGHT;
    M_C.elements = C; 

        
    Potential_solver(M_A, M_B, M_C);
        
    
    for(i = 0; i < M_WIDTH; i++)
    {
        phi_grid[i] = (float)C[i];
        // cout << phi_grid[i] << endl;
    }
    // cin.get();

    free(A); free(B); free(C);
    return 0;
}

