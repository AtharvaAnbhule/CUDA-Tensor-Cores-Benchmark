#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <iostream>

template<typename T>
__global__ void initialize_matrix(T* matrix, long n_rows, long n_cols, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
    if(workerID<n_elements)
    {
        matrix[workerID] = val;
    }
}

template<typename T>
__global__ void initialize_colnegpos_matrix(T* matrix, long n_rows, long n_cols, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
    long col = workerID / n_rows;
    if(workerID<n_elements)
    {
        if(col % 2 == 0)
        {
            matrix[workerID] = +val;
        }
        else
        {
            matrix[workerID] = -val;
        }
    }
}

template<typename T>
__global__ void initialize_rownegpos_matrix(T* matrix, long n_rows, long n_cols, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
    long row = workerID % n_rows;
    if(workerID<n_elements)
    {
        if(row % 2 == 0)
        {
            matrix[workerID] = +val;
        }
        else
        {
            matrix[workerID] = -val;
        }
    }
}

template<typename T>
__global__ void initialize_colposneg_matrix(T* matrix, long n_rows, long n_cols, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
    long col = workerID / n_rows;
    if(workerID<n_elements)
    {
        if(col % 2 == 0)
        {
            matrix[workerID] = -val;
        }
        else
        {
            matrix[workerID] = +val;
        }
    }
}


template <typename T>
__global__ void view_matrix_fp(T* matrix, long n_rows, long n_cols)
{
    for(long col=0; col<n_cols; col++)
    {
        for(long row=0; row<n_rows; row++)
        {
            double temp = matrix[col*n_rows+row];
            printf("%f ", temp);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void view_matrix_int(T* matrix, long n_rows, long n_cols)
{
    for(long col=0; col<n_cols; col++)
    {
        for(long row=0; row<n_rows; row++)
        {
            int temp = matrix[col*n_rows+row];
            printf("%d ", temp);
        }
        printf("\n");
    }
}