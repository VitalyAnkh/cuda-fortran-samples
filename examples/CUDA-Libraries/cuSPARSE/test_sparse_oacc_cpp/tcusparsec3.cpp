/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE.
 * Modified for OpenACC from the CUDA Samples example ConjugateGradientUM
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cusparse.h>
#include <cublas_v2.h>

/* Need openacc for the stream definitions */
#include "openacc.h"

/* Need openacc_curand to generate data on the GPU */
#include "openacc_curand.h"

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
  unsigned long long seed = 12345ULL;
  unsigned long long seq  = 0ULL;
  unsigned long long offset = 0ULL;
  curandState_t state;
  #pragma acc parallel loop copy(I[0:N+1], J[0:nz], val[0:nz]) private(state)
  {
    for (int ii = 0; ii < N; ii++) {
      int jj = ii * 3;
      seed += 2ULL * ii;
      curand_init(seed, seq, offset, &state);
      if (ii == 0) {
        val[jj] = curand_uniform(&state) + 10.0f;
        I[ii] = 0;
        I[ii+1] = 2;
        I[N] = nz;
        J[jj] = 0;
      } else {
        val[jj] = curand_uniform(&state) + 10.0f;
        val[jj-2] = val[jj-1] = curand_uniform(&state);
        I[ii+1] = (ii == N-1) ? jj+1 : jj+2;
        J[jj] = ii;
        J[jj-1] = ii-1;
        J[jj-2] = ii;
      }
    }
  }
}

int main(int argc, char **argv)
{
    int N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    const int max_iter = 10000;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    float dot;
    float *r, *p, *Ax;
    int k;
    float alpha, beta, alpham1;

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) printf("Error cublas\n");
    cublasStatus = cublasSetStream(cublasHandle, (cudaStream_t)
                                    acc_get_cuda_stream(acc_async_sync));
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) printf("Error cublas\n");

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) printf("Error cusparse\n");
    cusparseStatus = cusparseSetStream(cusparseHandle, (cudaStream_t)
                                    acc_get_cuda_stream(acc_async_sync));
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) printf("Error cusparse\n");

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    N = 1048576;
    nz = (N-2)*3 + 4;

    I = new int[N+1];
    J = new int[nz];
    val = new float[nz];

    #pragma acc data copyout(I[0:(N+1)], J[0:nz], val[0:nz])
    {
    genTridiag(I, J, val, N, nz);

    // temp memory for CG
    r = new float[N];
    p = new float[N];
    Ax = new float[N];

    // output data
    x   = new float[N];
    rhs = new float[N];

    #pragma acc data copyout(rhs[0:N], x[0:N]), create(r[0:N], p[0:N], Ax[0:N])
    {

    #pragma acc kernels loop independent
    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
        r[i] = rhs[i];
    }

#if CUDA_VERSION <= 10020
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
#else
    size_t bsize;
    float *buffer;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecP;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    cusparseCreateCsr(&matA, N, N, nz, I, J, val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
#endif

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = r1 = 0.;

    /* These three processing steps all occur on the same stream */
    #pragma acc host_data use_device(val, I, J, x, Ax, r)
    {
#if CUDA_VERSION <= 10020
    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, x, &beta, Ax);
#else
    cusparseCreateDnVec(&vecP, N,  p, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, N,  x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, Ax, CUDA_R_32F);

    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, &bsize);

    buffer = new float[bsize];

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, buffer);
#endif
    cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);
    }
    #pragma acc kernels loop independent
    for (int i=0; i < N; i++)
    {
        r1 += r[i]*r[i];
    }

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            #pragma acc host_data use_device(p, r)
            {
            cublasStatus = cublasSscal(cublasHandle, N, &b, p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
            }
        }
        else
        {
            #pragma acc host_data use_device(p, r)
            {
            cublasStatus = cublasScopy(cublasHandle, N, r, 1, p, 1);
            }
        }

        #pragma acc host_data use_device(val, I, J, x, Ax, p, r)
        {
#if CUDA_VERSION <= 10020
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, p, &beta, Ax);
#else
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP, &beta, vecY, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, buffer);
#endif
        cublasStatus = cublasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);
        }

        #pragma acc wait(acc_async_sync)
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
    }
    }

    printf("Final residual: %e\n",sqrt(r1));

    fprintf(stdout,"&&&& uvm_cg test %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    printf("Test Summary:  Error amount = %f, result = %s\n", err, (k <= max_iter) ? "SUCCESS" : "FAILURE");
    exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
}
