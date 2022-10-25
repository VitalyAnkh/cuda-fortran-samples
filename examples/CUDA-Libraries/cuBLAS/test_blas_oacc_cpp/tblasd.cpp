#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "stdio.h"

int main()
{
    float a[50];
    float b[50];
    float *a_d;
    float *b_d;
    int i, j, istat, nfailures;
    int n = 50;
    cublasHandle_t h;

    /* Initialize */
    for (i = 0; i < n; i++) {
       a[i] = 1.0f;
       b[i] = 2.0f;
    }
    cudaMalloc((void**)(&a_d), n*4);
    cudaMalloc((void**)(&b_d), n*4);

    cudaMemcpy(a_d, a, n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*4, cudaMemcpyHostToDevice);

    istat = cublasCreate(&h);
    for (j = 0; j < 101; j++) {
        if (istat == CUBLAS_STATUS_SUCCESS)
            istat = cublasSswap(h,n,a_d,1,b_d,1);
    }
    if (istat == CUBLAS_STATUS_SUCCESS)
        istat = cublasDestroy(h);

    cudaMemcpy(a, a_d, n*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, b_d, n*4, cudaMemcpyDeviceToHost);

    nfailures = 0;
    printf("Should be 2.0\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n",a[i]);
        if (a[i] != 2.0f) nfailures++;
    }
    printf("Should be 1.0\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n",b[i]);
        if (b[i] != 1.0f) nfailures++;
    }

    if (!nfailures) {
        printf(" Test PASSED\n");
    } else {
        printf(" Test FAILED\n");
    }
}
