#include "stdio.h"
#include "cublasXt.h"

#define N 1024

int matmul( const int n, const double* const a, const double* const b,
                                                       double * const c )
{
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    cublasXtHandle_t handle;
    stat = cublasXtCreate(&handle);
    if ( CUBLAS_STATUS_SUCCESS != stat ) {
	printf("CUBLAS initialization failed\n");
    }
			
    if ( CUBLAS_STATUS_SUCCESS == stat ) {
	const double alpha = 1.0;
	const double beta = 0.0;
        int devices[1] = { 0 };
        if (cublasXtDeviceSelect(handle,1,devices) != CUBLAS_STATUS_SUCCESS) {
	    printf("cublasXtDeviceSelect failed\n");
        } else {
	    stat = cublasXtDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                                           &alpha, a, n, b, n, &beta, c, n);
	    if (stat != CUBLAS_STATUS_SUCCESS) {
	        printf("cublasXtDgemm failed\n");
	    }
	}
	cublasXtDestroy(handle);
    }
    return CUBLAS_STATUS_SUCCESS == stat;
}

int main()
{
	const int n = N;
	double a[N*N];
	double b[N*N];
	double c[N*N];
	double expct[N*N];
	int error = 0;
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			expct[i*n+j] = 1.0;
			a[i*n+j] = 1.0;
			b[i*n+j] = 1.0/n;
		}
	}
	error = !matmul( n, a, b, c );
        if (error) {
          printf(" Test FAILED\n");
        } else {
	    int nfailures = 0;
	    printf("%lf %lf\n", c[0], c[n*n-1]);
	    for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
		    if (expct[i*n+j] != c[i*n+j]) nfailures++;
		}
	    }
	    if (nfailures)
          	printf(" Test FAILED\n");
	    else
          	printf(" Test PASSED\n");
  	}
	return error;
}
