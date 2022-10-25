#include "stdio.h"
#include "cublas_v2.h"

#define N 256

int matmul( const int n, const double* const a, const double* const b,
                                                       double * const c )
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
	#pragma acc data pcopyin( a[0:n*n], b[0:n*n] ) pcopyout( c[0:n*n] )
	{
		#pragma acc host_data use_device( a, b, c )
		{
			cublasHandle_t handle;
			stat = cublasCreate(&handle);
			if ( CUBLAS_STATUS_SUCCESS != stat ) {
				printf("CUBLAS initialization failed\n");
			}
			
			if ( CUBLAS_STATUS_SUCCESS == stat )
			{
				const double alpha = 1.0;
				const double beta = 0.0;
				stat = cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n);
				if (stat != CUBLAS_STATUS_SUCCESS) {
					printf("cublasDgemm failed\n");
				}
			}
			cublasDestroy(handle);
		}
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
	
	
	#pragma acc data copyin( a[0:n*n], b[0:n*n] ) copyout( c[0:n*n] )
	{
		error = !matmul( n, a, b, c );
	}	
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
