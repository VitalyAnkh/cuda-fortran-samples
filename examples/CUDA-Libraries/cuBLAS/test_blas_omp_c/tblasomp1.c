#include "stdio.h"
#include "cublas_v2.h"
#include "omp.h"

int matmul( const int n, const double* const a, const double* const b, double * const c )
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    #pragma omp target enter data map( to: a[0:n*n], b[0:n*n] ) map( alloc: c[0:n*n] )
        #pragma omp target data use_device_ptr( a, b, c )
	    {
	    	cublasHandle_t handle;
	    	stat = cublasCreate(&handle);
	    	if ( CUBLAS_STATUS_SUCCESS != stat ) {
	    		printf("CUBLAS initialization failed\n");
	    	}
	        stat = cublasSetStream(handle, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));
	    	if ( CUBLAS_STATUS_SUCCESS != stat ) {
	    		printf("CUBLAS set stream failed\n");
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
    #pragma omp target exit data map( from: c[0:n*n] ) map( delete: a[0:n*n], b[0:n*n] )

	return CUBLAS_STATUS_SUCCESS == stat;
}

int main()
{
// "int const n=256;" would result in a warning
#define n 256
	double a[n*n];
	double b[n*n];
	double c[n*n];
	double expct[n*n];
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			expct[i*n+j] = 1.0;
			a[i*n+j] = 1.0;
			b[i*n+j] = 1.0/n;
		}
	}
	
    int error = !matmul( n, a, b, c );

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
#undef n
}
