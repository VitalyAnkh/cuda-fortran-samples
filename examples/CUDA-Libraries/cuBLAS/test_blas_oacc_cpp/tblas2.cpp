#include <iostream>
#include <cmath>
#include "cublas_v2.h"
bool matmul( const int n, const double* const a, const double* const b,
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
				std::cerr<<"CUBLAS initialization failed"<<std::endl;
			}
			
			if ( CUBLAS_STATUS_SUCCESS == stat )
			{
				const double alpha = 1.0;
				const double beta = 0.0;
				stat = cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, a, n, b, n, &beta, c, n);
				if (stat != CUBLAS_STATUS_SUCCESS) {
					std::cerr<<"cublasDgemm failed"<<std::endl;
				}
			}
			cublasDestroy(handle);
		}
	}
	return CUBLAS_STATUS_SUCCESS == stat;
}

int main()
{
	const int n = 256;
	double* const a = new double[n*n];
	double* const b = new double[n*n];
	double* const c = new double[n*n];
	double* const exp = new double[n*n];
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			exp[i*n+j] = 1.0;
			a[i*n+j] = 1.0;
			b[i*n+j] = 1.0/n;
		}
	}
	
	bool error = false;
	
	#pragma acc data copyin( a[0:n*n], b[0:n*n] ) copyout( c[0:n*n] )
	{
		error = !matmul( n, a, b, c );
	}	
        if (error) {
          std::cout << " Test FAILED" << error << std::endl;
        } else {
	    int nfailures = 0;
	    for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
		    if (exp[i*n+j] != c[i*n+j]) nfailures++;
		}
	    }
	    if (nfailures)
          	std::cout << " Test FAILED" << std::endl;
	    else
          	std::cout << " Test PASSED" << std::endl;
  	}

	delete[] exp;
	delete[] c;
	delete[] b;
	delete[] a;
	return error;
}
