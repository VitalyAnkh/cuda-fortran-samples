#include "cufft.h"
#include "complex.h"
#include "omp.h"
#include "cuda_runtime.h"

int main()
{
    int m = 768;
    int n = 512;
    float complex a[m*n];
    float complex b[m*n];
    float complex c[m*n];
    float r[m*n];
    float q[m*n];

    cufftHandle plan1, plan2, plan3;
    int ierr;

    for ( int i = 0; i < m*n; ++i )
    {
        a[i] = 1.f;
        r[i] = 1.f;
    }

    ierr  = cufftPlan2d(&plan1, m, n, CUFFT_C2C);
    ierr += cufftSetStream(plan1, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));

    #pragma omp target enter data map( to:a[0:m*n] ) map( alloc:b[0:m*n], c[0:m*n] )
        #pragma omp target data use_device_ptr( a, b, c )
        {
            ierr += cufftExecC2C(plan1, (cufftComplex *) a,
                                        (cufftComplex *) b, CUFFT_FORWARD);
            ierr += cufftExecC2C(plan1, (cufftComplex *) b,
                                        (cufftComplex *) c, CUFFT_INVERSE);
        }
    #pragma omp target exit data map( from: b[0:m*n], c[0:m*n] ) map( delete: a[0:m*n] )

    float bmaxvalr = 0.0f;
    float bmaxvali = 0.0f;
    float bsumr = 0.0f;
    float bsumi = 0.0f;
    float cmaxval = 0.0f;

    #pragma omp parallel for collapse( 2 ) reduction( max: bmaxvalr, bmaxvali, cmaxval ) reduction( +: bsumr, bsumi )
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
#ifdef DEBUG
            if ( i % 100 == 0 && j % 100 == 0 )
            {
                printf("Check C2C [%d,%d] a = (%f, %f), c = (%f, %f)\n", i, j, crealf( a[i*m+j] ), cimagf( a[i*m+j] ),
                       crealf( c[i*m+j] / (m*n) ), cimagf( c[i*m+j] ) / (m*n) );
            }
#endif
            if ( fabs( crealf( b[i*m+j] ) ) > bmaxvalr ) bmaxvalr = fabs( crealf( b[i*m+j] ) );
            if ( cimagf( b[i*m+j] ) > bmaxvali) bmaxvali = cimagf( b[i*m+j] );
            bsumr += crealf( b[i*m+j] );
            bsumi += cimagf( b[i*m+j] );
            float complex x = a[i*m+j] - c[i*m+j] / (m*n);
            float cabsval = sqrtf( crealf( x ) * crealf( x ) + cimagf( x ) * cimagf( x ) );
            if (cabsval > cmaxval) cmaxval = cabsval;
        }
    }

    printf("Max error C2C FWD: (%f, %f)\n",bmaxvalr - bsumr, bmaxvali);
    printf("Max error C2C INV: %f\n",cmaxval);

    ierr += cufftPlan2d(&plan2, m, n, CUFFT_R2C);
    ierr += cufftPlan2d(&plan3, m, n, CUFFT_C2R);
    ierr += cufftSetStream(plan2, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));
    ierr += cufftSetStream(plan3, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));

    float rmaxval = 0.0f;

    #pragma omp target enter data map( to:r[0:m*n] ) map( alloc:b[0:m*n], q[0:m*n] )
    #pragma omp target data use_device_ptr( r, b, q )
    {
        ierr += cufftExecR2C(plan2, r, (cufftComplex *) b);
        ierr += cufftExecC2R(plan3, (cufftComplex *) b, q);
    }
    #pragma omp target exit data map( from: q[0:m*n] ) map( delete: r[0:m*n], b[0:m*n] )

    #pragma omp parallel
    {
        #pragma omp for collapse( 2 ) reduction( max: rmaxval )
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                float x = fabs(r[i*m+j] - q[i*m+j] / (m*n));
#ifdef DEBUG
                if ( i % 100 == 0 && j % 100 == 0 )
                {
                    printf( "Check R2C/C2R [%d, %d] r = %f; q = %f\n", i, j, r[i*m+j], q[i*m+j] / (m*n) );
                }
#endif
                if (x > rmaxval) rmaxval = x;
            }
        }
    }

    printf("Max error R2C/C2R: %f\n",rmaxval);

    ierr += cufftDestroy(plan1);
    ierr += cufftDestroy(plan2);
    ierr += cufftDestroy(plan3);

    if (ierr == 0 && bmaxvalr - bsumr < 1.e-6 && bmaxvali < 1.e-6 && cmaxval < 1.e-6 && rmaxval < 1.e-6 )
        printf(" Test PASSED\n");
    else
        printf(" Test FAILED, ierr = %d, bmaxvalr - bsumr = %f, bmaxvali = %f, cmaxval = %f, rmaxval = %f\n",
               ierr, bmaxvalr - bsumr, bmaxvali, cmaxval, rmaxval);

    return ierr;
}
