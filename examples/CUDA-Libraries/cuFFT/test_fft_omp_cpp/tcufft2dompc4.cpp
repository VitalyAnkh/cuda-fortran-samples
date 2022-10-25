#include <iostream>
#include <iomanip>
#include <complex>
#include "cuda_runtime.h"
#include "cufft.h"
#include "omp.h"

int main()
{
    const int m = 768;
    const int n = 512;
    std::complex< float > * const a = new std::complex< float >[m*n];
    std::complex< float > * const b = new std::complex< float >[m*n];
    std::complex< float > * const c = new std::complex< float >[m*n];
    float * const r = new float[m*n];
    float * const q = new float[m*n];

    cufftHandle plan1, plan2, plan3;
    int ierr;

    std::fill_n( a, m*n, std::complex< float >( 1.f, 0.f ) );
    std::fill_n( r, m*n, 1.f );

    ierr  = cufftPlan2d(&plan1, m, n, CUFFT_C2C);
    ierr += cufftSetStream(plan1, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));

    {
        ierr += cufftExecC2C(plan1, (cufftComplex *) a,
                                    (cufftComplex *) b, CUFFT_FORWARD);
        ierr += cufftExecC2C(plan1, (cufftComplex *) b,
                                    (cufftComplex *) c, CUFFT_INVERSE);
    }

    cudaDeviceSynchronize();

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
            c[i*m+j] = std::complex< float >( c[i*m+j].real() / (m*n), c[i*m+j].imag() );
            if ( b[i*m+j].real() > bmaxvalr ) bmaxvalr = b[i*m+j].real();
            if ( b[i*m+j].imag() > bmaxvali ) bmaxvali = b[i*m+j].imag();
            bsumr += b[i*m+j].real();
            bsumi += b[i*m+j].imag();
            std::complex< float > x = a[i*m+j] - c[i*m+j];
            float cabsval = sqrtf( x.real() * x.real() + x.imag() * x.imag() );
            if (cabsval > cmaxval) cmaxval = cabsval;
        }
    }

    std::cout << "Max error C2C FWD: " <<
                    std::setprecision(7) << std::setw(10) << std::scientific <<
                    std::complex<float>(bmaxvalr-bsumr, bmaxvali) <<
                    std::endl;
    std::cout << "Max error C2C INV: " << cmaxval << std::endl;

    ierr += cufftPlan2d(&plan2, m, n, CUFFT_R2C);
    ierr += cufftPlan2d(&plan3, m, n, CUFFT_C2R);
    ierr += cufftSetStream(plan2, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));
    ierr += cufftSetStream(plan3, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));

    float rmaxval = 0.0f;

    {
        ierr += cufftExecR2C(plan2, r, (cufftComplex *) b);
        ierr += cufftExecC2R(plan3, (cufftComplex *) b, q);
    }

    cudaDeviceSynchronize();

    #pragma omp parallel for collapse( 2 ) reduction( max: rmaxval )
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            q[i*m+j] /= (m*n);

            float x = fabs(r[i*m+j] - q[i*m+j]);
            if (x > rmaxval) rmaxval = x;
        }
    }

    std::cout << "Max error R2C/C2R: " << rmaxval << std::endl;

    ierr += cufftDestroy(plan1);
    ierr += cufftDestroy(plan2);
    ierr += cufftDestroy(plan3);

    if (ierr == 0 && bmaxvalr - bsumr < 1.e-6 && bmaxvali < 1.e-6 && cmaxval < 1.e-6 && rmaxval < 1.e-6 )
        std::cout << " Test PASSED" << std::endl;
    else
        std::cout << " Test FAILED, ierr = " << ierr << ", bmaxvalr - bsumr = " << bmaxvalr - bsumr
                  << ", bmaxvali = " << bmaxvali << ", cmaxval = " << cmaxval << ", rmaxval = " << rmaxval
                  << std::endl;

    delete[] a, b, c, r, q;

    return ierr;
}
