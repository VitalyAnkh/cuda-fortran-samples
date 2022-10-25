#include <iostream>
#include <algorithm>
#include <iomanip>
#include <complex>
#include <array>
#include "cuda_runtime.h"
#include "cufft.h"
#include "omp.h"

// This is an example for `target depend nowait`. The `depend` clause makes sure
// that a `target nowait` kernel is queued to the specified CUDA stream.
//
// An ompx_get_cuda_stream call should go before the 'target nowait depend' region,
// which returns a stream that will be used for the next stream dependency chain
// (in case of 'target nowait depend'), or for the next independent 'target nowait'
// construct.
//
// In this example, we perform three convolutions using FFTs, which consists of
// three steps:
// 1. Forward FFT: a1 -> b1 = F(a1), same thing for a2 and a3;
// 2. Point-wise product: c1 = b2*b3, c2 = b1*b3, c3 = b1*b2;
// 3. Inverse FFT: c1 -> d1 = F^(-1)(c1), same thing for c2 and c3.
//    (And d1 = conv(a2, a3))
// `target depend nowait` makes sure that all operations for one convolution are
// executated on the same stream, with the correct order, although the three sets
// of operations can happen simultaneously.
//
// For demonstration purpose, one can remove the `depend` clause on the
// `omp target parallel for` lines and observe that some of the results are
// incorrect.
//
// May need to increase the stack limit (e.g., using `ulimit -s unlimited`) to
// avoid segmentation faults.

void check( int m, int n, std::complex< float > const * a, float val )
{

    float amaxvalr = 0.0f;
    float amaxvali = 0.0f;

    #pragma omp parallel for collapse( 2 ) reduction( max: amaxvalr, amaxvali )
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if ( fabs( a[i*m+j].real() - val ) > amaxvalr ) amaxvalr = fabs( a[i*m+j].real() - val );
            if ( a[i*m+j].imag() > amaxvali ) amaxvali = a[i*m+j].imag();
        }
    }

    std::cout << "Max error Convolution: " <<
                    std::setprecision(7) << std::setw(10) << std::scientific <<
                    std::complex<float>(amaxvalr, amaxvali) <<
                    std::endl;

    if ( amaxvalr < 1.e-6 && amaxvali < 1.e-6 )
        std::cout << " Test PASSED" << std::endl;
    else
        std::cout << " Test FAILED, amaxvalr = " << amaxvalr
                  << ", amaxvali = " << amaxvali
                  << std::endl;
}

#ifndef N
#define N (2*768)
#endif

#ifndef M
#define M (2*512)
#endif

#ifndef NUM_FFT
#define NUM_FFT 4
#endif

int main()
{
    const int m = M;
    const int n = N;
    const int mn = m*n;
    const int nFFT = NUM_FFT;

    std::array< std::array< std::complex< float >, mn >, nFFT > a1, a2, a3;
    std::array< std::array< std::complex< float >, mn >, nFFT > b1, b2, b3;
    std::array< std::array< std::complex< float >, mn >, nFFT > c1, c2, c3;
    std::array< std::array< std::complex< float >, mn >, nFFT > d1, d2, d3;

    std::array< cufftHandle, nFFT > plans;
    int ierr = 0;

    std::array< cudaStream_t, nFFT > streams;

    for ( std::size_t i = 0; i < nFFT; ++i )
    {
        std::fill_n( a1[i].begin(), mn, std::complex< float >( 2.f + i, 0.f ) );
        std::fill_n( a2[i].begin(), mn, std::complex< float >( 3.f + i, 0.f ) );
        std::fill_n( a3[i].begin(), mn, std::complex< float >( 4.f + i, 0.f ) );
        ierr += cufftPlan2d(&plans[i], m, n, CUFFT_C2C);
    }

    for ( std::size_t i = 0; i < nFFT; ++i )
    {
        std::cout << "Initializing data on device, iteration " << i << ".." << std::endl;

        streams[i] = (cudaStream_t)ompx_get_cuda_stream( omp_get_default_device(), true );
        ierr += cufftSetStream( plans[i], streams[i] );

        std::complex<float> * a1ptr = &a1[i][0], * b1ptr = &b1[i][0], * c1ptr = &c1[i][0], * d1ptr = &d1[i][0];
        std::complex<float> * a2ptr = &a2[i][0], * b2ptr = &b2[i][0], * c2ptr = &c2[i][0], * d2ptr = &d2[i][0];
        std::complex<float> * a3ptr = &a3[i][0], * b3ptr = &b3[i][0], * c3ptr = &c3[i][0], * d3ptr = &d3[i][0];

        #pragma omp target enter data map( to   :a1ptr[0:mn], a2ptr[0:mn], a3ptr[0:mn] ) depend(out:streams[i]) nowait
        #pragma omp target enter data map( alloc:b1ptr[0:mn], b2ptr[0:mn], b3ptr[0:mn] ) depend(out:streams[i]) nowait
        #pragma omp target enter data map( alloc:c1ptr[0:mn], c2ptr[0:mn], c3ptr[0:mn] ) depend(out:streams[i]) nowait
        #pragma omp target enter data map( alloc:d1ptr[0:mn], d2ptr[0:mn], d3ptr[0:mn] ) depend(out:streams[i]) nowait
    }

    for ( std::size_t i = 0; i < nFFT; ++i )
    {
        std::cout << "Computing, iteration " << i << ".." << std::endl;

        std::complex<float> * a1ptr = &a1[i][0], * b1ptr = &b1[i][0], * c1ptr = &c1[i][0], * d1ptr = &d1[i][0];
        std::complex<float> * a2ptr = &a2[i][0], * b2ptr = &b2[i][0], * c2ptr = &c2[i][0], * d2ptr = &d2[i][0];
        std::complex<float> * a3ptr = &a3[i][0], * b3ptr = &b3[i][0], * c3ptr = &c3[i][0], * d3ptr = &d3[i][0];

        #pragma omp target data use_device_ptr( a1ptr, b1ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) a1ptr,
                                           (cufftComplex *) b1ptr, CUFFT_FORWARD);
        }

        #pragma omp target data use_device_ptr( a2ptr, b2ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) a2ptr,
                                           (cufftComplex *) b2ptr, CUFFT_FORWARD);
        }

        #pragma omp target data use_device_ptr( a3ptr, b3ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) a3ptr,
                                           (cufftComplex *) b3ptr, CUFFT_FORWARD);
        }

        #pragma omp target teams distribute parallel for depend(inout:streams[i]) nowait
        for ( std::size_t j = 0; j < mn; ++j )
        {
            c3ptr[j] = b1ptr[j] * b2ptr[j];
            c3ptr[j] = std::complex<float>( c3ptr[j].real() / mn / mn, c3ptr[j].imag() );
        }

        #pragma omp target teams distribute parallel for depend(inout:streams[i]) nowait
        for ( std::size_t j = 0; j < mn; ++j )
        {
            c2ptr[j] = b1ptr[j] * b3ptr[j];
            c2ptr[j] = std::complex<float>( c2ptr[j].real() / mn / mn, c2ptr[j].imag() );
        }

        #pragma omp target teams distribute parallel for depend(inout:streams[i]) nowait
        for ( std::size_t j = 0; j < mn; ++j )
        {
            c1ptr[j] = b2ptr[j] * b3ptr[j];
            c1ptr[j] = std::complex<float>( c1ptr[j].real() / mn / mn, c1ptr[j].imag() );
        }

        #pragma omp target data use_device_ptr( c1ptr, d1ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) c1ptr,
                                           (cufftComplex *) d1ptr, CUFFT_INVERSE);
        }

        #pragma omp target data use_device_ptr( c2ptr, d2ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) c2ptr,
                                           (cufftComplex *) d2ptr, CUFFT_INVERSE);
        }

        #pragma omp target data use_device_ptr( c3ptr, d3ptr )
        {
            ierr += cufftExecC2C(plans[i], (cufftComplex *) c3ptr,
                                           (cufftComplex *) d3ptr, CUFFT_INVERSE);
        }
   }

    for ( std::size_t i = 0; i < nFFT; ++i )
    {
        std::cout << "Finalizing data, iteration " << i << ".." << std::endl;

        std::complex<float> * a1ptr = &a1[i][0], * b1ptr = &b1[i][0], * c1ptr = &c1[i][0], * d1ptr = &d1[i][0];
        std::complex<float> * a2ptr = &a2[i][0], * b2ptr = &b2[i][0], * c2ptr = &c2[i][0], * d2ptr = &d2[i][0];
        std::complex<float> * a3ptr = &a3[i][0], * b3ptr = &b3[i][0], * c3ptr = &c3[i][0], * d3ptr = &d3[i][0];

        #pragma omp target exit data map( from  : d1ptr[0:mn], d2ptr[0:mn], d3ptr[0:mn] ) depend(in:streams[i]) nowait
        #pragma omp target exit data map( delete: a1ptr[0:mn], a2ptr[0:mn], a3ptr[0:mn] ) depend(in:streams[i]) nowait
        #pragma omp target exit data map( delete: b1ptr[0:mn], b2ptr[0:mn], b3ptr[0:mn] ) depend(in:streams[i]) nowait
        #pragma omp target exit data map( delete: c1ptr[0:mn], c2ptr[0:mn], c3ptr[0:mn] ) depend(in:streams[i]) nowait
    }

    #pragma omp taskwait

    // a1 = 2+i, a2 = 3+i, a3 = 4+i
    // d1 = conv(a2,a3), d2 = conv(a1,a3), d3 = conv(a1,a2)
    for ( std::size_t i = 0; i < nFFT; ++i )
    {
        check( m, n, d1[i].data(), ( 3. + i ) * ( 4. + i ) );
        check( m, n, d2[i].data(), ( 2. + i ) * ( 4. + i ) );
        check( m, n, d3[i].data(), ( 2. + i ) * ( 3. + i ) );
    }

    for ( std::size_t i = 0; i < nFFT; ++i )
    {
         ierr += cufftDestroy(plans[i]);
    }

    return ierr;
}
