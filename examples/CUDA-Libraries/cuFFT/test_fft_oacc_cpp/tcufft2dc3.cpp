#include <iostream>
#include <iomanip>
#include <complex>
#include "cufft.h"
#include "openacc.h"

int main()
{
    const int m = 768;
    const int n = 512;
    std::complex<float> a[m*n];
    std::complex<float> b[m*n];
    std::complex<float> c[m*n];
    float r[m*n];
    float q[m*n];
    cufftHandle plan1, plan2, plan3;
    int ierr;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            a[i*m+j] = 1.0;
            r[i*m+j] = 1.0;
            c[i*m+j] = 99.0;
        }
    }

    #pragma acc data copyin(a[0:m*n]) copyout(b[0:m*n],c[0:m*n])
    {
        ierr  = cufftPlan2d(&plan1, m, n, CUFFT_C2C);
        ierr += cufftSetStream(plan1,
                       (cudaStream_t) acc_get_cuda_stream(acc_async_sync));
        #pragma acc host_data use_device(a, b, c)
        {
            ierr += cufftExecC2C(plan1, (cufftComplex *) a,
                                        (cufftComplex *) b, CUFFT_FORWARD);
            ierr += cufftExecC2C(plan1, (cufftComplex *) b,
                                        (cufftComplex *) c, CUFFT_INVERSE);
        }

        // Currently need to mark these loops as independent
        #pragma acc kernels
        {
            #pragma acc loop independent
            for (int i = 0; i < n; ++i)
            {
                #pragma acc loop independent
                for (int j = 0; j < m; ++j)
                {
                    c[i*m+j] = std::complex<float>(c[i*m+j].real() / (m*n),
                                                   c[i*m+j].imag() / (m*n));
                }
            }
        }
    }
    float bmaxvalr = 0.0f;
    float bmaxvali = 0.0f;
    float bsumr = 0.0f;
    float bsumi = 0.0f;
    float cmaxval = 0.0f;
    float cabsval;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (b[i*m+j].real() > bmaxvalr) bmaxvalr = b[i*m+j].real();
            if (b[i*m+j].imag() > bmaxvali) bmaxvali = b[i*m+j].imag();
            bsumr += b[i*m+j].real();
            bsumi += b[i*m+j].imag();
            std::complex<float> x = a[i*m+j] - c[i*m+j];
            cabsval = sqrtf(x.real()*x.real() + x.imag()*x.imag());
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
    ierr += cufftSetStream(plan2,
                (cudaStream_t) acc_get_cuda_stream(acc_async_sync));
    ierr += cufftSetStream(plan3,
                (cudaStream_t) acc_get_cuda_stream(acc_async_sync));

    float rmaxval = 0.0f;
    #pragma acc data copyin(r[0:m*n]) create(b[0:m*n],q[0:m*n])
    {
        #pragma acc host_data use_device(r, b, q)
        {
            ierr += cufftExecR2C(plan2, r, (cufftComplex *) b);
            ierr += cufftExecC2R(plan3, (cufftComplex *) b, q);
        }
        #pragma acc kernels
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j)
                {
                    float x = fabs(r[i*m+j] - q[i*m+j] / (m*n));
                    if (x > rmaxval) rmaxval = x;
                }
            }
        }
    }
    std::cout << "Max error R2C/C2R: " << rmaxval << std::endl;
    ierr += cufftDestroy(plan1);
    ierr += cufftDestroy(plan2);
    ierr += cufftDestroy(plan3);

    if (ierr == 0)
        std::cout << " Test PASSED" << std::endl;
    else
        std::cout << " Test FAILED, ierr = " << ierr << std::endl;

    return ierr;
}
