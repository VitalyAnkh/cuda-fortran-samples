#include "cufft.h"
#include "openacc.h"
#include "complex.h"

#define M 768
#define N 512

int main()
{
    const int m = M;
    const int n = N;
    float complex a[M*N];
    float complex b[M*N];
    float complex c[M*N];
    float r[M*N];
    float q[M*N];
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
        #pragma acc kernels
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j)
                {
                    c[i*m+j] = c[i*m+j] / (m*n);
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
            if (crealf(b[i*m+j]) > bmaxvalr) bmaxvalr = crealf(b[i*m+j]);
            if (cimagf(b[i*m+j]) > bmaxvali) bmaxvali = cimagf(b[i*m+j]);
            bsumr += crealf(b[i*m+j]);
            bsumi += cimagf(b[i*m+j]);
            float complex x = a[i*m+j] - c[i*m+j];
            cabsval = sqrtf(crealf(x)*crealf(x) + cimagf(x)*cimagf(x));
            if (cabsval > cmaxval) cmaxval = cabsval;
        }
    }

    printf("Max error C2C FWD: (%f, %f)\n",bmaxvalr - bsumr, bmaxvali);
    printf("Max error C2C INV: %f\n",cmaxval);

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
    printf("Max error R2C/C2R: %f\n",rmaxval);
    ierr += cufftDestroy(plan1);
    ierr += cufftDestroy(plan2);
    ierr += cufftDestroy(plan3);

    if (ierr == 0)
        printf(" Test PASSED\n");
    else
        printf(" Test FAILED, ierr = %d\n",ierr);

    return ierr;
}
