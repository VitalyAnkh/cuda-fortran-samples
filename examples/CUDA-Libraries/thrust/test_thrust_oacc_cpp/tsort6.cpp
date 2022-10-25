#include <iostream>
#include <iomanip>
#include "curand.h"
#include "openacc.h"

int mysort (int n);
void thrust_float_sort_wrapper( float *dev_data, int *dev_idx,
                                    int N, cudaStream_t stream);

int main()
{
    mysort(100);
    mysort(500);
    mysort(1000);
    mysort(5000);
    mysort(10000);
}

int mysort(int n)
{
    float* const a = new float[n];
    int* const idx = new int[n];
    int istat;
    float suml, sumh, sumf;
    curandGenerator_t g;

    cudaStream_t oaccstream = (cudaStream_t) acc_get_cuda_stream(acc_async_sync);

    for (int i = 0; i < n; i++)
        a[i] = 0.0f;

    suml = 0.0f; sumh = 0.0f; sumf = 0.0f;

    istat  = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
    istat += curandSetStream(g, oaccstream);

    #pragma acc data copy(a[0:n], idx[0:n])
    {
        #pragma acc host_data use_device(a, idx)
        {
            istat += curandGenerateUniform(g, a, n);
            thrust_float_sort_wrapper(a, idx, n, oaccstream);
        }
        #pragma acc kernels
        {
            for (int i = 0; i < n; ++i)
            {
                // lower and upper halves
                if (i < n/2) {
                    suml += a[idx[i]];
                } else {
                    sumh += a[idx[i]];
                }
                sumf += a[i];
            }
        }
    }

    istat += curandDestroyGenerator(g);
    delete[] a;
    delete[] idx;

    std::cout << "Lower half, upper half, total, diff: " <<
                  std::setprecision(7) << std::setw(10) << suml <<
                  std::setprecision(7) << std::setw(10) << sumh <<
                  std::setprecision(7) << std::setw(10) << sumf <<
                  std::setprecision(4) << std::setw(12) << sumf-(suml+sumh) <<
                  std::endl;

    int ilo  = (int) suml;
    int ihi  = (int) sumh;
    int iall = (int) sumf;

    if (istat == 0) {
        if ((2*(ihi - ilo) > (n/2)-(n/10)) &&
            (2*(ihi - ilo) < (n/2)+(n/10)) &&
            (2*(ihi - ilo) >  iall-(n/10)) &&
            (2*(ihi - ilo) <  iall+(n/10)) ) {
            std::cout << " Test PASSED" << std::endl;
        } else {
            std::cout << " Test FAILED, wrong answers" << std::endl;
            istat = -1;
        }
    } else
        std::cout << " Test FAILED, istat = " << istat << std::endl;

    return istat;
}
