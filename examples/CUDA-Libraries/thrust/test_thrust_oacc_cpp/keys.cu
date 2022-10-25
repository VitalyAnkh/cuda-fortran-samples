// Filename: index.cu
// nvcc -c -arch sm_35 key.cu
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <cuda_runtime_api.h>

using namespace std;

__device__ int *a_i;
__device__ float *a_f;
__device__ double *a_d;

struct cmpi : public binary_function<int, int, bool>
{
  __device__  bool operator()(const int i, const int j) const
  {return ( a_i[i] < a_i[j]);}
};

struct cmpf : public binary_function<int, int, bool>
{
  __device__  bool operator()(const int i, const int j) const
  {return ( a_f[i] < a_f[j]);}
};

struct cmpd : public binary_function<int, int, bool>
{
  __device__  bool operator()(const int i, const int j) const
  {return ( a_d[i] < a_d[j]);}
};

// index sort for integer arrays
void thrust_int_sort_wrapper( int *dev_data, int *dev_idx, int N, cudaStream_t stream)
{
thrust::device_ptr <int> dev_ptr(dev_idx);
cudaMemcpyToSymbolAsync(a_i, &dev_data,
                        sizeof(int *), (size_t) 0,
                        cudaMemcpyHostToDevice, stream);
thrust::sequence(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N);
thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N, cmpi());
}

// index sort for float arrays
void thrust_float_sort_wrapper( float *dev_data, int *dev_idx, int N, cudaStream_t stream)
{
thrust::device_ptr <int> dev_ptr(dev_idx);
cudaMemcpyToSymbolAsync(a_f, &dev_data,
                        sizeof(float *), (size_t) 0,
                        cudaMemcpyHostToDevice, stream);
thrust::sequence(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N);
thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N, cmpf());
}

// index sort for double arrays
void thrust_double_sort_wrapper( double * dev_data, int *dev_idx, int N, cudaStream_t stream)
{
thrust::device_ptr <int> dev_ptr(dev_idx);
cudaMemcpyToSymbolAsync(a_d, &dev_data,
                        sizeof(double *), (size_t) 0,
                        cudaMemcpyHostToDevice, stream);
thrust::sequence(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N);
thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+N, cmpd());
}
