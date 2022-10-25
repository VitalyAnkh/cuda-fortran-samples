// Filename: csort.cu
// nvcc -c -arch sm_35 csort.cu
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

extern "C" {

//Sort for integer arrays
void thrust_int_sort_wrapper( int *data, int N)
{
thrust::device_ptr <int> dev_ptr(data);
thrust::sort(dev_ptr, dev_ptr+N);
}

//Sort for float arrays
void thrust_float_sort_wrapper( float *data, int N)
{
thrust::device_ptr <float> dev_ptr(data);
thrust::sort(dev_ptr, dev_ptr+N);
}

//Sort for double arrays
void thrust_double_sort_wrapper( double *data, int N)
{
thrust::device_ptr <double> dev_ptr(data);
thrust::sort(dev_ptr, dev_ptr+N);
}
}
