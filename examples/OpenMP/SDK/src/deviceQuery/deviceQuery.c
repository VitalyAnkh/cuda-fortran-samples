/*
 *     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuda2mp.h"

// query number of different type devices
void deviceQueryMp()
{
    int num_nvidia = omp_get_num_devices();
    printf("nvidia devices: %d\n", num_nvidia);
    printf("\n");

    // for cuda platforms
    //acc_set_device_type(acc_device_nvidia);
    float arr[1024];
    finit_rand(arr, 1024);
    #pragma omp target teams loop map(tofrom: arr[0:1024])
    for (int i = 0; i < 1024; i++) {
        arr[i] = i;
    }

    //printf("cuda device is: %p\n", acc_get_current_cuda_device());
    //printf("cuda context is: %p\n", acc_get_current_cuda_context());
    //printf("cuda stream is: %p\n", acc_get_cuda_stream(0));
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    print_gpuinfo(argc, (const char **)argv);
    deviceQueryMp();
    return 0;
}
