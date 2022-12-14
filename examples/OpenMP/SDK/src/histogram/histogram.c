/*
 *     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* 
 * histogram
 * ----------
 * This sample implements 256-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

#include "cuda2mp.h"
#include "timer.h"

const int numRuns = 16;

double histogram256CPU(uint *hist, uchar *data, uint count)
{
    uint i;
    StartTimer();
    for(int i = 0; i < 256; i++)
        hist[i] = 0;
    for(i = 0; i < count; i++)
        hist[data[i]]++;
    return GetTimer();
}

double histogram256GPU(uint *hist, uint *data, uint count)
{
    // assuming (count % 4 == 0 && count > 0)
    count >>= 2;
    #pragma omp target enter data map(alloc:hist[0:256]) map(to:data[0:count])
    StartTimer();
    #pragma omp target teams loop
    for (int i = 0; i < 256; i++) {
        uint tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
        #pragma omp loop reduction(+:tmp0) reduction(+:tmp1) reduction(+:tmp2) reduction(+:tmp3)
        for(int j = 0; j < count; j++) {
            tmp0 += ((data[j] & 0xFFU) == i ? 1 : 0);
            tmp1 += ((data[j] >> 8 & 0xFFU) == i ? 1 : 0);
            tmp2 += ((data[j] >> 16 & 0xFFU) == i ? 1 : 0);
            tmp3 += ((data[j] >> 24 & 0xFFU) == i ? 1 : 0);
        }
        hist[i] = tmp0 + tmp1 + tmp2 + tmp3;
    }
    double gettimer = GetTimer();
    #pragma omp target exit data map(from:hist[0:256]) map(delete:data[0:count])
    return gettimer;
}

// run test
void runtest(int n, float th)
{
    double tu_gpu = 0.0, tu_cpu = 0.0;
    unsigned int time(void *);

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    uchar *data = (uchar *)malloc(sizeof(uchar) * n);

    printf("...generating input data\n");
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        data[i] = rand() % 256;
    uint *hist_cpu = (uint *)malloc(sizeof(uint) * 256);
    uint *hist_gpu = (uint *)malloc(sizeof(uint) * 256);

    uint byteCount = n * sizeof(uchar);
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
    for (int iter = 0; iter < numRuns; iter++) {
        tu_gpu += histogram256GPU(hist_gpu, (uint*)data, n);
    }

    double dAvgSecs = tu_gpu/numRuns * 1.0e-3;

    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);

    printf("\nValidating GPU results...\n");
    printf(" ...histogram256CPU()\n");
    tu_cpu = histogram256CPU(hist_cpu, data, n);
    dAvgSecs = tu_cpu * 1.0e-3;
    printf("histogram256CPU() time : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf(" ...comparing the results...\n");
    printf("%s\n", (icheck((int *)hist_gpu, (int *)hist_cpu, 256, th) ? "Test FAILED" : "Test PASSED"));

    printf("Shutting down 256-bin histogram...\n\n\n");
    free(hist_cpu);
    free(hist_gpu);
    free(data);
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    uint n = 256*1048576;
    float th = 0.0;

    char *names[] = { "n", "thresh" };
    int flags[] = { 1, 1 };
    int map[] = { 0, 1 };
    struct OptionTable *opttable = make_opttable(2, (const char **) names, flags, map);
    argproc(argc, argv, opttable);

    printf("[%s] - Starting...\n", argv[0]);
    print_gpuinfo(argc, (const char**)argv);

    const char *str_n = opttable->table[0].val, *str_th = opttable->table[1].val;
    if (str_n)
        n = atoi(str_n);
    if (str_th)
        th = atof(str_th);
    runtest(n, th);

    free_opttable(opttable);
    return 0;
}
