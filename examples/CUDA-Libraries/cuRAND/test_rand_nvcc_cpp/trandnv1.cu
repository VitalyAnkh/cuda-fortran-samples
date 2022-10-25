#include "cuda_runtime_api.h"
#include "curand.h"
#include "stdio.h"

int main()
{
   float *a;
   float *a_d;
   int istat, n;
   int i, nc1, nc2;
   double rmean, sumd;
   int passing = 1;
   n = 1234;
   curandGenerator_t g;
   a = (float *) malloc(n*4);
   /* Initialize it to zero, to be sure */
   for (int i = 0; i < n; i++) {
     a[i] = 0.0f;
   }
   cudaMalloc((void**)(&a_d), n*4);
   istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
   if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
   istat = curandGenerateUniform(g, a_d, n);
   if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
   cudaMemcpy(a, a_d, n*4, cudaMemcpyDeviceToHost);

   printf("Should be normal around 0.0\n");
   sumd = 0.0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %f\n",i,a[i]);
     if ((a[i] < 0.0f) || (a[i] > 1.0f)) passing = 0;
     sumd += (double) a[i];
   }
   rmean = sumd / (double) n;
   if ((rmean < 0.4) || (rmean > 0.6))
      passing = 0;
   else
      printf("mean found is %lf, which is passing\n",rmean);

   /* Now Normal */
   istat = curandGenerateNormal(g, a_d, n, 0.0f, 1.0f);
   if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
   cudaMemcpy(a, a_d, n*4, cudaMemcpyDeviceToHost);

   istat = curandDestroyGenerator(g);
   if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);

   printf("Should be normal around 0.0\n");
   sumd = 0.0; nc1 = nc2 = 0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %f\n",i,a[i]);
     if ((a[i] > -4.0f) && (a[i] < 0.0f)) {
        nc1++;
        sumd += (double) a[i];
     } else if ((a[i] > 0.0f) && (a[i] < 4.0f)) {
        nc2++;
        sumd += (double) a[i];
     }
   }
   printf("Found on each side of zero %d %d\n",nc1,nc2);
   if (abs(nc1-nc2) > (n/10)) passing = 0;
   rmean = sumd / (double) n;
   if ((rmean < -0.1f) || (rmean > 0.1f))
     passing = 0;
   else
     printf("Mean found to be %lf which is passing\n",rmean);

   if (passing)
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}
