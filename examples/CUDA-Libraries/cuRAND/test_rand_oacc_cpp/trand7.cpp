#include "stdlib.h"
#include "openacc.h"
#include "malloc.h"
#include "curand.h"

extern "C" int __blt_pgi_popcount(unsigned int);

void foo1(int);
void foo2(int);
void foo3(int);

int main()
{
   foo1(5000);
   foo2(5000);
   foo3(5000);
   return 0;
}

void foo1(int n) {
   unsigned int *a;
   int i, istat;
   unsigned long nbits;
   curandGenerator_t g;
   int passing = 1;
   a = (unsigned int *) malloc(n*4);
   for (i = 0; i < n; i++)
     a[i] = 0;
#pragma acc data copy(a[0:n])
   {
     istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
#pragma acc host_data use_device(a)
     {
       istat = curandGenerate(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
     istat = curandDestroyGenerator(g);
     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
   }
   nbits = 0;
   printf("Should be roughly half the bits set\n");
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %10u\n",i,a[i]);
     nbits += __blt_pgi_popcount(a[i]);
   }
   nbits = nbits / n;
   if ((nbits < 12) || (nbits > 20)) 
      passing = 0;
   else
      printf("nbits is %lu which passes\n",nbits);
   if (passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}

void foo2(int n) {
   float *a, *b;
   int i, istat, nc1, nc2;
   curandGenerator_t g;
   int passing = 1;
   double rmean, sumd;
   a = (float *) malloc(n*4);
   b = (float *) malloc(n*4);
   for (i = 0; i < n; i++) {
     a[i] = 0.0f;
     b[i] = 1.0f;
   }
   istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
   istat = curandSetStream(g,(cudaStream_t)acc_get_cuda_stream(acc_async_sync));
   if (istat != CURAND_STATUS_SUCCESS) printf("Error from set stream\n");

   /* Uniform */
   printf("Should be uniform around 0.5\n");
#pragma acc data copy(a[0:n], b[0:n])
   {
#pragma acc host_data use_device(a)
     {
       istat = curandGenerateUniform(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
#pragma acc update host(a[0:n])
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
   printf("Should be normal around 0.0\n");
#pragma acc host_data use_device(b)
     {
       istat = curandGenerateNormal(g, b, n, 0.0f, 1.0f);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }

   }  /* end data */
   sumd = 0.0; nc1 = nc2 = 0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %f\n",i,b[i]);
     if ((b[i] > -4.0f) && (b[i] < 0.0f)) {
	nc1++;
	sumd += (double) b[i];
     } else if ((b[i] > 0.0f) && (b[i] < 4.0f)) {
	nc2++;
	sumd += (double) b[i];
     } 
   }

   printf("Found on each side of zero %d %d\n",nc1,nc2);
   if (abs(nc1-nc2) > (n/10)) passing = 0;
   rmean = sumd / (double) n;
   if ((rmean < -0.1f) || (rmean > 0.1f))
     passing = 0;
   else
     printf("Mean found to be %lf which is passing\n",rmean);
   istat = curandDestroyGenerator(g);

   if (passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}

void foo3(int n) {
   double *a, *b;
   int i, istat, nc1, nc2;
   curandGenerator_t g;
   int passing = 1;
   double rmean, sumd;
   a = (double *) malloc(n*8);
   b = (double *) malloc(n*8);
   for (i = 0; i < n; i++) {
     a[i] = 0.0;
     b[i] = 1.0;
   }
   istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
   istat = curandSetStream(g,(cudaStream_t)acc_get_cuda_stream(acc_async_sync));
   if (istat != CURAND_STATUS_SUCCESS) printf("Error from set stream\n");

   /* Uniform */
   printf("Should be uniform around 0.5\n");
#pragma acc data copy(a[0:n])
   {
#pragma acc host_data use_device(a)
     {
       istat = curandGenerateUniformDouble(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
   }
   sumd = 0.0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %lf\n",i,a[i]);
     if ((a[i] < 0.0) || (a[i] > 1.0)) passing = 0;
     sumd += a[i];
   }
   rmean = sumd / (double) n;
   if ((rmean < 0.4) || (rmean > 0.6)) 
      passing = 0;
   else
      printf("mean found is %lf, which is passing\n",rmean);

   /* Now Normal */
   printf("Should be normal around 0.0\n");
#pragma acc data copy(b[0:n])
   {
#pragma acc host_data use_device(b)
     {
       istat = curandGenerateNormalDouble(g, b, n, 0.0, 1.0);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
   }  /* end data */

   istat = curandDestroyGenerator(g);

   sumd = 0.0; nc1 = nc2 = 0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %lf\n",i,b[i]);
     if ((b[i] > -4.0) && (b[i] < 0.0)) {
	nc1++;
	sumd += b[i];
     } else if ((b[i] > 0.0) && (b[i] < 4.0)) {
	nc2++;
	sumd += b[i];
     } 
   }
   printf("Found on each side of zero %d %d\n",nc1,nc2);
   if (abs(nc1-nc2) > (n/10)) passing = 0;
   rmean = sumd / (double) n;
   if ((rmean < -0.1) || (rmean > 0.1))
     passing = 0;
   else
     printf("Mean found to be %lf which is passing\n",rmean);

   if (passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}
