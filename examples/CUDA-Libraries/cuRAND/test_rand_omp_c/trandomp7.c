#include "stdlib.h"
#include "omp.h"
#include "malloc.h"
#include "curand.h"

#ifdef __WIN64
  int __blt_pgi_popcount(unsigned int x) {
    int y;
    for (y=0; x!=0; x=x>>1) y += (x&1);
    return y;
  }
#endif

void curandTestOmpInt   (int);
void curandTestOmpFloat (int);
void curandTestOmpDouble(int);

int main()
{
   curandTestOmpInt   (5000);
   curandTestOmpFloat (5000);
   curandTestOmpDouble(5000);
   return 0;
}

void curandTestOmpInt(int n) {
   unsigned int *a;
   int i, istat;
   unsigned long nbits;
   curandGenerator_t g;
   int passing = 1;
   a = (unsigned int *) malloc(n*4);
   for (i = 0; i < n; i++)
     a[i] = 0;

   #pragma omp target enter data map( to:a[0:n] )
     istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
     istat = curandSetStream(g, (cudaStream_t) ompx_get_cuda_stream(omp_get_default_device(), 0));
     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);

     #pragma omp target data use_device_ptr( a )
     {
       istat = curandGenerate(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }

     istat = curandDestroyGenerator(g);
     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
   #pragma omp target exit data map( from:a[0:n] )

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

void curandTestOmpFloat(int n) {
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
   istat = curandSetStream(g,(cudaStream_t)ompx_get_cuda_stream(omp_get_default_device(), 0));
   if (istat != CURAND_STATUS_SUCCESS) printf("Error from set stream\n");

   /* Uniform */
   printf("Should be uniform around 0.5\n");

   #pragma omp target enter data map( to:a[0:n], b[0:n] )
     #pragma omp target data use_device_ptr( a )
     {
       istat = curandGenerateUniform(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }

     #pragma omp target update from ( a[0:n] )
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

     #pragma omp target data use_device_ptr( b )
     {
       istat = curandGenerateNormal(g, b, n, 0.0f, 1.0f);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }

     istat = curandDestroyGenerator(g);
   #pragma omp target exit data map( from:b[0:n] )

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

   if (passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}

void curandTestOmpDouble(int n) {
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
   istat = curandSetStream(g,(cudaStream_t)ompx_get_cuda_stream(omp_get_default_device(), 0));
   if (istat != CURAND_STATUS_SUCCESS) printf("Error from set stream\n");

   /* Uniform */
   printf("Should be uniform around 0.5\n");

   #pragma omp target enter data map( to: a[0:n] )
     #pragma omp target data use_device_ptr( a )
     {
       istat = curandGenerateUniformDouble(g, a, n);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
   #pragma omp target exit data map( from: a[0:n] )

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

   #pragma omp target enter data map( to:b[0:n] )
     #pragma omp target data use_device_ptr( b )
     {
       istat = curandGenerateNormalDouble(g, b, n, 0.0, 1.0);
       if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
     }
   #pragma omp target exit data map( from:b[0:n] )

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
