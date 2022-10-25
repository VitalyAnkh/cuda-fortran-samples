#include "stdlib.h"
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
   istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
   istat = curandGenerate(g, a, n);
   istat = curandDestroyGenerator(g);
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
   if (istat==0 && passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}

void curandTestOmpFloat(int n) {
   float *a;
   int i, istat, nc1, nc2;
   curandGenerator_t g;
   int passing = 1;
   double rmean, sumd;
   a = (float *) malloc(n*4);
   for (i = 0; i < n; i++)
     a[i] = 0.0f;
   istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);

   /* Uniform */
   printf("Should be uniform around 0.5\n");
   istat = curandGenerateUniform(g, a, n);
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
   istat = curandGenerateNormal(g, a, n, 0.0f, 1.0f);
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
   istat = curandDestroyGenerator(g);

   if (istat==0 && passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}

void curandTestOmpDouble(int n) {
   double *a;
   int i, istat, nc1, nc2;
   curandGenerator_t g;
   int passing = 1;
   double rmean, sumd;
   a = (double *) malloc(n*8);
   for (i = 0; i < n; i++)
     a[i] = 0.0;
   istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);

   /* Uniform */
   printf("Should be uniform around 0.5\n");
   istat = curandGenerateUniformDouble(g, a, n);
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
   istat = curandGenerateNormalDouble(g, a, n, 0.0, 1.0);
   sumd = 0.0; nc1 = nc2 = 0;
   for (i = 0; i < n; i++) {
     if (i<10) printf("%d %lf\n",i,a[i]);
     if ((a[i] > -4.0) && (a[i] < 0.0)) {
	nc1++;
	sumd += a[i];
     } else if ((a[i] > 0.0) && (a[i] < 4.0)) {
	nc2++;
	sumd += a[i];
     } 
   }
   printf("Found on each side of zero %d %d\n",nc1,nc2);
   if (abs(nc1-nc2) > (n/10)) passing = 0;
   rmean = sumd / (double) n;
   if ((rmean < -0.1) || (rmean > 0.1))
     passing = 0;
   else
     printf("Mean found to be %lf which is passing\n",rmean);
   istat = curandDestroyGenerator(g);

   if (istat==0 && passing) 
      printf(" Test PASSED\n");
   else
      printf(" Test FAILED\n");
}
