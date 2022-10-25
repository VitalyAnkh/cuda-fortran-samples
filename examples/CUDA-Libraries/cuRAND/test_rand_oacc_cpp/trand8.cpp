#include "stdlib.h"
#include "openacc_curand.h"

void
t1a(float *restrict a, float *restrict b, int n)
{
  unsigned long long seed;
  unsigned long long seq;
  unsigned long long offset;
  curandState_t state;
  #pragma acc parallel num_gangs(1) copy(a[0:n],b[0:n]) private(state)
  {
    seed = 12345ULL;
    seq = 0ULL;
    offset = 0ULL;
    curand_init(seed, seq, offset, &state);
    #pragma acc loop seq
    for (int i = 0; i < n; i++) {
      a[i] = curand_uniform(&state);
      b[i] = curand_normal(&state);
    }
  }
}
  
#include "malloc.h"

int main()
{
  int n = 1000;
  float *a, *b;
  int i, nc1, nc2;
  double rmean, sumd;
  int passing = 1;
  a = (float *) malloc(n*4);
  b = (float *) malloc(n*4);
  for (int i = 0; i < n; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }
  t1a(a, b, n);

  printf("Should be uniform around 0.5\n");
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
  return 0;
}
