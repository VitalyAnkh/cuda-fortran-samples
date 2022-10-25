#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

double second()
{
#ifdef _WIN32
	double PCFreq = 0.0;

	LARGE_INTEGER li;
	if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed!\n");

	PCFreq = (double)li.QuadPart;

	QueryPerformanceCounter(&li);

	return (double)(li.QuadPart)/PCFreq;
#else
	struct timeval time;
	struct timezone tz;

	gettimeofday(&time, &tz);
	return( (double) time.tv_sec + ( (double) time.tv_usec ) / 1000000 );
#endif
}
