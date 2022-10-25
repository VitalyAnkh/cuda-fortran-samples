program testcurandOmp4
call testcurandOmpInteger(1000)
call testcurandOmpReal   (1000)
call testcurandOmpDouble (1000)
end
!
subroutine testcurandOmpInteger(n)
use curand
use omp_lib
integer :: a(n)
type(curandGenerator) :: g
integer(8) nbits
logical passing
a = 0
passing = .true.
!$omp target enter data map( to:a( 1:n ) )
istat = curandCreateGenerator(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandSetStream(g,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target data use_device_ptr( a )
istat = curandGenerate(g, a, n)
!$omp end target data
istat = curandDestroyGenerator(g)
!$omp target exit data map( from:a )
nbits = 0
do i = 1, n
  if (i.lt.10) print *,i,a(i)
  nbits = nbits + popcnt(a(i))
end do
print *,"Should be roughly half the bits set"
nbits = nbits / n
if ((nbits .lt. 12) .or. (nbits .gt. 20)) then
  passing = .false.
else
  print *,"nbits is ",nbits," which passes"
endif
if (passing) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
!
subroutine testcurandOmpReal(n)
use curand
use omp_lib
real :: a(n)
type(curandGenerator) :: g
logical passing
a = 0.0
passing = .true.
!$omp target enter data map( to:a(1:n) )
istat = curandCreateGenerator(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandSetStream(g,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target data use_device_ptr( a )
istat = curandGenerate(g, a, n)
!$omp end target data
istat = curandDestroyGenerator(g)
!$omp target exit data map( from:a )
print *,"Should be uniform around 0.5"
do i = 1, n
  if (i.lt.10) print *,i,a(i)
  if ((a(i).lt.0.0) .or. (a(i).gt.1.0)) passing = .false.
end do
rmean = sum(a)/n
if ((rmean .lt. 0.4) .or. (rmean .gt. 0.6)) then
  passing = .false.
else
  print *,"Mean is ",rmean," which passes"
endif
if (passing) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
!
subroutine testcurandOmpDouble(n)
use curand
use omp_lib
real(8) :: a(n)
type(curandGenerator) :: g
logical passing
a = 0.0d0
passing = .true.
!$omp target enter data map( to:a( 1:n ) )
istat = curandCreateGenerator(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandSetStream(g,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target data use_device_ptr( a )
istat = curandGenerate(g, a, n)
!$omp end target data
istat = curandDestroyGenerator(g)
!$omp target exit data map( from: a )
do i = 1, n
  if (i.lt.10) print *,i,a(i)
  if ((a(i).lt.0.0d0) .or. (a(i).gt.1.0d0)) passing = .false.
end do
rmean = sum(a)/n
if ((rmean .lt. 0.4d0) .or. (rmean .gt. 0.6d0)) then
  passing = .false.
else
  print *,"Mean is ",rmean," which passes"
endif
if (passing) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
