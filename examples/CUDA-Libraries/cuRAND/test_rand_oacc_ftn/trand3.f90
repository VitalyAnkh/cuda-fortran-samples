program testcurand1
call foo1(1000)
call foo2(1000)
call foo3(1000)
end
!
subroutine foo1(n)
use curand
integer :: a(n)
type(curandGenerator) :: g
integer(8) nbits
logical passing
a = 0
passing = .true.
istat = curandCreateGeneratorHost(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandGenerate(g, a, n)
istat = curandDestroyGenerator(g)
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
subroutine foo2(n)
use curand
real :: a(n)
type(curandGenerator) :: g
logical passing
a = 0.0
passing = .true.
istat = curandCreateGeneratorHost(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandGenerate(g, a, n)
istat = curandDestroyGenerator(g)
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
subroutine foo3(n)
use curand
real(8) :: a(n)
type(curandGenerator) :: g
logical passing
a = 0.0d0
passing = .true.
istat = curandCreateGeneratorHost(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandGenerate(g, a, n)
istat = curandDestroyGenerator(g)
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
