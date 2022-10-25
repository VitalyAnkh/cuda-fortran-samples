module mtests
  integer, parameter :: n = 1000
  contains
    subroutine testany( a, b )
    use openacc_curand
    real :: a(n), b(n)
    type(curandStateXORWOW) :: h
    integer(8) :: seed, seq, offset

    !$acc parallel num_gangs(1) vector_length(1) copy(a,b) private(h)
    seed = 12345
    seq = 0
    offset = 0
    call curand_init(seed, seq, offset, h)
    !$acc loop seq
    do i = 1, n
      a(i) = curand_uniform(h)
      b(i) = curand_normal(h)
    end do
    !$acc end parallel
    return
    end subroutine
end module mtests

program t
use mtests
real :: a(n), b(n), c(n)
logical passing
a = 1.0
b = 2.0
passing = .true.
call testany(a,b)
c = a
print *,"Should be uniform around 0.5"
do i = 1, n
  if (i.lt.10) print *,i,c(i)
  if ((c(i).lt.0.0) .or. (c(i).gt.1.0)) passing = .false.
end do
rmean = sum(c)/n
if ((rmean .lt. 0.4) .or. (rmean .gt. 0.6)) then
  passing = .false.
else
  print *,"Mean is ",rmean," which passes"
endif
c = b
print *,"Should be normal around 0.0"
nc1 = 0;
nc2 = 0;
do i = 1, n
  if (i.lt.10) print *,i,c(i)
  if ((c(i) .gt. -4.0) .and. (c(i) .lt. 0.0)) nc1 = nc1 + 1
  if ((c(i) .gt.  0.0) .and. (c(i) .lt. 4.0)) nc2 = nc2 + 1
end do
print *,"Found on each side of zero ",nc1,nc2
if (abs(nc1-nc2) .gt. (n/10)) npassing = .false.
rmean = sum(c,mask=abs(c).lt.4.0)/n
if ((rmean .lt. -0.1) .or. (rmean .gt. 0.1)) then
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
