program testcublas2
call foo1(1000)
call foo2(1000)
call foo3(1000)
call foo4(1000)
end
!
subroutine foo1(n)
use cublas
integer :: a(n), b(n)
a = 1
b = 2
!$acc data copy(a, b)
call sswap(n, a, 1, b, 1)
call cublasSswap(n, a, 1, b, 1)
!$acc end data
if (all(a.eq.1).and.all(b.eq.2)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
!
subroutine foo2(n)
use cublas
integer(8) :: a(n), b(n)
a = 1
b = 2
!$acc data copy(a, b)
call dswap(n, a, 1, b, 1)
call cublasDswap(n, a, 1, b, 1)
!$acc end data
if (all(a.eq.1).and.all(b.eq.2)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
!
subroutine foo3(n)
use cublas
real :: a(n), b(n)
a = 1.
b = 2.
!$acc data copy(a, b)
call sswap(n, a, 1, b, 1)
call cublasSswap(n, a, 1, b, 1)
!$acc end data
if (all(a.eq.1.).and.all(b.eq.2.)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
!
subroutine foo4(n)
use cublas
real(8) :: a(n), b(n)
a = 1.0d0
b = 2.0d0
!$acc data copy(a, b)
call dswap(n, a, 1, b, 1)
call cublasDswap(n, a, 1, b, 1)
!$acc end data
if (all(a.eq.1.0d0).and.all(b.eq.2.0d0)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
end
