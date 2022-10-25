
!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
!

program test_matmul
real*8, dimension(2000,2000) :: a, b, c, c_expd
integer*4 :: n = 2000, m = 2000, l = 2000
real*8 :: t1, t2, t3, t4
integer*4 :: i, j, niters, nerrors

call random_number(a)
call random_number(b)
c = 0.0d0
c_expd = 0.0d0
nerrors = 0
niters = 1   ! was 10000

call cpu_time(t1)
do i = 1, niters
  call matmul_s(a,b,c_expd,n,m,l)
enddo
call cpu_time(t2)

call cpu_time(t3)
do i = 1, niters
  call matmul_p(a,b,c,n,m,l)
enddo
call cpu_time(t4)

do j = 1, m
  do i = 1, n
    if (abs(c(i,j) - c_expd(i,j)) .gt. 0.00000000001) then
       nerrors = nerrors + 1
    endif
  enddo
enddo

print *, "Serial Time(seconds): ",t2-t1
print *, "Parallel Time(seconds): ",t4-t3

if (nerrors .ne. 0) then
  print *, "Test FAILED"
else
  print *, "Test PASSED"
endif
end



subroutine matmul_p(a,b,c,n,m,l)
integer*4 :: n, m, l
real*8, dimension(n,l) :: a
real*8, dimension(l,m) :: b
real*8, dimension(n,m) :: c

integer*4 :: i, j, k

do i = 1, n
  do j = 1, m
    do k = 1, l
       c(i,j) = c(i,j) + (a(i,k) * b(k,m))
    enddo
  enddo
enddo

end

!pgi$r noconcur
subroutine matmul_s(a,b,c,n,m,l)
integer*4 :: n, m, l
real*8, dimension(n,l) :: a
real*8, dimension(l,m) :: b
real*8, dimension(n,m) :: c

integer*4 :: i, j, k

do i = 1, n
  do j = 1, m
    do k = 1, l
       c(i,j) = c(i,j) + (a(i,k) * b(k,m))
    enddo
  enddo
enddo

end
