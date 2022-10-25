! 
!  Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
! 
!  NVIDIA CORPORATION and its licensors retain all intellectual property
!  and proprietary rights in and to this software, related documentation
!  and any modifications thereto.  Any use, reproduction, disclosure or
!  distribution of this software and related documentation without an express
!  license agreement from NVIDIA CORPORATION is strictly prohibited.
! 
module testUtils

integer(4) :: n, lda, ldb, info
integer(4) :: ii, jj
integer(4) :: cr, t1, t2
real   (8) :: mysecond
real   (8) :: gflops
integer(4), parameter :: m    = 7000
integer(4), parameter :: nrhs = 2

contains
   subroutine init
   implicit none
   n   = m
   lda = m
   ldb = m
   call system_clock( count_rate = cr )
   gflops = ((dble(n)*dble(n)*dble(n)*0.6666666666667d0) + (dble(n)*dble(n))) / 1000000000.0
   end subroutine init

   subroutine setInput( a, b )
   implicit none
   real(8), intent(out) :: a(m,m), b(m,nrhs)
   call random_number( a )
   do ii = 1, n
      b(ii,1) = sum( a(ii,:) )
      b(ii,2) = 2*b(ii,1)
   enddo
   end subroutine setInput

   subroutine checkOutput( x )
   implicit none
   real(8), intent(in) :: x(m,nrhs)
   real(8) :: err(2)
   do ii = 1, 2
     err(ii) = maxval( abs( x(:,ii) - ii ) )
   enddo
   print*, 'Max error LU:', maxval(err)

   if ( info .eq. 0 .and. maxval(err) .lt. 1.e-9 ) then
      print*, "Test PASSED"
   else
      print*, "Test FAILED, max error =", maxval(err)
   endif
   end subroutine checkOutput

   subroutine startTimer
   implicit none
   call system_clock(t1)
   end subroutine startTimer

   subroutine endTimer
   implicit none
   call system_clock(t2)
   print*, "GFlops:", gflops * real(cr) / ( t2 - t1 )
   end subroutine endTimer
end module testUtils

program FallbackLUTest
use testUtils
call init

! `export NV_LAMATH_ARGCHECK=1; expoert NV_LAMATH_FALLBACK=1` to enable the CPU
! fallback logic
! If one of the arrays is not accessible on the GPU, will try to perform the
! calculation on the CPU if all arrays are accessible on the CPU; a warning
! message will be printed
call LUFallback

end program FallbackLUTest

subroutine LUFallback
use testUtils
implicit none

real   (8), managed, allocatable :: a(:,:)
real   (8),          allocatable :: b(:,:)
integer(4), managed, allocatable :: ipiv(:)

! Setup input
allocate(a(m,n))
allocate(b(m,nrhs))
allocate(ipiv(m))

call setInput(a,b)

! Call LU routines
call startTimer()
call dgetrf(m,n,a,lda,ipiv,info)
call endTimer()
call dgetrs('n',n,nrhs,a,lda,ipiv,b,ldb,info)

! Check answer
call checkOutput(b)

deallocate(a)
deallocate(b)
deallocate(ipiv)

end subroutine LUFallback
