!
!  Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
!
!  NVIDIA CORPORATION and its licensors retain all intellectual property
!  and proprietary rights in and to this software, related documentation
!  and any modifications thereto.  Any use, reproduction, disclosure or
!  distribution of this software and related documentation without an express
!  license agreement from NVIDIA CORPORATION is strictly prohibited.
!
!  For an existing code that uses OpenACC to accelerate some of the computation
!  but has calls to LAPACK routines (on the CPU), the LAPACK computation can be
!  moved to the GPU by:
!  1. Adding `oacc data` or `oacc enter data` directives with appropriate data
!     movement
!  2. Adding `-gpu=nvlamath` to the compile line to access the device call
!     interface
!  3. Adding `-cudalib=nvlamath` to the link line to build the code.

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
   use cusolverdn
   implicit none
   n   = m
   lda = m
   ldb = m
   call system_clock( count_rate = cr )
   ! For computing dgetrf gflops
   gflops = ((dble(n)*dble(n)*dble(n)*0.6666666666667d0) + (dble(n)*dble(n))) / 1000000000.0
   end subroutine init

   subroutine setInput( a, b )
   implicit none
   real(8), intent(out) :: a(m,m), b(m,nrhs)
   integer(4) :: seedsize
   integer(4), allocatable :: seed(:)
   call random_seed()
   call random_seed( size = seedsize )
   allocate( seed( seedsize ) )
   seed = 123
   call random_seed( put = seed )
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
   print*, 'Max error     :', maxval(err)

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
   real(8) :: t
   call system_clock(t2)
   t = ( t2 - t1 ) / real(cr)
   print*, "Time (seconds):", t
   print*, "GFlops        :", gflops / t
   end subroutine endTimer
end module testUtils

program LUOpenACCTest
use testUtils

call init

! Warm-up and creation of cuSOLVER handle
call LUOpenACC

! Timing
call LUOpenACC

end program LUOpenACCTest

subroutine LUOpenACC
use testUtils
implicit none

real   (8), allocatable :: a(:,:), b(:,:)
integer(4), allocatable :: ipiv(:)

! Setup input
allocate(a(m,n))
allocate(b(m,nrhs))
allocate(ipiv(m))

call setInput(a,b)

! Call LU routines
!$acc data copyin( a ) copy( b ) create( ipiv )
   !$acc host_data use_device( a, ipiv, b )
   call startTimer()
   call dgetrf(m,n,a,lda,ipiv,info)
   call endTimer()
   call dgetrs('n',n,nrhs,a,lda,ipiv,b,ldb,info)
   !$acc end host_data
!$acc end data

! Check answer
call checkOutput(b)

deallocate(a)
deallocate(b)
deallocate(ipiv)

end subroutine LUOpenACC
