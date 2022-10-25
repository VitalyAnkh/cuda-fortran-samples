!
!  Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
!
!  NVIDIA CORPORATION and its licensors retain all intellectual property
!  and proprietary rights in and to this software, related documentation
!  and any modifications thereto.  Any use, reproduction, disclosure or
!  distribution of this software and related documentation without an express
!  license agreement from NVIDIA CORPORATION is strictly prohibited.
!
!  For an existing code that uses OpenMP target offload to accelerate some of
!  the computation but has calls to LAPACK routines (on the CPU), the LAPACK
!  computation can be moved to the GPU by:
!  1. Adding `omp target enter data` directive with appropriate data movement
!  2. Adding `-gpu=nvlamath` to the compile line to access the device call
!     interface
!  3. Adding `-cudalib=nvlamath` to the link line to build the code.

module testUtils

integer(4) :: n, lda, ldb, info
integer(4) :: ii, jj
integer(4) :: cr, t1, t2
integer(4) , parameter :: m    = 2000
character*1, parameter :: jobz = 'V'
character*1, parameter :: uplo = 'U'

contains
   subroutine init
   use cusolverdn
   implicit none
   n   = m
   lda = m
   ldb = m
   call system_clock( count_rate = cr )
   end subroutine init

   subroutine setInput( a, b )
   implicit none
   real(8), intent(out) :: a(m,m), b(m)
   real(8)    :: work( 2*m )
   integer(4) :: i, imax, info, seedsize
   integer(4), allocatable :: seed(:)
   call random_seed()
   call random_seed( size = seedsize )
   allocate( seed( seedsize ) )
   seed = 123
   call random_seed( put = seed )
   call random_number( b )
   a = 0.d0
   do i = 1, m
      imax    = maxloc( b, dim = 1 )
      a(i,i)  = b(imax)
      b(imax) = -100.d0
   enddo
   do i = 1, m
      b(i) = a(i,i)
   enddo
   ! Create test matrix with given eigen-values
   call dlarge( m, a, m, seed, work, info )
   end subroutine setInput

   subroutine checkOutput( b, w )
   implicit none
   real(8), intent(in)    :: b(m)
   real(8), intent(inout) :: w(m)
   integer(4) :: i, imax
   real   (8) :: maxerr
   w = abs(w)
   maxerr = 0.d0
   do i = 1, m
      imax = maxloc( w, dim = 1 )
      maxerr = max( maxerr, abs( b(i) - w(imax) ) )
      w(imax) = -100.d0
   enddo
   print*, 'Max error     :', maxerr

   if ( info .eq. 0 .and. maxerr .lt. 1.e-12 ) then
      print*, "Test PASSED"
   else
      print*, "Test FAILED, max error =", maxerr
   endif
   end subroutine checkOutput

   subroutine startTimer
   implicit none
   call system_clock(t1)
   end subroutine startTimer

   subroutine endTimer
   implicit none
   call system_clock(t2)
   print*, "Time (seconds):", ( t2 - t1 ) / real(cr)
   end subroutine endTimer
end module testUtils

program OpenMPEigenTest
use testUtils

call init

! Warm-up and creation of cuSOLVER handle
call EigenOpenMP

! Timing
call EigenOpenMP

end program OpenMPEigenTest

subroutine EigenOpenMP
use testUtils
implicit none

real   (8), allocatable :: a(:,:), b(:), w(:), work(:)
integer(4), allocatable :: iwork(:)
integer(4) :: i, j, lwork, liwork

! Setup input
allocate(a(m,n))
allocate(b(m))
allocate(w(m))
allocate(work(1))
allocate(iwork(1))

call setInput(a,b)

! Get buffer size
!$omp target enter data map( to: a ) map( alloc: w, work, iwork )
!$omp target data use_device_ptr( a, w, work, iwork )
   call dsyevd( jobz, uplo, n, a, n, w, work, -1, iwork, -1, info )
!$omp end target data
!$omp target exit data map( from: work, iwork )

lwork  = int(  work( 1 ) )
liwork = int( iwork( 1 ) )
deallocate(work)
deallocate(iwork)
allocate(  work(  lwork ) )
allocate( iwork( liwork ) )

! Call solver
call startTimer()
!$omp target enter data map( alloc: work, iwork )
!$omp target data use_device_ptr( a, w, work, iwork )
   call dsyevd( jobz, uplo, n, a, n, w, work, lwork, iwork, liwork, info )
!$omp end target data
!$omp target exit data map( from: a, w ) map( delete: work, iwork )
call endTimer()

! Check answer
call checkOutput(b,w)

deallocate(a)
deallocate(b)
deallocate(w)
deallocate(work)
deallocate(iwork)

end subroutine EigenOpenMP
