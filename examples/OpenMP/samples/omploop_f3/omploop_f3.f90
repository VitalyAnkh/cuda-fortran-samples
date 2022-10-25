! 
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
! 


!
! Jacobi iteration example using OpenACC Directives in Fortran
! Build with
!   nvfortran -acc -Minfo=accel -fast f3.f90
!
module sm
contains
 subroutine smooth( a, b, w0, w1, w2, n, m, niters )
  real, dimension(:,:) :: a,b
  real :: w0, w1, w2
  integer :: n, m, niters
  integer :: i, j, iter
  !$acc data copy(a(:,:)) copyin(b(:,:))
  !$omp target data map(tofrom:a(:,:)) map(to:b(:,:))
   do iter = 1,niters
    !$omp target teams loop
    do i = 2,n-1
     do j = 2,m-1
      a(i,j) = w0 * b(i,j) + &
               w1 * (b(i-1,j) + b(i,j-1) + b(i+1,j) + b(i,j+1)) + &
               w2 * (b(i-1,j-1) + b(i-1,j+1) + b(i+1,j-1) + b(i+1,j+1))
     enddo
    enddo
    !$omp target teams loop
    do i = 2,n-1
     do j = 2,m-1
      b(i,j) = w0 * a(i,j) + &
               w1 * (a(i-1,j) + a(i,j-1) + a(i+1,j) + a(i,j+1)) + &
               w2 * (a(i-1,j-1) + a(i-1,j+1) + a(i+1,j-1) + a(i+1,j+1))
     enddo
    enddo
   enddo
  !$omp end target data
 end subroutine
 subroutine smoothhost( a, b, w0, w1, w2, n, m, niters )
  real, dimension(:,:) :: a,b
  real :: w0, w1, w2
  integer :: n, m, niters
  integer :: i, j, iter
   do iter = 1,niters
    do i = 2,n-1
     do j = 2,m-1
      a(i,j) = w0 * b(i,j) + &
               w1 * (b(i-1,j) + b(i,j-1) + b(i+1,j) + b(i,j+1)) + &
               w2 * (b(i-1,j-1) + b(i-1,j+1) + b(i+1,j-1) + b(i+1,j+1))
     enddo
    enddo
    do i = 2,n-1
     do j = 2,m-1
      b(i,j) = w0 * a(i,j) + &
               w1 * (a(i-1,j) + a(i,j-1) + a(i+1,j) + a(i,j+1)) + &
               w2 * (a(i-1,j-1) + a(i-1,j+1) + a(i+1,j-1) + a(i+1,j+1))
     enddo
    enddo
   enddo
 end subroutine
end module

program main
 use sm
 use omp_lib
 real,dimension(:,:),allocatable :: aa, bb
 real,dimension(:,:),allocatable :: aahost, bbhost
 real :: w0, w1, w2
 integer :: i,j,n,m,iters
 integer :: c0, c1, c2, c3, cgpu, chost
 integer :: errs, args
 character(10) :: arg
 real :: dif, tol

 n = 0
 m = 0
 iters = 0
 args = command_argument_count()
 if( args .gt. 0 )then
   call get_command_argument( 1, arg )
   read(arg,'(i10)') n
   if( args .gt. 1 )then
    call get_command_argument( 2, arg )
    read(arg,'(i10)') m
    if( args .gt. 2 )then
     call get_command_argument( 3, arg )
     read(arg,'(i10)') iters
    endif
   endif
 endif

 if (omp_get_num_devices() .gt.0) then
   print *, 'Running on the GPU'
 else
   print *, 'Running on the Host'
 endif

 if( n .le. 0 ) n = 1000
 if( m .le. 0 ) m = n
 if( iters .le. 0 ) iters = 10

 allocate( aa(n,m) )
 allocate( bb(n,m) )
 allocate( aahost(n,m) )
 allocate( bbhost(n,m) )
 do i = 1,n
   do j = 1,m
     aa(i,j) = 0
     bb(i,j) = i*1000 + j
     aahost(i,j) = 0
     bbhost(i,j) = i*1000 + j
   enddo
 enddo
 w0 = 0.5
 w1 = 0.3
 w2 = 0.2
 call system_clock( count=c1 )
 call smooth( aa, bb, w0, w1, w2, n, m, iters )
 call system_clock( count=c2 )
 cgpu = c2 - c1
 call smoothhost( aahost, bbhost, w0, w1, w2, n, m, iters )
 call system_clock( count=c3)
 chost = c3 - c2
 ! check the results
 errs = 0
 tol = 0.000005
 do i = 1,n
  do j = 1,m
   dif = abs(aa(i,j) - aahost(i,j))
   if( aahost(i,j) .ne. 0 ) dif = abs(dif/aahost(i,j))
   if( dif .gt. tol )then
    errs = errs + 1
    if( errs .le. 10 )then
     print *, i, j, aa(i,j), aahost(i,j)
    endif
   endif
  enddo
 enddo
 print *, cgpu, ' microseconds on target device'
 print *, chost, ' microseconds on host'
 if (errs .ne. 0) then
    print *, "Test FAILED"
    print *, errs, ' errors found'
 else
    print *, "Test PASSED"
 endif

end program
