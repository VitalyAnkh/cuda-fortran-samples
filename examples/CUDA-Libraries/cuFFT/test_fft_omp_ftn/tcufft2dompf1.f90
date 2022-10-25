!
!     Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
!

program cufft2dompTest
  use cufft
  use omp_lib

  implicit none
  integer, parameter :: m=768, n=512
  complex :: a(m,n),b(m,n),c(m,n)
  real    :: r(m,n),q(m,n)
  integer :: iplan1, iplan2, iplan3, ierr
  real, dimension(4) :: res

  a = 1; r = 1

  ierr = cufftPlan2D(iplan1,m,n,CUFFT_C2C)
  ierr = ierr + cufftSetStream(iplan1,ompx_get_cuda_stream(omp_get_default_device(), .false.))

  !$omp target enter data map( to: a( 1:m, 1:n ) ) map( alloc: b( 1:m, 1:n ), c( 1:m, 1:n ) )
  !$omp target data use_device_ptr( a, b, c )
  ierr = ierr + cufftExecC2C(iplan1,a,b,CUFFT_FORWARD)
  ierr = ierr + cufftExecC2C(iplan1,b,c,CUFFT_INVERSE)
  !$omp end target data
  !$omp target exit data map( from:b, c ) map( delete: a )

  res(1) = maxval( real(b) ) - sum( real(b) )
  res(2) = maxval( imag(b) )
  res(3) = maxval( abs( a - c / (m*n) ) )

  ! Check forward answer
  write(*,*) 'Max error C2C FWD: ', cmplx( res(1), res(2) ) 

  ! Check inverse answer
  write(*,*) 'Max error C2C INV: ', res(3)

  ! Real transform
  ierr = ierr + cufftPlan2D(iplan2,m,n,CUFFT_R2C)
  ierr = ierr + cufftPlan2D(iplan3,m,n,CUFFT_C2R)
  ierr = ierr + cufftSetStream(iplan2,ompx_get_cuda_stream(omp_get_default_device(), .false.))
  ierr = ierr + cufftSetStream(iplan3,ompx_get_cuda_stream(omp_get_default_device(), .false.))

  !$omp target enter data map( to: r( 1:m, 1:n ) ) map( alloc: b( 1:m, 1:n ), q( 1:m, 1:n ) )
  !$omp target data use_device_ptr( r, b, q )
  ierr = ierr + cufftExecR2C(iplan2,r,b)
  ierr = ierr + cufftExecC2R(iplan3,b,q)
  !$omp end target data
  !$omp target exit data map( from:q ) map( delete: r, b )

  res(4) = maxval( abs( r - q / (m*n) ) )

  ! Check R2C + C2R answer
  write(*,*) 'Max error R2C/C2R: ', res(4)

  ierr = ierr + cufftDestroy(iplan1)
  ierr = ierr + cufftDestroy(iplan2)
  ierr = ierr + cufftDestroy(iplan3)

  if (ierr.eq.0 .and. maxval( res ) .lt. 1.e-6) then
    print *,"Test PASSED"
  else
    print *,"Test FAILED, max error =", maxval( res )
  endif

end program cufft2dompTest
