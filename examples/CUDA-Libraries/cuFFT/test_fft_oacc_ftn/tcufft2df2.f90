!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
!

program cufft2dTest
  use cufft
  use openacc
  integer, parameter :: m=768, n=512
  complex, allocatable  :: a(:,:),b(:,:),c(:,:)
  real, allocatable     :: r(:,:),q(:,:)
  integer :: iplan1, iplan2, iplan3, ierr

  allocate(a(m,n),b(m,n),c(m,n))
  allocate(r(m,n),q(m,n))

  a = 1; r = 1
  xmx = -99.0

  ierr = cufftPlan2D(iplan1,m,n,CUFFT_C2C)
  ierr = ierr + cufftSetStream(iplan1,acc_get_cuda_stream(acc_async_sync))
  !$acc host_data use_device(a,b,c)
  ierr = ierr + cufftExecC2C(iplan1,a,b,CUFFT_FORWARD)
  ierr = ierr + cufftExecC2C(iplan1,b,c,CUFFT_INVERSE)
  !$acc end host_data

  ! scale c
  !$acc kernels
  c = c / (m*n)
  !$acc end kernels

  ! Check forward answer
  write(*,*) 'Max error C2C FWD: ', cmplx(maxval(real(b)) - sum(real(b)), &
                                          maxval(imag(b)))
  ! Check inverse answer
  write(*,*) 'Max error C2C INV: ', maxval(abs(a-c))

  ! Real transform
  ierr = ierr + cufftPlan2D(iplan2,m,n,CUFFT_R2C)
  ierr = ierr + cufftPlan2D(iplan3,m,n,CUFFT_C2R)
  ierr = ierr + cufftSetStream(iplan2,acc_get_cuda_stream(acc_async_sync))
  ierr = ierr + cufftSetStream(iplan3,acc_get_cuda_stream(acc_async_sync))

  !$acc host_data use_device(r,b,q)
  ierr = ierr + cufftExecR2C(iplan2,r,b)
  ierr = ierr + cufftExecC2R(iplan3,b,q)
  !$acc end host_data

  !$acc kernels
  xmx = maxval(abs(r-q/(m*n)))
  !$acc end kernels

  ! Check R2C + C2R answer
  write(*,*) 'Max error R2C/C2R: ', xmx

  ierr = ierr + cufftDestroy(iplan1)
  ierr = ierr + cufftDestroy(iplan2)
  ierr = ierr + cufftDestroy(iplan3)

  if (ierr.eq.0) then
    print *,"test PASSED"
  else
    print *,"test FAILED"
  endif

end program cufft2dTest
