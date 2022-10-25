
!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
!

! An example of how to call the cublas single precision matrix multiply
! routine cublasSgemm
!
! Build for running on the host:
!   nvfortran -o sgemm_host cublasSgemm.F90 -lblas
!
! Build for running on the gpu:
!   nvfortran -cuda -o sgemm_gpu cublasSgemm.F90 -cudalib=cublas
!


program test_cublasSgemm
#ifdef _CUDA
use cudafor
use cublas
real, device, allocatable, dimension(:,:) :: A, B, C
#else
real, allocatable, dimension(:,:) :: A, B, C
#endif
real, allocatable :: T(:,:), Expct(:)

real :: alpha = 1.0e0
real :: beta  = 0.0e0
real :: t1, t2, tt, gflops
integer :: i, j, k, nerrors

! print *, "Enter N: "
! read(5,*) n

n = 1000

allocate(A(n,n), B(n,n), C(n,n), T(n,n), Expct(n))

call random_number(T)
do j = 1, n
  Expct(j) = 2.0 * sum(T(:,j))
end do
    
A = 2.0
B = T
C = -9.9

call cpu_time(t1)

call sgemm('n','n', n, n, n, alpha, A, n, B, n, beta, C, n)

#ifdef _CUDA
istat = cudaDeviceSynchronize()  ! Only needed for a fair time
#endif

call cpu_time(t2)

print *, "Checking results...."

T = C

nerrors = 0
do j = 1, n
  do i = 1, n
    if (abs(T(i,j) - Expct(j)) .gt. 5.0e-3) then
       nerrors = nerrors + 1
    endif
  enddo
enddo

if (nerrors .ne. 0) then
   print *, "Test FAILED"
else
   print *, "Test PASSED"
endif

gflops = (real(n) * real(n) * real(n) * 2.0) / 1000000000.0
tt = t2 - t1
print *, "Total Time: ",tt
print *, "Total SGEMM gflops: ",gflops/tt

end
