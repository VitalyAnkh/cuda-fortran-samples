program testcublas2
call cublasTestOmpInteger (1000)
call cublasTestOmpInteger8(1000)
call cublasTestOmpReal    (1000)
call cublasTestOmpDouble  (1000)
end
!
subroutine cublasTestOmpInteger(n)
use cublas_v2
use omp_lib
integer :: a(n), b(n)
integer :: ierr
type( cublasHandle ) :: handle

a = 1
b = 2
ierr = cublasCreate( handle )
ierr = ierr + cublasSetStream(handle,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target enter data map( to: a(1:n), b(1:n) )
!$omp target data use_device_ptr( a, b )
call sswap(n, a, 1, b, 1)
ierr = cublasSswap(handle, n, a, 1, b, 1)
!$omp end target data
!$omp target exit data map( from: a, b )
if (all(a.eq.1).and.all(b.eq.2)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
ierr = cublasDestroy( handle )
end
!
subroutine cublasTestOmpInteger8(n)
use cublas_v2
use omp_lib
integer(8) :: a(n), b(n)
integer :: ierr
type( cublasHandle ) :: handle

a = 1
b = 2
ierr = cublasCreate( handle )
ierr = ierr + cublasSetStream(handle,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target enter data map( to: a(1:n), b(1:n) )
!$omp target data use_device_ptr( a, b )
call dswap(n, a, 1, b, 1)
ierr = cublasDswap(handle, n, a, 1, b, 1)
!$omp end target data
!$omp target exit data map( from: a, b )
if (all(a.eq.1).and.all(b.eq.2)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
ierr = cublasDestroy( handle )
end
!
subroutine cublasTestOmpReal(n)
use cublas_v2
use omp_lib
real :: a(n), b(n)
integer :: ierr
type( cublasHandle ) :: handle

a = 1.
b = 2.
ierr = cublasCreate( handle )
ierr = ierr + cublasSetStream(handle,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target enter data map( to: a(1:n), b(1:n) )
!$omp target data use_device_ptr( a, b )
call sswap(n, a, 1, b, 1)
ierr = cublasSswap(handle, n, a, 1, b, 1)
!$omp end target data
!$omp target exit data map( from: a, b )
if (all(a.eq.1.).and.all(b.eq.2.)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
ierr = cublasDestroy( handle )
end
!
subroutine cublasTestOmpDouble(n)
use cublas_v2
use omp_lib
real(8) :: a(n), b(n)
integer :: ierr

type( cublasHandle ) :: handle
a = 1.0d0
b = 2.0d0
ierr = cublasCreate( handle )
ierr = ierr + cublasSetStream(handle,ompx_get_cuda_stream(omp_get_default_device(), .false.))
!$omp target enter data map( to: a(1:n), b(1:n) )
!$omp target data use_device_ptr( a, b )
call dswap(n, a, 1, b, 1)
ierr = cublasDswap(handle, n, a, 1, b, 1)
!$omp end target data
!$omp target exit data map( from: a, b )
if (all(a.eq.1.0d0).and.all(b.eq.2.0d0)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
ierr = cublasDestroy( handle )
end
