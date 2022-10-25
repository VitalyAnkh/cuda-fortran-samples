program testcublasxt
call foo1(1000)
call foo2(1000)
call foo3(1000)
call foo4(1000)
end
!
subroutine foo1(n)
use cublasxt
real(4) :: a(n,n), b(n,n), c(n,n), alpha, beta
type(cublasXtHandle) :: h
integer ndevices(1)
a = 1.0
b = 2.0
c = -1.0
alpha = 1.0
beta = 0.0
istat = cublasXtCreate(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
ndevices(1) = 0
istat = cublasXtDeviceSelect(h, 1, ndevices)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, &
                      n, n, n, &
                      alpha, A, n, B, n, beta, C, n)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtDestroy(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
if (all(c.eq.2.0*n)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
print *,c(1,1),c(n,n)
end
!
subroutine foo2(n)
use cublasxt
real(8) :: a(n,n), b(n,n), c(n,n), alpha, beta
type(cublasXtHandle) :: h
integer ndevices(1)
a = 1.0d0
b = 2.0d0
c = -1.0d0
alpha = 1.0d0
beta = 0.0
istat = cublasXtCreate(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
ndevices(1) = 0
istat = cublasXtDeviceSelect(h, 1, ndevices)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, &
                      n, n, n, &
                      alpha, A, n, B, n, beta, C, n)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtDestroy(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
if (all(c.eq.2.0d0*n)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
print *,c(1,1),c(n,n)
end
!
subroutine foo3(n)
use cublasXt
complex*8 :: a(n,n), b(n,n), c(n,n), alpha, beta
type(cublasXtHandle) :: h
integer ndevices(1)
a = cmplx(1.,0.)
b = cmplx(2.,0.)
c = cmplx(-1.0,0.)
alpha = cmplx(1.0d0,0.0)
beta = cmplx(0.0,0.0)
istat = cublasXtCreate(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
ndevices(1) = 0
istat = cublasXtDeviceSelect(h, 1, ndevices)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtCgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, &
                      n, n, n, &
                      alpha, A, n, B, n, beta, C, n)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtDestroy(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
if (all(real(c).eq.2.0*n)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
print *,c(1,1),c(n,n)
end
!
subroutine foo4(n)
use cublasXt
complex*16 :: a(n,n), b(n,n), c(n,n), alpha, beta
type(cublasXtHandle) :: h
integer ndevices(1)
a = cmplx(1.0d0,0.0d0)
b = cmplx(2.0d0,0.0d0)
c = cmplx(-1.0d0,0.0d0)
alpha = cmplx(1.0d0,0.0d0)
beta = cmplx(0.0d0,0.0d0)
istat = cublasXtCreate(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
ndevices(1) = 0
istat = cublasXtDeviceSelect(h, 1, ndevices)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, &
                      n, n, n, &
                      alpha, A, n, B, n, beta, C, n)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
istat = cublasXtDestroy(h)
if (istat .ne. CUBLAS_STATUS_SUCCESS) print *,istat
if (all(dble(c).eq.2.0d0*n)) then
  print *,"Test PASSED"
else
  print *,"Test FAILED"
endif
print *,c(1,1),c(n,n)
end
