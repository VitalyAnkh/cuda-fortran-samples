program sparseMatVec
    integer n
    n = 25 ! # rows/cols in dense matrix
    call sparseMatVecSub1(n)
    n = 45 ! # rows/cols in dense matrix
    call sparseMatVecSub1(n)
end program

subroutine sparseMatVecSub1(n)
  use openacc
  use cusparse

  implicit none

  integer n

  ! dense data
  real(4), allocatable :: Ade(:,:), x(:), y(:)
  
  ! sparse CSR arrays
  real(4), allocatable :: csrValA(:) 
  integer, allocatable :: nnzPerRowA(:), csrRowPtrA(:), csrColIndA(:)

  allocate(Ade(n,n), x(n), y(n))
  allocate(csrValA(n))
  allocate(nnzPerRowA(n), csrRowPtrA(n+1), csrColIndA(n))

  call sparseMatVecSub2(Ade, x, y, csrValA, nnzPerRowA, csrRowPtrA, &
                                                         csrColIndA, n)
  deallocate(Ade)
  deallocate(x)
  deallocate(y)
  deallocate(csrValA)
  deallocate(nnzPerRowA)
  deallocate(csrRowPtrA)
  deallocate(csrColIndA)
end subroutine

subroutine sparseMatVecSub2(Ade, x, y, csrValA, nnzPerRowA, csrRowPtrA, &
                                                             csrColIndA, n)
  use openacc
  use cusparse

  implicit none

  ! dense data
  real(4) :: Ade(n,n), x(n), y(n)
  
  ! sparse CSR arrays
  real(4) :: csrValA(n) 
  integer :: nnzPerRowA(n), csrRowPtrA(n+1), csrColIndA(n)

  integer :: n, nnz, status, i
  type(cusparseHandle) :: h
  type(cusparseMatDescr) :: descrA
#if CUDA_VERSION >= 11000
  integer(8) :: bsize
  type(c_devptr) :: bdptr
  type(c_ptr) :: bcptr
  integer(1), pointer :: buffer(:)
  integer(1) :: crayp(*); pointer(pbuf,crayp)
  type(cusparseSpMatDescr) :: matA
  type(cusparseDnVecDescr) :: vecX, vecY
#endif

  ! parameters
  real(4) :: alpha, beta

  ! result
  real(4) :: xerr
  
  ! initalize CUSPARSE and matrix descriptor  
  status = cusparseCreate(h)
  if (status /= CUSPARSE_STATUS_SUCCESS) &
       write(*,*) 'cusparseCreate error: ', status
  status = cusparseCreateMatDescr(descrA)
  status = cusparseSetMatType(descrA, &
       CUSPARSE_MATRIX_TYPE_GENERAL)
  status = cusparseSetMatIndexBase(descrA, &
       CUSPARSE_INDEX_BASE_ONE)
  status = cusparseSetStream(h, acc_get_cuda_stream(acc_async_sync))
  
  !$acc data create(Ade, x, y, csrValA, nnzPerRowA, csrRowPtrA, csrColIndA)

  ! Initialize matrix (upper circular shift matrix)
  !$acc kernels
  Ade = 0.0  
  do i = 1, n-1
     Ade(i,i+1) = 1.0          
  end do
  Ade(n,1) = 1.0

  ! Initialize vectors and constants
  do i = 1, n
     x(i) = i
  enddo
  y = 0.0
  !$acc end kernels

  !$acc update host(x)
  write(*,*) 'Original vector:'
  write(*,'(5(1x,f7.2))') x

  ! convert matrix from dense to csr format
  !$acc host_data use_device(Ade, nnzPerRowA, csrValA, csrRowPtrA, csrColIndA)
  status = cusparseSnnz_v2(h, CUSPARSE_DIRECTION_ROW, &
       n, n, descrA, Ade, n, nnzPerRowA, nnz) 
  status = cusparseSdense2csr(h, n, n, descrA, Ade, n, &
       nnzPerRowA, csrValA, csrRowPtrA, csrColIndA)
  !$acc end host_data

  ! A is upper circular shift matrix
  ! y = alpha*A*x + beta*y
  alpha = 1.0
  beta = 0.0
  !$acc host_data use_device(csrValA, csrRowPtrA, csrColIndA, x, y)
#if CUDA_VERSION <= 10020
  status = cusparseScsrmv(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &
       n, n, n, alpha, descrA, csrValA, csrRowPtrA, &
       csrColIndA, x, beta, y)
#else
  status = cusparseCreateDnVec(vecX, n, x, CUDA_R_32F)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseCreateDnVec: ",status

  status = cusparseCreateDnVec(vecY, n, y, CUDA_R_32F)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseCreateDnVec: ",status

  status = cusparseCreateCsr(matA, n, n, nnz, csrRowPtrA, csrColIndA, &
             csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, &
             CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseCreateCsr: ",status

  status = cusparseSpMV_buffersize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, &
           matA, vecX, beta, vecY, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, bsize)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseSpMV_buffersize: ",status

  print *,"SpMV buffersize required: ",bsize
  if (bsize.gt.0) then
    bdptr = acc_malloc(bsize)
    ! One way without CUF is move the c_depvtr to a c_ptr, then call c_f_pointer
    bcptr = transfer(bdptr, bcptr)
    call c_f_pointer(bcptr, buffer, [ bsize ])
  else
    nullify(buffer)
  end if

  status = cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, &
           matA, vecX, beta, vecY, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, buffer)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseSpMV: ",status
  if (bsize.gt.0) call acc_free(bdptr)
#endif
  !$acc end host_data

  !$acc wait
  write(*,*) 'Shifted vector:'
  write(*,'(5(1x,f7.2))') y

  ! shift-down y and add original x
  ! A' is lower circular shift matrix
  ! x = alpha*A'*y + beta*x
  beta = -1.0
  !$acc host_data use_device(csrValA, csrRowPtrA, csrColIndA, x, y)
#if CUDA_VERSION <= 10020
  status = cusparseScsrmv(h, CUSPARSE_OPERATION_TRANSPOSE, &
       n, n, n, alpha, descrA, csrValA, csrRowPtrA, &
       csrColIndA, y, beta, x)
#else
  status = cusparseSpMV_buffersize(h, CUSPARSE_OPERATION_TRANSPOSE, alpha, &
           matA, vecY, beta, vecX, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, bsize)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseSpMV_buffersize: ",status

  print *,"SpMV buffersize required: ",bsize
  if (bsize.gt.0) then
    bdptr = acc_malloc(bsize)
    ! One way without CUF is to use cray pointers
    pbuf = transfer(bdptr, pbuf)
  end if

  status = cusparseSpMV(h, CUSPARSE_OPERATION_TRANSPOSE, alpha, &
           matA, vecY, beta, vecX, CUDA_R_32F, CUSPARSE_CSRMV_ALG1, crayp)
  if (status.ne.CUSPARSE_STATUS_SUCCESS) print *,"cusparseSpMV: ",status
  if (bsize.gt.0) call acc_free(bdptr)
#endif
  !$acc end host_data

  !$acc kernels
  xerr = maxval(abs(x))
  !$acc end kernels
  !$acc end data

  write(*,*) 'Max error = ', xerr
  if (xerr.le.1.e-5) then
    write(*,*) 'Test PASSED'
  else
    write(*,*) 'Test FAILED'
  endif

end subroutine
