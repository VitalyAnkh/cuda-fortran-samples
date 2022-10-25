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
! sAXPY example using Do Concurrent construct in Fortran
! Build with
!   nvfortran -stdpar -Minfo -fast saxpy.f90
! Build with to target Multicore
!   nvfortran -stdpar=multicore -Minfo=accel -fast saxpy.f90
!

module sm
    contains
    subroutine saxpy_concurrent(x,y,n,a)
        real,dimension(:) :: x, y
        real :: a
        integer :: n, i  
        do concurrent (i = 1: n)
          y(i) = a*x(i)+y(i)
        enddo  
    end subroutine 

    subroutine saxpy_do(x,y,n,a)
        real,dimension(:) :: x, y
        real :: a
        integer :: n, i  
        do i = 1, n
          y(i) = a*x(i)+y(i)
        enddo  
    end subroutine 
end module

program main
    use sm
    real,dimension(:),allocatable :: x, x2, y
    real :: a = 2.0
    integer :: n, i, err = 0
    integer :: c0, c1, c2, cpar, cseq
    n = 1000000
    allocate(x2(n), x(n), y(n))

    do i = 1, n
       x(i)  = 1
       x2(i) = 1
       y(i)  = i
    enddo

    call system_clock( count=c0 )
    call saxpy_do(x2, y, n, a)
    call system_clock( count=c1 )
    call saxpy_concurrent(x, y, n, a)
    call system_clock( count=c2 )
    cseq = c1 - c0
    cpar = c2 - c1

    do i = 1, n
      if(x(i) .ne. x2(i)) then
          err = err + 1
      endif
    enddo

    print *, cseq, ' microseconds sequential'
    print *, cpar, ' microseconds parallel with stdpar'
    if(err .eq. 0) then
      print *, "Test PASSED"
    else
      print *, "Test FAILED"
    endif

end program
