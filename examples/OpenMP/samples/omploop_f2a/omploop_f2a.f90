! 
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
! 

program main
    use omp_lib
    implicit none
    integer :: n        ! size of the vector
    real,dimension(:),allocatable :: a  ! the vector
    real,dimension(:),allocatable :: r  ! the results
    real,dimension(:),allocatable :: e  ! expected results
    integer :: i, j, nerrors
    integer :: c0, c1, c2, c3, cgpu, chost
    character(10) :: arg1
    if( command_argument_count() .gt. 0 )then
	call get_command_argument( 1, arg1 )
	read(arg1,'(i10)') n
    else
	n = 1000000
    endif
    if( n .le. 0 ) n = 1000000
    allocate(a(n))
    allocate(r(n))
    allocate(e(n))
    do i = 1,n
	a(i) = i*2.0
    enddo
do j = 1, 3
    call system_clock( count=c1 )
    !$omp target teams loop map(from:r(:n))
	do i = 1,n
	    r(i) = sin(a(i)) ** 2 + cos(a(i)) ** 2
	enddo
    call system_clock( count=c2 )
    cgpu = c2 - c1
	do i = 1,n
	    e(i) = sin(a(i)) ** 2 + cos(a(i)) ** 2
	enddo
    call system_clock( count=c3 )
    chost = c3 - c2
    ! check the results
    nerrors = 0
    do i = 1,n
	if( abs(r(i) - e(i)) .gt. 0.000001 )then
	    print *, i, r(i), e(i)
            nerrors = nerrors + 1
	endif
    enddo
    print *, n, ' iterations completed'
    print *, cgpu, ' microseconds on GPU'
    print *, chost, ' microseconds on host'
    if (nerrors .ne. 0) then
       print *, "Test FAILED"
    else
       print *, "Test PASSED"
    endif
enddo
end program
