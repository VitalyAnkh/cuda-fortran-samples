
!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
! 

      program verify

      character*10 h1, h2, h3, h4
      integer n1, n2, n3, i, nerrors
      real*8 r1, r2, r1_expd, r2_expd

      open (unit=8,file='../expd_results')
      open (unit=9,file='fft.out')

      read(8,*) h1, h2, h3, h4
      read(8,*) h1, h2, n1, n2, n3
      read(9,*) h1, h2, h3, h4
      read(9,*) h1, h2, n1, n2, n3

      nerrors = 0
      do i = 1, 6
         read(8,*) h1, h2, n1, h3, h4, r1_expd, r2_expd
         read(9,*) h1, h2, n1, h3, h4, r1, r2
         if ( abs(r1_expd - r1) .gt. 1.0d-11  ) then
            nerrors = nerrors + 1
         endif    
         if ( abs(r2_expd - r2) .gt. 1.0d-11  ) then
            nerrors = nerrors + 1
         endif    
      enddo

      if (nerrors .ne. 0) then
         print *, "Test FAILED"
      else
         print *, "Test PASSED"
      endif

      close (unit=8)
      close (unit=9)

      end
