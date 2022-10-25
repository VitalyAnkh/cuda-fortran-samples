
!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
! 

      program matmul_time
c
#if defined (PGF90)
      include 'lib3f.h'
#else
      real etime
#endif
      integer i, j, k
      integer size, ntimes, m, n, p
      parameter (size=1000)
      parameter (m=size,n=size,p=size)
      parameter (ntimes=5)
      real*8 a, b, c, arow
      dimension a(m,n), b(n,p), c(n,p), arow(n)
!hpf$ distribute (*,block) :: a, b
!hpf$ align c(:,:) with b(:,:) 
c
      integer l
      real walltime, mflops
      integer hz, clock0, clock1, clock2
      integer t, nerrors
      allocatable t(:)
      real tarray(2), time0, time1, time2
!hpf$ distribute t(cyclic)
#if defined (HPF)
      allocate (t(number_of_processors()))
#else
      allocate (t(1))
#endif
#if defined (SYSCLK)
      call system_clock(count_rate=hz)
#endif
c
      do i = 1, m
      do j = 1, n
         a(i,j) = 1.0
      enddo
      enddo
      do i = 1, n
      do j = 1, p
         b(i,j) = 1.0
      enddo
      enddo
      do j = 1, p
      do i = 1, m
         c(i,j) = 0.0
      enddo
      enddo
c
#if defined (SYSCLK)
      call system_clock(count=clock0)
#else
      time0 = etime(tarray)
#endif
      do l = 1, ntimes
#if defined MATMUL
         c = matmul(a, b)
#elif defined BLAS
         call dgemm('no transpose','no transpose',m,n,p,
     &              1.0d0, a, m, b, n, 1.0d0, c, m)
#else
!$omp parallel 
         do i = 1, m
!$omp do
            do ii = 1, n
               arow(ii) = a(i,ii)
            enddo
!hpf$ independent
!$omp do
            do j = 1, p
               do k = 1, n
                  c(i,j) = c(i,j) + arow(k) * b(k,j)
               enddo
            enddo
         enddo
!$omp end parallel
#endif
         call dummy(c)
         call cpu_time(t2)
      enddo
#if defined (SYSCLK)
      call system_clock(count=clock1)
#else
      time1 = etime(tarray)
#endif
      do i = 1, ntimes
         call dummy(c)
      enddo
#if defined (SYSCLK)
      call system_clock(count=clock2)
#else
      time2 = etime(tarray)
#endif
c
#if defined (SYSCLK)
      t = (clock1 - clock0) - (clock2 - clock1)
      walltime = (real(sum(t)) / real(hz)) / real(ntimes)
      mflops = (m*p*(2*n-1)) / (walltime * 1.0e+06)
      print *, walltime, clock0, clock1, clock2
#else
      walltime = ((time1 - time0) - (time2 - time1)) / real(ntimes)
      mflops = (m*p*(2*n-1)) / (walltime * 1.0e+06)
      print *, walltime, time0, time1, time2
#endif
c
      print *, "M =",M,", N =",N,", P =",P
      print *, "MFLOPS = ", mflops

      nerrors = 0
      do j = 1, p
         do i = 1, m
            if (c(i,j) .ne. (dble(size)*dble(ntimes))) then
               nerrors = nerrors + 1
            endif
         enddo
      enddo
      if (nerrors .ne. 0) then
         print *, "Test FAILED"
      else
         print *, "Test PASSED"
      endif
c
      end
c
      subroutine dummy(c)
      return
      end
