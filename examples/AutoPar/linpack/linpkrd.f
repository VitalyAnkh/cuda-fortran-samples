	program linpack
	common // aa(200,200),a(201,200),b(200),x(200)
	doubleprecision aa,a,b,x
      doubleprecision time(8,6),cray,ops,total,norma,normx
      doubleprecision resid,residn,eps,epslon
      doubleprecision t1,tm2
      integer ipvt(200)
      interface second
      doubleprecision function second() bind (C,name='second')
#ifdef _WIN32
      !dec$ attributes stdcall, decorate :: second
#endif
      end function
      end interface

c	call fpmode(128)
	lda = 201
      ldaa = 200
c
      n = 100
      cray = .056
      ops = (2.0e0*n**3)/3.0e0 + 2.0e0*n**2
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call sgefa(a,lda,n,ipvt,info)
         time(1,1) = second() - t1
         t1 = second()
         call sgesl(a,lda,n,ipvt,b,0)
         time(1,2) = second() - t1
         total = time(1,1) + time(1,2)
C
C     COMPUTE A RESIDUAL TO VERIFY RESULTS.
C
         do 10 i = 1,n
            x(i) = b(i)
   10    continue
         call matgen(a,lda,n,b,norma)
         do 20 i = 1,n
            b(i) = -b(i)
   20    continue
         CALL SMXPY(n,b,n,lda,x,a)
         RESID = 0.0
         NORMX = 0.0
         DO 30 I = 1,N
            RESID = max( RESID, ABS(b(i)) )
            NORMX = max( NORMX, ABS(X(I)) )
   30    CONTINUE
         eps = epslon(1.0d0)
	RESIDn = RESID/( N*NORMA*NORMX*EPS )
         write(6,40)
   40    format('     norm. resid      resid           machep',
     $          '         x(1)-1        x(n)-1')
         write(6,50) residn,resid,eps,x(1)-1,x(n)-1
   50    format(1p5e16.8)
c
         write(6,60) n
   60    format(//'    times are reported for matrices of order ',i5)
         write(6,70)
   70    format(6x,'sgefa',6x,'sgesl',6x,'total',5x,'Kflops',7x,'unit',
     $         6x,'ratio')
c
         time(1,3) = total
         time(1,4) = ops/(1.0e3*total)
         time(1,5) = 2.0e3/time(1,4)
         time(1,6) = total/cray
         write(6,80) lda
   80    format(' times for array with leading dimension of',i4)
         write(6,110) (time(1,i),i=1,6)
c	goto 998
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call sgefa(a,lda,n,ipvt,info)
         time(2,1) = second() - t1
         t1 = second()
         call sgesl(a,lda,n,ipvt,b,0)
         time(2,2) = second() - t1
         total = time(2,1) + time(2,2)
         time(2,3) = total
         time(2,4) = ops/(1.0e3*total)
         time(2,5) = 2.0e3/time(2,4)
         time(2,6) = total/cray
         write(6,110) (time(2,i),i=1,6)
c
         call matgen(a,lda,n,b,norma)
         t1 = second()
         call sgefa(a,lda,n,ipvt,info)
         time(3,1) = second() - t1
         t1 = second()
         call sgesl(a,lda,n,ipvt,b,0)
         time(3,2) = second() - t1
         total = time(3,1) + time(3,2)
         time(3,3) = total
         time(3,4) = ops/(1.0e3*total)
         time(3,5) = 2.0e3/time(3,4)
         time(3,6) = total/cray
         write(6,110) (time(3,i),i=1,6)
c
         ntimes = 500
         t1 = second()
         do 89 i = 1,ntimes
            call matgen(a,lda,n,b,norma)
 89	continue
	tm2 = second() - t1
	t1 = second()
	do 90 i = 1,ntimes
            call matgen(a,lda,n,b,norma)
            call sgefa(a,lda,n,ipvt,info)
            call sgesl(a,lda,n,ipvt,b,0)
   90    continue
         time(4,1) = (second() - t1 - tm2)/ntimes
         time(4,2) = 0
         total = time(4,1) + time(4,2)
         time(4,3) = total
         time(4,4) = ops/(1.0e3*total)
         time(4,5) = 2.0e3/time(4,4)
         time(4,6) = total/cray
c
         write(6,110) (time(4,i),i=1,6)
  110    format(3(f11.5),f11.0,2(f11.5))
 998	continue
c
         write(6,140) ldaa
  140    format(/' times for array with leading dimension of',i4)
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call sgefa(aa,ldaa,n,ipvt,info)
         time(5,1) = second() - t1
         t1 = second()
         call sgesl(aa,ldaa,n,ipvt,b,0)
         time(5,2) = second() - t1
         total = time(5,1) + time(5,2)
         time(5,3) = total
         time(5,4) = ops/(1.0e3*total)
         time(5,5) = 2.0e3/time(5,4)
         time(5,6) = total/cray
         write(6,110) (time(5,i),i=1,6)
c	goto 999
c
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call sgefa(aa,ldaa,n,ipvt,info)
         time(6,1) = second() - t1
         t1 = second()
         call sgesl(aa,ldaa,n,ipvt,b,0)
         time(6,2) = second() - t1
         total = time(6,1) + time(6,2)
         time(6,3) = total
         time(6,4) = ops/(1.0e3*total)
         time(6,5) = 2.0e3/time(6,4)
         time(6,6) = total/cray
         write(6,110) (time(6,i),i=1,6)
c
         call matgen(aa,ldaa,n,b,norma)
         t1 = second()
         call sgefa(aa,ldaa,n,ipvt,info)
         time(7,1) = second() - t1
         t1 = second()
         call sgesl(aa,ldaa,n,ipvt,b,0)
         time(7,2) = second() - t1
         total = time(7,1) + time(7,2)
         time(7,3) = total
         time(7,4) = ops/(1.0e3*total)
         time(7,5) = 2.0e3/time(7,4)
         time(7,6) = total/cray
         write(6,110) (time(7,i),i=1,6)
c
         ntimes = 500
         t1 = second()
         do 119 i = 1, ntimes
            call matgen(aa,ldaa,n,b,norma)
 119	continue
            tm2 = second() - t1
         t1 = second()
		do 120 i = 1,ntimes
            call matgen(aa,ldaa,n,b,norma)
            call sgefa(aa,ldaa,n,ipvt,info)
            call sgesl(aa,ldaa,n,ipvt,b,0)
  120    continue
         time(8,1) = (second() - t1 - tm2)/ntimes
         time(8,2) = 0
         total = time(8,1) + time(8,2)
         time(8,3) = total
         time(8,4) = ops/(1.0e3*total)
         time(8,5) = 2.0e3/time(8,4)
         time(8,6) = total/cray
c
         write(6,110) (time(8,i),i=1,6)
 999	continue
	print *
	print *,'ROLLED',' DOUBLE ',' PRECISION LINPACK PERFORMANCE ',
     $		nint(min(time(4,4),time(8,4))), ' KFLOPS '

         if (resid .gt. 1.0d-13) then
            print *, "Test FAILED"
         else
            print *, "Test PASSED"
         endif
	 stop
      end
      subroutine matgen(a,lda,n,b,norma)
      doubleprecision a(lda,1),b(1),norma
c
      init = 1325
      norma = 0.0
      do 30 j = 1,n
         do 20 i = 1,n
            init = mod(3125*init,65536)
            a(i,j) = (init - 32768.0)/16384.0
	    norma = max(a(i,j), norma)
   20    continue
   30 continue
      do 35 i = 1,n
          b(i) = 0.0
   35 continue
      do 50 j = 1,n
         do 40 i = 1,n
            b(i) = b(i) + a(i,j)
   40    continue
   50 continue
      return
      end
      subroutine sgefa(a,lda,n,ipvt,info)
      integer lda,n,ipvt(1),info
      doubleprecision a(lda,1)
c
c     sgefa factors a real matrix by gaussian elimination.
c
c     sgefa is usually called by dgeco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for dgeco) = (1 + 9/n)*(time for sgefa) .
c
c     on entry
c
c        a       real(lda, n)
c                the matrix to be factored.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix and the multipliers
c                which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if  u(k,k) .eq. 0.0 .  this is not an error
c                     condition for this subroutine, but it does
c                     indicate that sgesl or dgedi will divide by zero
c                     if called.  use  rcond  in dgeco for a reliable
c                     indication of singularity.
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas saxpy,sscal,isamax
c
c     internal variables
c
      doubleprecision t
      integer isamax,j,k,kp1,l,nm1
c
c
c     gaussian elimination with partial pivoting
c
      info = 0
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 k = 1, nm1
         kp1 = k + 1
c
c        find l = pivot index
c
         l = isamax(n-k+1,a(k,k),1) + k - 1
         ipvt(k) = l
c
c        zero pivot implies this column already triangularized
c
         if (a(l,k) .eq. 0.0d0) go to 40
c
c           interchange if necessary
c
            if (l .eq. k) go to 10
               t = a(l,k)
               a(l,k) = a(k,k)
               a(k,k) = t
   10       continue
c
c           compute multipliers
c
            t = -1.0d0/a(k,k)
            call sscal(n-k,t,a(k+1,k),1)
c
c           row elimination with column indexing
c
            do 30 j = kp1, n
               t = a(l,j)
               if (l .eq. k) go to 20
                  a(l,j) = a(k,j)
                  a(k,j) = t
   20          continue
               call saxpy(n-k,t,a(k+1,k),1,a(k+1,j),1)
   30       continue
         go to 50
   40    continue
            info = k
   50    continue
   60 continue
   70 continue
      ipvt(n) = n
      if (a(n,n) .eq. 0.0d0) info = n
      return
      end
      subroutine sgesl(a,lda,n,ipvt,b,job)
      integer lda,n,ipvt(1),job
      doubleprecision a(lda,1),b(1)
c
c     sgesl solves the real system
c     a * x = b  or  trans(a) * x = b
c     using the factors computed by dgeco or sgefa.
c
c     on entry
c
c        a       real(lda, n)
c                the output from dgeco or sgefa.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c        ipvt    integer(n)
c                the pivot vector from dgeco or sgefa.
c
c        b       real(n)
c                the right hand side vector.
c
c        job     integer
c                = 0         to solve  a*x = b ,
c                = nonzero   to solve  trans(a)*x = b  where
c                            trans(a)  is the transpose.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains a
c        zero on the diagonal.  technically this indicates singularity
c        but it is often caused by improper arguments or improper
c        setting of lda .  it will not occur if the subroutines are
c        called correctly and if dgeco has set rcond .gt. 0.0
c        or sgefa has set info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dgeco(a,lda,n,ipvt,rcond,z)
c           if (rcond is too small) go to ...
c           do 10 j = 1, p
c              call sgesl(a,lda,n,ipvt,c(1,j),0)
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas saxpy,sdot
c
c     internal variables
c
      doubleprecision sdot,t
      integer k,kb,l,nm1
c
      nm1 = n - 1
      if (job .ne. 0) go to 50
c
c        job = 0 , solve  a * x = b
c        first solve  l*y = b
c
         if (nm1 .lt. 1) go to 30
         do 20 k = 1, nm1
            l = ipvt(k)
            t = b(l)
            if (l .eq. k) go to 10
               b(l) = b(k)
               b(k) = t
   10       continue
            call saxpy(n-k,t,a(k+1,k),1,b(k+1),1)
   20    continue
   30    continue
c
c        now solve  u*x = y
c
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/a(k,k)
            t = -b(k)
            call saxpy(k-1,t,a(1,k),1,b(1),1)
   40    continue
      go to 100
   50 continue
c
c        job = nonzero, solve  trans(a) * x = b
c        first solve  trans(u)*y = b
c
         do 60 k = 1, n
            t = sdot(k-1,a(1,k),1,b(1),1)
            b(k) = (b(k) - t)/a(k,k)
   60    continue
c
c        now solve trans(l)*x = y
c
         if (nm1 .lt. 1) go to 90
         do 80 kb = 1, nm1
            k = n - kb
            b(k) = b(k) + sdot(n-k,a(k+1,k),1,b(k+1),1)
            l = ipvt(k)
            if (l .eq. k) go to 70
               t = b(l)
               b(l) = b(k)
               b(k) = t
   70       continue
   80    continue
   90    continue
  100 continue
      return
      end
      
      subroutine saxpy(n,da,dx,incx,dy,incy)
      doubleprecision dx(1),dy(1),da
      integer i,incx,incy,ix,iy,m,mp1,n
      if(n.le.0)return
      if (da .eq. 0.0d0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return

 20	continue
      do 30 i = 1,n
        dy(i) = dy(i) + da*dx(i)
   30 continue


      return
      end

	doubleprecision function sdot(n,dx,incx,dy,incy)
c
c     forms the dot product of two vectors.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      doubleprecision dx(1),dy(1),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      sdot = 0.0d0
      dtemp = 0.0d0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      sdot = dtemp
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c

 20	continue
      do 30 i = 1,n
        dtemp = dtemp + dx(i)*dy(i)
   30 continue


   60 sdot = dtemp
      return
      end
      subroutine  sscal(n,da,dx,incx)
c
c     scales a vector by a constant.
c     uses unrolled loops for increment equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      doubleprecision da,dx(1)
      integer i,incx,m,mp1,n,nincx
c
      if(n.le.0)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      nincx = n*incx
      do 10 i = 1,nincx,incx
        dx(i) = da*dx(i)
   10 continue
      return
c
c        code for increment equal to 1
c
c
c        clean-up loop
c

 20	continue
      do 30 i = 1,n
        dx(i) = da*dx(i)
   30 continue


      return
      end
      integer function isamax(n,dx,incx)
c
c     finds the index of element having max. absolute value.
c     jack dongarra, linpack, 3/11/78.
c
      doubleprecision dx(1),dmax
      integer i,incx,ix,n
c
      isamax = 0
      if( n .lt. 1 ) return
      isamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      dmax = abs(dx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(abs(dx(ix)).le.dmax) go to 5
         isamax = i
         dmax = abs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 dmax = abs(dx(1))
      do 30 i = 2,n
         if(abs(dx(i)).le.dmax) go to 30
         isamax = i
         dmax = abs(dx(i))
   30 continue
      return
      end
      doubleprecision FUNCTION EPSLON (X)
      doubleprecision X
C
C     ESTIMATE UNIT ROUNDOFF IN QUANTITIES OF SIZE X.
C
      doubleprecision A,B,C,EPS
C
C     THIS PROGRAM SHOULD FUNCTION PROPERLY ON ALL SYSTEMS
C     SATISFYING THE FOLLOWING TWO ASSUMPTIONS,
C        1.  THE BASE USED IN REPRESENTING FLOATING POINT
C            NUMBERS IS NOT A POWER OF THREE.
C        2.  THE QUANTITY  A  IN STATEMENT 10 IS REPRESENTED TO 
C            THE ACCURACY USED IN FLOATING POINT VARIABLES
C            THAT ARE STORED IN MEMORY.
C     THE STATEMENT NUMBER 10 AND THE GO TO 10 ARE INTENDED TO
C     FORCE OPTIMIZING COMPILERS TO GENERATE CODE SATISFYING 
C     ASSUMPTION 2.
C     UNDER THESE ASSUMPTIONS, IT SHOULD BE TRUE THAT,
C            A  IS NOT EXACTLY EQUAL TO FOUR-THIRDS,
C            B  HAS A 0.0d0 FOR ITS LAST BIT OR DIGIT,
C            C  IS NOT EXACTLY EQUAL TO 1.0d0,
C            EPS  MEASURES THE SEPARATION OF 1.0 FROM
C                 THE NEXT LARGER FLOATING POINT NUMBER.
C     THE DEVELOPERS OF EISPACK WOULD APPRECIATE BEING INFORMED
C     ABOUT ANY SYSTEMS WHERE THESE ASSUMPTIONS DO NOT HOLD.
C
C     *****************************************************************
C     THIS ROUTINE IS 1.0d0 OF THE AUXILIARY ROUTINES USED BY EISPACK III
C     TO AVOID MACHINE DEPENDENCIES.
C     *****************************************************************
C
C     THIS VERSION DATED 4/6/83.
C
      A = dble(4)/dble(3)
   10 B = A - 1.0d0
      C = B + B + B
      EPS = ABS(C-1.0d0)
      IF (EPS .EQ. 0.0d0) GO TO 10
      EPSLON = EPS*ABS(X)
      RETURN
      END
      SUBROUTINE MM (A, LDA, N1, N3, B, LDB, N2, C, LDC)
      doubleprecision A(LDA,*), B(LDB,*), C(LDC,*)
C
C   PURPOSE:
C     MULTIPLY MATRIX B TIMES MATRIX C AND STORE THE RESULT IN MATRIX A.
C
C   PARAMETERS:
C
C     A doubleprecision(LDA,N3), MATRIX OF N1 ROWS AND N3 COLUMNS
C
C     LDA INTEGER, LEADING DIMENSION OF ARRAY A
C
C     N1 INTEGER, NUMBER OF ROWS IN MATRICES A AND B
C
C     N3 INTEGER, NUMBER OF COLUMNS IN MATRICES A AND C
C
C     B doubleprecision(LDB,N2), MATRIX OF N1 ROWS AND N2 COLUMNS
C
C     LDB INTEGER, LEADING DIMENSION OF ARRAY B
C
C     N2 INTEGER, NUMBER OF COLUMNS IN MATRIX B, AND NUMBER OF ROWS IN
C         MATRIX C
C
C     C doubleprecision(LDC,N3), MATRIX OF N2 ROWS AND N3 COLUMNS
C
C     LDC INTEGER, LEADING DIMENSION OF ARRAY C
C
C ----------------------------------------------------------------------
C
      DO 20 J = 1, N3
         DO 10 I = 1, N1
            A(I,J) = 0.0d0
   10    CONTINUE
         CALL SMXPY (N2,A(1,J),N1,LDB,C(1,J),B)
   20 CONTINUE
C
      RETURN
      END
      SUBROUTINE SMXPY (N1, Y, N2, LDM, X, M)
      doubleprecision Y(*), X(*), M(LDM,*)
C
C   PURPOSE:
C     MULTIPLY MATRIX M TIMES VECTOR X AND ADD THE RESULT TO VECTOR Y.
C
C   PARAMETERS:
C
C     N1 INTEGER, NUMBER OF ELEMENTS IN VECTOR Y, AND NUMBER OF ROWS IN
C         MATRIX M
C
C     Y doubleprecision(N1), VECTOR OF LENGTH N1 TO WHICH IS ADDED THE PRODUCT M*X
C
C     N2 INTEGER, NUMBER OF ELEMENTS IN VECTOR X, AND NUMBER OF COLUMNS
C         IN MATRIX M
C
C     LDM INTEGER, LEADING DIMENSION OF ARRAY M
C
C     X doubleprecision(N2), VECTOR OF LENGTH N2
C
C     M doubleprecision(LDM,N2), MATRIX OF N1 ROWS AND N2 COLUMNS
C
C ----------------------------------------------------------------------
C
C   CLEANUP ODD VECTOR
C
      J = MOD(N2,2)
      IF (J .GE. 1) THEN
         DO 10 I = 1, N1
            Y(I) = (Y(I)) + X(J)*M(I,J)
   10    CONTINUE
      ENDIF
C
C   CLEANUP ODD GROUP OF TWO VECTORS
C
      J = MOD(N2,4)
      IF (J .GE. 2) THEN
         DO 20 I = 1, N1
            Y(I) = ( (Y(I))
     $             + X(J-1)*M(I,J-1)) + X(J)*M(I,J)
   20    CONTINUE
      ENDIF
C
C   CLEANUP ODD GROUP OF FOUR VECTORS
C
      J = MOD(N2,8)
      IF (J .GE. 4) THEN
         DO 30 I = 1, N1
            Y(I) = ((( (Y(I))
     $             + X(J-3)*M(I,J-3)) + X(J-2)*M(I,J-2))
     $             + X(J-1)*M(I,J-1)) + X(J)  *M(I,J)
   30    CONTINUE
      ENDIF
C
C   CLEANUP ODD GROUP OF EIGHT VECTORS
C
      J = MOD(N2,16)
      IF (J .GE. 8) THEN
         DO 40 I = 1, N1
            Y(I) = ((((((( (Y(I))
     $             + X(J-7)*M(I,J-7)) + X(J-6)*M(I,J-6))
     $             + X(J-5)*M(I,J-5)) + X(J-4)*M(I,J-4))
     $             + X(J-3)*M(I,J-3)) + X(J-2)*M(I,J-2))
     $             + X(J-1)*M(I,J-1)) + X(J)  *M(I,J)
   40    CONTINUE
      ENDIF
C
C   MAIN LOOP - GROUPS OF SIXTEEN VECTORS
C
      JMIN = J+16
      DO 60 J = JMIN, N2, 16
         DO 50 I = 1, N1
            Y(I) = ((((((((((((((( (Y(I))
     $             + X(J-15)*M(I,J-15)) + X(J-14)*M(I,J-14))
     $             + X(J-13)*M(I,J-13)) + X(J-12)*M(I,J-12))
     $             + X(J-11)*M(I,J-11)) + X(J-10)*M(I,J-10))
     $             + X(J- 9)*M(I,J- 9)) + X(J- 8)*M(I,J- 8))
     $             + X(J- 7)*M(I,J- 7)) + X(J- 6)*M(I,J- 6))
     $             + X(J- 5)*M(I,J- 5)) + X(J- 4)*M(I,J- 4))
     $             + X(J- 3)*M(I,J- 3)) + X(J- 2)*M(I,J- 2))
     $             + X(J- 1)*M(I,J- 1)) + X(J)   *M(I,J)
   50    CONTINUE
   60 CONTINUE
      RETURN
      END
