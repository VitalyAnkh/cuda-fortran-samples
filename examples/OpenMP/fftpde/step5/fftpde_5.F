      PROGRAM FFTPDE
C
C   This is the standard Fortran-77 version of the APP Benchmark 4, the
C   3-D FFT PDE benchmark.
C   On 64 bit systems, double precision should be disabled.
C   Computer specific and tuning notes may be located by searching for C>>.
C>>
C   David H. Bailey     January 8, 1991
C
C   In the following parameter statement, M1, M2 and M3 are the Log_2 of the
C   three dimensions of the 3-D input array.  Set MX = MAX (M1, M2, M3).
C   A is the multiplier of the random number generator (here set to 5^13),
C   and S is the initial seed.  AL is the value of alpha.  NT is the number
C   of iterations.  timer is a double precision function that returns elapsed
C   CPU time in timers.
C
      IMPLICIT real*8 (A-H, O-Z)
      PARAMETER (M1 = 7, M2 = 7, M3 = 6, MM = M1 + M2 + M3, MP = M3,     
     $  MX = 4, N1 = 2 ** M1, N2 = 2 ** M2, N3 = 2 ** M3, NN = 2 ** MM, 
     $  NP = 2 ** MP, NT = 6, NX = 2 ** MX, A = 1220703125.D0,
     $  AL = 1D-6, PI = 3.141592653589793238D0, S = 314159265.D0)
c     PARAMETER (M1 = 8, M2 = 8, M3 = 7, MM = M1 + M2 + M3,
c    $  MX = 8, N1 = 2 ** M1, N2 = 2 ** M2, N3 = 2 ** M3, NN = 2 ** MM, 
c    $  NT = 6, NX = 2 ** MX, A = 1220703125.D0,
c    $  AL = 1D-6, PI = 3.141592653589793238D0, S = 314159265.D0)
c     DIMENSION U(2*NX), X0(2*N1*N2*N3), X1(N1,N2,2*N3),                
c    $  X2(N1,N2,2*N3), X3(N1,N2,N3), Y(4*NX)
      DIMENSION X1real(N1,N2,N3), X2real(N1,N2,N3), X3(N1,N2,N3)
      DIMENSION X1imag(N1,N2,N3), X2imag(N1,N2,N3)
      real*8 tm1,tm0,timer
c
c for some reason SPARC doesnt like passing double prec const to vranlc
c make it a variable.
c
      open (unit=6,file='fft.out',status='unknown')
      aa=a
C
C   Initialize.
C
      WRITE (6, 1) N1, N2, N3
 1    FORMAT ('3-D FFT PDE TEST'/'DIMENSIONS =',3I5)
      TM0 = timer ()
c     CALL CFFTZ (0, MX, U, X0, Y)
      CALL VRANLC (0, T1, Aa, X0)
      RN = 1.D0 / NN
      AP = - 4.D0 * AL * PI ** 2
      N12 = N1 / 2
      N22 = N2 / 2
      N32 = N3 / 2
C
C   Compute AN = A ^ (2 * NQ) (mod 2^46).
C
      T1 = A
C
      DO 100 I = 1, M1+M2+1
        T1 = RANDLC (T1, T1)
 100  CONTINUE
C
      AN = T1
      TT = S
C
C   Each instance of this loop may be performed independently.
C
!$OMP PARALLEL PRIVATE(KK,KL,T1,T2,IK)
!$OMP DO
      DO 130 K = 1, N3
        KK = K - 1
        KL = KK
        T1 = S
        T2 = AN
C
C   Find starting seed T1 for this KK using the binary rule for exponentiation.
C
        DO 110 I = 1, 100
          IK = KK / 2
          IF (2 * IK .NE. KK) T2 = RANDLC (T1, T2)
          IF (IK .EQ. 0) GOTO 120
          T2 = RANDLC (T2, T2)
          KK = IK
 110    CONTINUE
C
C   Compute 2 * NQ pseudorandom numbers.
C
 120   continue
        CALL VRANLC (N1*N2, T2, aa, x1real(1,1,k))
        CALL VRANLC (N1*N2, T2, aa, x1imag(1,1,k))
 130  CONTINUE
!$OMP END PARALLEL
C
C   On a single processor system, the 110 loop and line 120 can be replaced
C   by the following single line.
C
C120     CALL VRANLC (2 * NQ, TT, Aa, X0(2*KL*NQ+1))
C
C   Copy data in X0 into the 3-D array X1.
C
c     DO 160 K = 1, N3
c       K1 = K - 1
c
c       DO 150 J = 1, N2
c         J1 = J - 1
c         JK = 2 * (J1 + K1 * N2) * N1
c
c         DO 140 I = 1, N1
c           X1(I,J,K) = X0(2*I-1+JK)
c           X1(I,J,K+N3) = X0(2*I+JK)
c140      CONTINUE
C
c150    CONTINUE
c160  CONTINUE
C
C   Perform a forward 3-D FFT on X1.
C
      CALL CFFT3 (-1, M1, M2, M3, N1, N2, N3, X1real,x1imag)
C
C   Compute exponential terms.
C
!$OMP PARALLEL PRIVATE(K1,J1,JK,I1)
!$OMP DO
      DO 190 K = 1, N3
        K1 = K - 1
        IF (K .GT. N32) K1 = K1 - N3
C
        DO 180 J = 1, N2
          J1 = J - 1
          IF (J .GT. N22) J1 = J1 - N2
          JK = J1 ** 2 + K1 ** 2
C
          DO 170 I = 1, N1
            I1 = I - 1
            IF (I .GT. N12) I1 = I1 - N1
            X3(I,J,K) = EXP (AP * (I1 ** 2 + JK))
 170      CONTINUE
C
 180    CONTINUE
 190  CONTINUE
!$OMP END PARALLEL
C
C   Perform the following for KT = 1, ..., NT.
C
      DO 270 KT = 1, NT
C
C   Multiply by the exponential term raised to the KT power.
C
!$OMP PARALLEL PRIVATE(T1)
!$OMP DO
        DO 220 K = 1, N3
          DO 210 J = 1, N2
            DO 200 I = 1, N1
              T1 = X3(I,J,K) ** KT
              X2real(I,J,K) = T1 * X1real(I,J,K)
              X2imag(I,J,K) = T1 * X1imag(I,J,K)
 200        CONTINUE
 210      CONTINUE
 220    CONTINUE
!$OMP END PARALLEL
C
C   Compute inverse 3-D FFT.
C
        CALL CFFT3 (1, M1, M2, M3, N1, N2, N3, x2real,x2imag)
C
C   Normalize by 1 / (N1 * N2 * N3).
C
!$OMP PARALLEL
!$OMP DO
        DO 250 K = 1, N3
          DO 240 J = 1, N2
            DO 230 I = 1, N1
              X2real(I,J,K) = RN * X2real(I,J,K)
              X2imag(I,J,K) = RN * X2imag(I,J,K)
 230        CONTINUE
 240      CONTINUE
 250    CONTINUE
!$OMP END PARALLEL
C
C   Compute checksum.
C
        T1 = 0.D0
        T2 = 0.D0
C
        DO 260 I = 1, N3
        DO 260 J = 1, 1024/N3
		  I1 = J - 1 + (I-1) * (1024/N3)
          JJ = MOD (3 * I1, N2) + 1
          KK = MOD (5 * I1, N3) + 1
          T1 = T1 + X2real(JJ,KK,I)
          T2 = T2 + X2imag(JJ,KK,I)
 260    CONTINUE
c       DO 260 I = 1, 1024
c         I1 = I - 1
c         II = MOD (I1, N1) + 1
c         JJ = MOD (3 * I1, N2) + 1
c         KK = MOD (5 * I1, N3) + 1
c         T1 = T1 + X2real(II,JJ,KK)
c         T2 = T2 + X2imag(II,JJ,KK)
c260    CONTINUE
C
        WRITE (6, 2) KT, T1, T2
 2      FORMAT ('T =',I5,5X,'CHECKSUM =',1P2D22.12)
 270  CONTINUE
C
      TM1 = timer ()
      TM = TM1 - TM0
      WRITE (6, 3) TM
 3    FORMAT ('TEST COMPLETED'/'CPU TIME =',F12.6)
      STOP
      END
C
      FUNCTION RANDLC (X, A)
C
C   This routine returns a uniform pseudorandom double precision number in the
C   range (0, 1) by using the linear congruential generator
C
C   x_{k+1} = a x_k  (mod 2^46)
C
C   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
C   before repeating.  The argument A is the same as 'a' in the above formula,
C   and X is the same as x_0.  A and X must be odd double precision integers
C   in the range (1, 2^46).  The returned value RANDLC is normalized to be
C   between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
C   the new seed x_1, so that subsequent calls to RANDLC using the same
C   arguments will generate a continuous sequence.
C
C   This routine should produce the same results on any computer with at least
C   48 mantissa bits in double precision floating point data.  On 64 bit
C   systems, double precision should be disabled.
C
C   David H. Bailey     October 26, 1990
C
      IMPLICIT real*8 (A-H, O-Z)
      SAVE KS, R23, R46, T23, T46
      DATA KS/0/
C
C   If this is the first call to RANDLC, compute R23 = 2 ^ -23, R46 = 2 ^ -46,
C   T23 = 2 ^ 23, and T46 = 2 ^ 46.  These are computed in loops, rather than
C   by merely using the ** operator, in order to insure that the results are
C   exact on all systems.  This code assumes that 0.5D0 is represented exactly.
C
      IF (KS .EQ. 0) THEN
        R23 = 1.D0
        R46 = 1.D0
        T23 = 1.D0
        T46 = 1.D0
C
        DO 100 I = 1, 23
          R23 = 0.5D0 * R23
          T23 = 2.D0 * T23
 100    CONTINUE
C
        DO 110 I = 1, 46
          R46 = 0.5D0 * R46
          T46 = 2.D0 * T46
 110    CONTINUE
C
        KS = 1
      ENDIF
C
C   Break A into two parts such that A = 2^23 * A1 + A2.
C
      T1 = R23 * A
      A1 = AINT (T1)
      A2 = A - T23 * A1
C
C   Break X into two parts such that X = 2^23 * X1 + X2, compute
C   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
C   X = 2^23 * Z + A2 * X2  (mod 2^46).
C
      T1 = R23 * X
      X1 = AINT (T1)
      X2 = X - T23 * X1
      T1 = A1 * X2 + A2 * X1
      T2 = AINT (R23 * T1)
      Z = T1 - T23 * T2
      T3 = T23 * Z + A2 * X2
      T4 = AINT (R46 * T3)
      X = T3 - T46 * T4
c     RANDLC = R46 * X
      RANDLC = X
C
      RETURN
      END
C
      SUBROUTINE VRANLC (N, X, A, Y)
C
C   This routine generates N uniform pseudorandom double precision numbers in
C   the range (0, 1) by using the linear congruential generator
C
C   x_{k+1} = a x_k  (mod 2^46)
C
C   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
C   before repeating.  The argument A is the same as 'a' in the above formula,
C   and X is the same as x_0.  A and X must be odd double precision integers
C   in the range (1, 2^46).  The N results are placed in Y and are normalized
C   to be between 0 and 1.  X is updated to contain the new seed, so that
C   subsequent calls to VRANLC using the same arguments will generate a
C   continuous sequence.  If N is zero, only initialization is performed, and
C   the variables X, A and Y are ignored.
C
C   This routine is the standard version designed for scalar or RISC systems.
C   However, it should produce the same results on any single processor
C   computer with at least 48 mantissa bits in double precision floating point
C   data.  On 64 bit systems, double precision should be disabled.
C
C   David H. Bailey     October 26, 1990
C
      IMPLICIT real*8 (A-H, O-Z)
      DIMENSION Y(N)
      SAVE KS, R23, R46, T23, T46
      DATA KS/0/
C
C   If this is the first call to VRANLC, compute R23 = 2 ^ -23, R46 = 2 ^ -46,
C   T23 = 2 ^ 23, and T46 = 2 ^ 46.  See comments in RANDLC.
C
      IF (KS .EQ. 0 .OR. N .EQ. 0) THEN
        R23 = 1.D0
        R46 = 1.D0
        T23 = 1.D0
        T46 = 1.D0
C
        DO 100 I = 1, 23
          R23 = 0.5D0 * R23
          T23 = 2.D0 * T23
 100    CONTINUE
C
        DO 110 I = 1, 46
          R46 = 0.5D0 * R46
          T46 = 2.D0 * T46
 110    CONTINUE
C
        KS = 1
        IF (N .EQ. 0) RETURN
      ENDIF
C
C   Break A into two parts such that A = 2^23 * A1 + A2.
C
      T1 = R23 * A
      A1 = AINT (T1)
      A2 = A - T23 * A1
C
C   Generate N results.   This loop is not vectorizable.
C
      DO 120 I = 1, N
C
C   Break X into two parts such that X = 2^23 * X1 + X2, compute
C   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
C   X = 2^23 * Z + A2 * X2  (mod 2^46).
C
        T1 = R23 * X
        X1 = AINT (T1)
        X2 = X - T23 * X1
        T1 = A1 * X2 + A2 * X1
        T2 = AINT (R23 * T1)
        Z = T1 - T23 * T2
        T3 = T23 * Z + A2 * X2
        T4 = AINT (R46 * T3)
        X = T3 - T46 * T4
        Y(I) = R46 * X
 120  CONTINUE
C
      RETURN
      END
C
      SUBROUTINE CFFT3 (IS, M1, M2, M3, N1, N2, N3, X1real,X1imag)
C
C   This performs a 3-D complex-to-complex FFT on the array X1, which is
C   assumed to have dimensions (N1,N2,2*N3).  It is assumed that N1 = 2 ^ M1,
C   N2 = 2 ^ M2 and N3 = 2 ^ M3.  Real and imaginary parts are stored
C   completely separated (i.e. separated by N3 units in the last dimension of
C   X1).  IS is the sign of the transform, either 1 or -1.  U is the root of
C   unity array, which must have been previously initialized by calling CFFTZ
C   with 0 as the first argument and MM = MAX (M1,M2,M3) as the timer
C   argument.  X2 is used as a scratch array and must be the same size as X1.
C   Y is a scratch array of size 4 * 2 ^ MM.
C
C   David H. Bailey     October 26, 1990
C
      IMPLICIT real*8 (A-H, O-Z)
      DIMENSION X1real(N1,N2,N3),X1imag(N1,N2,N3)
	  parameter(mxsz=8192)
	  complex z(mxsz),w(mxsz/2)
	  logical inverse
	  real timer
	  if(n1.gt.mxsz.or.n2.gt.mxsz.or.n3.gt.mxsz)then
		print *,'Error in CFFT3. Maximum size allowed=',mxsz
		stop 998
	  endif
c	  print *,'cfft3'
c	  print *,'is=',is,' m1=',m1,' m2=',m2,' m3=',m3
c	  print *,' n1=',n1,' n2=',n2,' n3=',n3
c	  print *,'time=',timer()
c	  print *,'input x1 real ='
c	  do k=1,n3
c		do j=1,n2
c			print '(4e12.5)',(x1(i,j,k),i=1,n1)
c		end do
c	  end do
c	  print *,'input x1 imaj ='
c	  do k=n3+1,2*n3
c		do j=1,n2
c			print '(4e12.5)',(x1(i,j,k),i=1,n1)
c		end do
c	  end do
	  if(is.eq.1)then 
		inverse=.false.
	  else
		inverse=.true.
	  endif
	  call tabfft(w,n1)
c	  print *,'first w:'
c	  print '(4e12.5)',(w(i),i=1,n1/2)
!$OMP PARALLEL PRIVATE(Z)
!$OMP DO
	  do k=1,n3
		do j=1,n2
			do i=1,n1
				z(i)=cmplx(x1real(i,j,k),x1imag(i,j,k))
			end do
			call fft(z,inverse,w,n1,m1)
			do i=1,n1
				x1real(i,j,k)=real(z(i))
				x1imag(i,j,k)=aimag(z(i))
			end do
		end do
	  end do
!$OMP END PARALLEL
c	  print *,'Time after first stage=',timer()
	  call tabfft(w,n2)
c	  print *,'second w:'
c	  print '(4e12.5)',(w(i),i=1,n2/2)
!$OMP PARALLEL PRIVATE(Z)
!$OMP DO
	  do k=1,n3
		do j=1,n1
			do i=1,n2
				z(i)=cmplx(x1real(j,i,k),x1imag(j,i,k))
			end do
			call fft(z,inverse,w,n2,m2)
			do i=1,n2
				x1real(j,i,k)=real(z(i))
				x1imag(j,i,k)=aimag(z(i))
			end do
		end do
	  end do
!$OMP END PARALLEL
c	  print *,'Time after second stage=',timer()
	  call tabfft(w,n3)
c	  print *,'second w:'
c	  print '(4e12.5)',(w(i),i=1,n3/2)
!$OMP PARALLEL PRIVATE(Z)
!$OMP DO
	  do k=1,n2
		do j=1,n1
			do i=1,n3
				z(i)=cmplx(x1real(j,k,i),x1imag(j,k,i))
			end do
			call fft(z,inverse,w,n3,m3)
			do i=1,n3
				x1real(j,k,i)=real(z(i))
				x1imag(j,k,i)=aimag(z(i))
			end do
		end do
	  end do
!$OMP END PARALLEL
c	  print *,'Time after third stage=',timer()
c	  print *,'output x1 real ='
c	  do k=1,n3
c		do j=1,n2
c			print '(4e12.5)',(x1(i,j,k),i=1,n1)
c		end do
c	  end do
c	  print *,'output x1 imaj ='
c	  do k=n3+1,2*n3
c		do j=1,n2
c			print '(4e12.5)',(x1(i,j,k),i=1,n1)
c		end do
c	  end do
C
      RETURN
      END

      subroutine fft( x,inverse,w,nx,lnx )
      complex x(nx),
     $        w(nx/2)
      logical  inverse

c   fft
c   (note that variable iter = log(base 2) of nx

      real    rnx
      complex t,
     $        wk

      integer itab(12)
      integer iter

      data itab / 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 /
      iter=lnx

      nxp2 = nx
      rnx  = 1./float(nx)

      do 30 it = 1,iter
         n = 1
         nxp = nxp2
         nxp2 = nxp/2
         do 20 m = 1,nxp2
			if( inverse ) then
				wk = conjg(w(n))
			else
				wk = w(n)
			end if
            do 10 mxp = nxp,nx,nxp
               j1 = mxp - nxp + m
               j2 = j1 + nxp2
                  t = x(j1) - x(j2)
                  x(j1) = x(j1) + x(j2)
                  x(j2) = t*wk
c	  print *,'it=',it,' m=',m,' mxp=',mxp,' j1=',j1,' j2=',j2
  10        continue
            n = n +itab(it)
  20     continue
  30  continue

      n2 = nx/2
      n1 = nx - 1
      j = 1

      do i = 1,n1
		if( i .lt. j ) then
			t = x(j)
			x(j) = x(i)
			x(i) = t
		end if
        k = n2
  42    if( k .lt. j ) then
            j = j - k
            k = k/2
            go to 42
		end if
         j = j + k
      end do

      if( inverse ) then
         do k = 1,nx
            x(k) = rnx*x(k)
         end do
      end if

      end
      subroutine tabfft(w,nx)
c----------------------------------------------------------------------
c     calculate some arrays needed for the fft filter
c----------------------------------------------------------------------
      complex w(nx/2)

      do 1 i=1,nx/2
      arg = (i-1)*3.141592653589793/float(nx/2)
      w(i) = cmplx(cos(arg),sin(arg))
   1  continue
      return
      end
c
      real*8 function timer()
      real*4 etime, tarray(2)
      timer = etime(tarray)
      end
