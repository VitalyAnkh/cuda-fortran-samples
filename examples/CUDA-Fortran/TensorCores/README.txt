Introduction
------------
    For certain specific operations, tensor cores offer a 4x performance speedup
    over typical CUDA GPU core programming.  Those specific operations are
    currently matrix multiply operations of two forms.  The first, simpler form:

      Fortran: C = Matmul(A, B), where
        A is a 2 dimensional array dimensioned A(m,k)
        B is a 2 dimensional array dimensioned B(k,n)
        C is a 2 dimensional array dimensioned C(m,n)
        C(i,j) = sum(a(i,:) * b(:,j)), for i=1,m, j=1,n

    A more general form which tensor cores support in hardware, is:
      Fortran: D = C + Matmul(A, B), where
        A, B, C are defined as above.
        D is a 2 dimensional array dimensioned D(m,n), similar to C.

    The tensor core unit itself currently supports 3 combinations of sizes
    for m, n, and k.  The current CUDA naming practice is to run the sizes
    together, so we will refer to them here as m16n16k16, m8n32k16, and
    m32n8k16.  Note that k, the inner dot-product dimension, is currently
    always 16.  For GPU memory arrays of different dimensions than these,
    it is the programmers' responsibility to add padding and blocking.

    We've created a CUDA Fortran device module named WMMA (Warp-Matrix-Multiply-Add). 
    This is the name of the CUDA C++ namespace, and is used like this in CUDA C++:

       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    In Fortran, in the declaration section of your device subroutines and
    functions, you add this line to enable the tensor core functionality:

       use wmma

Data Declaration
----------------
    The most synonymous translation of the C++ declaration above to Fortran
    would be to use parameterized derived types, i.e. the c_frag would be
    declared like this:

      type(fragment(wmmatype=acc, m=WMMA_M, n=WMMA_N, k=WMMA_K, kind=4)) :: c_frag

    We've currently chosen to NOT go that route.  The current implementation of
    nvfortran stores the derived type parameters within the derived type, as
    derived type parameter enquiry must be supported by the language
    implementation, i.e. "print *,c_frag%m" must work.  Storing all of these
    parameters takes up a lot of register space, which adversely affects
    performance on a register-limited GPU.

    The WMMA module defines derived types that have this form:

      type(subMatrixC_m16n16k16_Real4) :: c_frag

    In 2019, we recommend developers use the macros we have provided in
    these examples.  We've determined that through macro pre-processing we
    can generate either the parameterized derived type declaration or the
    specific type with the same code base.  Here is the declaration we use
    before macro pre-processing:
      WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind4) :: c_frag

    In both CUDA C++, and CUDA Fortran, it is important for performance
    reasons that the underlying operations can be called quickly without
    any runtime parameter enquiry overhead.  Using C++ templates and these
    specific Fortran types allows for that.

    The A and B subMatrix types must be distinct.  In addition, as part of their
    declaration, they must be defined to be stored internally as either column
    major or row major.  Our implementation today only supports real(2) data.
    The macro declarations look like this, where WMMAColMajor and WMMARowMajor
    are defined in the WMMA module:

      WMMASubMatrix(WMMAMatrixA, 16, 16, 16, Real, WMMAColMajor) :: sa
      WMMASubMatrix(WMMAMatrixB, 16, 16, 16, Real, WMMARowMajor) :: sb

    Declaring B to be row major and using it in a Matmul operation is equivalent
    to this Fortran statement:
      C = Matmul(A, Transpose(B))

    Likewise, declaring A to be row major is equivalent to this;
      C = Matmul(Transpose(A), B)

    The C subMatrix is not declared with storage order, but it can be loaded
    and stored to memory in either order.  For instance, storing the C result,
    computed in column major order, in row major order is equivalent to this:
      C = Transpose(Matmul(A,B))

    Users of the BLAS dgemm and other *gemm routines will recognize these
    transpose capabilities.

    For both the larger matrices stored in CUDA global memory, and for operating
    on the individual elements of the subMatrices, we use a 16-bit data type,
    which at some point soon in the future will be real(2).  Until then, we have
    created other macros, which developers may use while they get started using
    tensor cores.  This is a straight-forward translation, and so for the entire
    declaration section of the simplest use case, it looks like this:
      use wmma
      CUFReal2, device :: a(m,k)
      CUFReal2, device :: b(k,n)
      real(4), device :: c(m,n)
      WMMASubMatrix(WMMAMatrixA, 16, 16, 16, Real, WMMAColMajor) :: sa
      WMMASubMatrix(WMMAMatrixB, 16, 16, 16, Real, WMMAColMajor) :: sb
      WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind4)    :: sc

WMMA Basic Operations
---------------------
    At the most basic level, to use the tensor cores, you just load the input
    subMatrices from global memory, compute C = Matmul(A,B), and store the
    result back to memory.  The simplest use of the declarations above looks
    like this in Fortran:
      call wmmaLoadMatrix(sa, a(1,1), m)
      call wmmaLoadMatrix(sb, b(1,1), k)
      call wmmaMatmul(sc, sa, sb)
      call wmmaStoreMatrix(c(1,1), sc, m)

    We've named each of these subroutines starting with "wmma" because they
    require every thread of the warp to make the call, and the threads in the
    warp will cooperate to do the work.

    The wmmaLoadMatrix subroutine takes as arguments the derived type to fill,
    the address of the global or shared memory upper left corner, and the stride
    between columns.  The stride is required and corresponds to the lda, ldb,
    and ldc arguments in *gemm BLAS calls.

    The wmmaMatmul subroutine can take 3 or 4 arguments.  These uses are most
    typical:
      call wmmaMatmul(sc, sa, sb)
      call wmmaMatmul(sc, sa, sb, sc)
      call wmmaMatmul(sd, sa, sb, sc)
    As mentioned above, A and B are always real(2), but can be any combination
    of column major or row major ordered.  C and D can be any combination of
    real(2) and real(4).

    The wmmaStoreMatrix is the partner to wmmaLoadMatrix.  The upper left corner
    in global or shared memory is the first argument.  Then the type, then the
    stride.  The A and B subMatrix types cannot be stored, only C.  A fourth
    argument is optional, and specifies the layout, either column major (the
    default) or row major when the store occurs.

WMMA Advanced Operations
------------------------
    We've added some additional warp-level functionality, and expect to work
    with our early adopters to fill in the gaps.  The problems we expect
    developers to encounter include how to detect numeric issues, and how
    to potentially fix them up within device kernels.

    Treating the derived type subMatrices as just 2D numeric arrays through
    overloaded functions can solve some of these problems.  Here are some
    things we've added.

    1. Logical operations:
       You can declare, or even generate without declaring, logical, kind=1,
       WMMA subMatrixC types.  This allows logical operations between two
       WMMAMatrixC types, or between one C type and a scalar.  These logical
       results can then be passed, in turn, to Fortran intrinsic functions like
       Merge(), and reductions like Any(), All(), and Count().  The code for
       that looks like this:
         WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind4)    :: sd
         ...
         call wmmaMatmul(sd, sa, sb, sc)
         if (wmmaAll(sd .lt. 64.0)) then
           . . .
         npos = wmmaCount(sd .ge. 0.0)
           . . .
         call wmmaMerge(sc, -99.0, sd, mask=sd.eq.sc)

       If you would prefer to use the common Fortran names, you can rename the
       local name on the use statement like this:
         use wmma, any => wmmaAny, all => wmmaAll

       Remember in our naming convention, functions and subroutines that start
       with "wmma" imply the work is shared amongst the threads in a warp.

       Merge() is a subroutine, while count(), any(), and all() are functions.
       In general, subprograms which natively return less than a 64-bit scalar
       are okay as device functions.  We prefer to make array-valued subprograms
       and those returning large derived types subroutines for performance
       reasons, and the returned value or intent(out) array or object is
       typically the first argument..

    2. Assignment
       Initializing a subMatrix to zero is a basic operation:
         sc = 0.0
       You can also assign one subMatrixC type to another:
         sc = sd

Operations on the WMMA Underlying Data
--------------------------------------
    We have settled on the name of the data held under the derived types to
    be "x".  The size of that data and how it maps to the original matrix
    locations may be subject to change.  Operations on that data should be
    elemental and applied uniformly by all threads in the warp.  We expect
    something like the following should always be safe:

        WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind4) :: sd
        real(4), device :: xoff, xtol
        ...
        do klp = 1, maxtimes
          if (wmmaAny(sd .gt. xtol)) then
            do i = 1, size(sd%x)
              sd%x(i) = sd%x(i) - xoff
            end do
          end if
        end do

    For the equivalent functionality using the current implementation when
    kind=2, declare the local scalars using the macro:

        WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind2) :: sd
        CUFReal2 :: xoff, xtol, xtmp
        ...
        xoff = CUFreal(10, kind=2)
        ...
        do klp = 1, maxtimes
          if (wmmaAny(sd .gt. xtol)) then
            do i = 1, size(sd%x)
              ! CUFReal2 arithmetic has to be done on scalars for now
              xtmp = sd%x(i)
              sd%x(i) = xtmp - xoff
            end do
          end if
        end do

    Note that we've defined a function named CUFreal() to perform type
    conversions.  Three specific cases are currently implemented:
         real2 = CUFReal(int4,  kind=2)
         real2 = CUFReal(real4, kind=2)
         real4 = CUFReal(real2, kind=4)
    We expect this function to change to the Fortran intrinsic real() when
    we add real(2) support to the language in an upcoming release.  The
    macro can be redefined at that point.

    For the most advanced users, and for top performance, you should operate
    on the 16-bit data two at a time.  We have an experimental module, named
    vector_types, which we have provided to interested users.  Using a vector
    of real(2) of length 2, in combination with cray pointers, results in
    very efficient updates of the subMatrix data:

        use vector_types
        WMMASubMatrix(WMMAMatrixC, 16, 16, 16, Real, WMMAKind2)    :: sd
        CUFVector(N2, Real, 2), device :: cv, dc(*)
        pointer(dp, dc)
        CUFReal2 :: xoff

        dp = loc(sd%x(1))
        xoff = CUFreal(10, kind=2)
        cv = makeCUFVector(xoff,xoff)
        do klp = 1, maxtimes
          if (wmmaAny(sd .gt. xtol)) then
            do i = 1, size(sd%x)/2
              dc(i) = dc(i) - cv
            end do
          end if
        end do

    For efficiency, we also support using the hardware fma operations for both
    the CUFReal2 type and the vector of two real(2) types, i.e.
      d = __fma_rn(a, b, c), which is d = a*b+c.

Host-side Support For Tensor Core Programming
---------------------------------------------
    Host-side support is limited.  For our testing purposes, we have created
    a few functions which have shared in this PGI examples area under the
    Utils directory.  Modern X86 processors support real(4) <-> real(2) type
    conversion uing SSE instructions, and our test code takes advantage of that
    when possible.  The cpuid feature to key on is "f16c", and we have a function
    in our runtime to test that.  We have also written a scalar version of
    half2float() and float2half(), the latter taking the current CPU rounding
    mode into account.

    We have versions of these two conversion functions which take 1D and 2D
    arrays, which is nice for testing purposes.  These should go away once
    real(2) support is in the compiler, but the code can be used as a guide.


Provided Tests
--------------
    The tests are divided into three directories, to correspond to the three
    matrix sizes currently supported.

In m16n16k16:
  wmma1.CUF: Simplest example, multiply two 16x16 real(2) arrays, real(4) result.
  wmma2.CUF: A Submatrix is row major, real(2) result
  wmma3.CUF: A, B Submatrix is row major, real(2) result
  wmma4.CUF: C Submatrix is real(2), D is real(4)
  wmma5.CUF: Store C Submatrix as row major, real(4) result
  wmma6.CUF: Use Submatrix comparison, merge
  wmma7.CUF: Use Submatrix comparison, all
  wmma8.CUF: Use Submatrix comparison, any, v2real2

In m32n8k16:
  wmma1.CUF: Simplest example, 32x16 X 16x8 real(2) arrays, real(4) result.
  wmma2.CUF: B Submatrix is row major, real(2) result
  wmma3.CUF: A, B Submatrix is row major, real(4) result
  wmma4.CUF: C Submatrix is real(2), D is real(4)
  wmma5.CUF: Store C Submatrix as row major, real(4) result
  wmma6.CUF: Use Submatrix comparison, merge
  wmma7.CUF: Use Submatrix comparison, all
  wmma8.CUF: Use Submatrix comparison, count, v2real2

In m8n32k16:
  wmma1.CUF: Simplest example, 8x16 X 16x32 real(2) arrays, real(4) result.
  wmma2.CUF: A Submatrix is row major, real(2) result
  wmma3.CUF: B Submatrix is row major, real(4) result
  wmma4.CUF: C Submatrix is real(2), D is real(4)
  wmma5.CUF: Store C Submatrix as row major, real(4) result
  wmma6.CUF: Use Submatrix comparison, merge
  wmma7.CUF: Use Submatrix comparison, count
  wmma8.CUF: Use Submatrix comparison, any, v2real2

