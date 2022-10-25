!
!     Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
!
!          THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT
!   WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT
!   NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
!   FITNESS FOR A PARTICULAR PURPOSE.
!
#ifdef NOCUDA
#define FUNCTIONTYPE 
#else
#define FUNCTIONTYPE attributes(device)
#endif 

#define WMMAREAL2 real(2)

MODULE VECTOR_TYPES

TYPE V2REAL2
  WMMAREAL2 :: x, y
END TYPE

TYPE V4REAL2
  WMMAREAL2 :: x, y, z, w
END TYPE

TYPE V2REAL4
  REAL(4) :: x, y
END TYPE

TYPE V3REAL4
  REAL(4) :: x, y, z
END TYPE

TYPE V4REAL4
  REAL(4) :: x, y, z, w
END TYPE

TYPE V2REAL8
  REAL(8) :: x, y
END TYPE

TYPE V2LOGICAL2
  LOGICAL(2) :: x, y
END TYPE

TYPE CUFRealVector(kind, length)
  REAL(kind) :: x(length)
  INTEGER, kind :: kind, length
END TYPE

TYPE CUFLogicalVector(kind, length)
  LOGICAL(kind) :: x(length)
  INTEGER, kind :: kind, length
END TYPE

! Make the types from the component parts
!------------------------------------------------------------------------------
interface makeCUFVector
  FUNCTIONTYPE function __pgi_make_v2real2(x, y) bind(C) result(v)
  import V2REAL2
  WMMAREAL2 :: x, y
  type(V2REAL2) :: v
  end function

  FUNCTIONTYPE function __pgi_make_v2real4(x, y) bind(C) result(v)
  import V2REAL4
  real(4), value :: x, y
  type(V2REAL4) :: v
  end function

  FUNCTIONTYPE function __pgi_make_v3real4(x, y, z) bind(C) result(v)
  import V3REAL4
  real(4), value :: x, y, z
  type(V3REAL4) :: v
  end function

  FUNCTIONTYPE function __pgi_make_v4real4(x, y, z, w) bind(C) result(v)
  import V4REAL4
  real(4), value :: x, y, z, w
  type(V4REAL4) :: v
  end function

  FUNCTIONTYPE function __pgi_make_v2real8(x, y) bind(C) result(v)
  import V2REAL8
  real(8), value :: x, y
  type(V2REAL8) :: v
  end function
end interface

! V2REAL2 Assignment
!------------------------------------------------------------------------------
interface assignment(=)
  FUNCTIONTYPE subroutine __pgi_assgn1_v2real2(y,x) bind(c)
  import V2REAL2
  type(V2REAL2), intent(out) :: y
  WMMAREAL2, dimension(2), intent(in) :: x
  end subroutine
  FUNCTIONTYPE subroutine __pgi_assgn2_v2real2(y,x) bind(c)
  import V2REAL2
  WMMAREAL2, dimension(2), intent(out) :: y
  type(V2REAL2), intent(in) :: x
  end subroutine
  FUNCTIONTYPE subroutine __pgi_assgn3_v2real2(y,x) bind(c)
  import V2REAL2
  type(V2REAL2), intent(out) :: y
  real(4), value, intent(in) :: x
  end subroutine
end interface

! V2REAL2 Arithmetic
!------------------------------------------------------------------------------
interface operator(-)
  FUNCTIONTYPE function __pgi_negate_v2real2(x) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x
  type(V2REAL2) :: v
  end function

  FUNCTIONTYPE function __pgi_sub_v2real2(x, y) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2REAL2) :: v
  end function
end interface

interface operator(+)
  FUNCTIONTYPE function __pgi_add_v2real2(x, y) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2REAL2) :: v
  end function
end interface

interface operator(*)
  FUNCTIONTYPE function __pgi_mul_v2real2(x, y) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2REAL2) :: v
  end function
end interface

interface operator(/)
  FUNCTIONTYPE function __pgi_div_v2real2(x, y) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2REAL2) :: v
  end function
end interface

interface __fma_rn
  FUNCTIONTYPE function __pgi_fma_v2real2(x, y, z) bind(C) result(v)
  import V2REAL2
  type(V2REAL2), intent(in) :: x, y, z
  type(V2REAL2) :: v
  end function
end interface

! V2REAL2 Comparisons
!------------------------------------------------------------------------------
interface operator(.eq.)
  FUNCTIONTYPE function __pgi_cmpeq_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.ne.)
  FUNCTIONTYPE function __pgi_cmpne_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.gt.)
  FUNCTIONTYPE function __pgi_cmpgt_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.lt.)
  FUNCTIONTYPE function __pgi_cmplt_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.ge.)
  FUNCTIONTYPE function __pgi_cmpge_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.le.)
  FUNCTIONTYPE function __pgi_cmple_v2real2(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

! V2REAL2 Comparisons against scalars
!------------------------------------------------------------------------------
interface operator(.eq.)
  FUNCTIONTYPE function __pgi_cmpeq_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.ne.)
  FUNCTIONTYPE function __pgi_cmpne_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.gt.)
  FUNCTIONTYPE function __pgi_cmpgt_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.lt.)
  FUNCTIONTYPE function __pgi_cmplt_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.ge.)
  FUNCTIONTYPE function __pgi_cmpge_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.le.)
  FUNCTIONTYPE function __pgi_cmple_v2realr(x, y) bind(C) result(v)
  import V2REAL2, V2LOGICAL2
  type(V2REAL2), intent(in) :: x
  real(4), value :: y
  type(V2LOGICAL2) :: v
  end function
end interface

! V2Logical2 Operations
!------------------------------------------------------------------------------
interface operator(.not.)
  FUNCTIONTYPE function __pgi_lognot_v2logical2(x) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.and.)
  FUNCTIONTYPE function __pgi_logand_v2logical2(x, y) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.or.)
  FUNCTIONTYPE function __pgi_logor_v2logical2(x, y) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.eqv.)
  FUNCTIONTYPE function __pgi_logeqv_v2logical2(x, y) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface operator(.neqv.)
  FUNCTIONTYPE function __pgi_logneqv_v2logical2(x, y) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x, y
  type(V2LOGICAL2) :: v
  end function
end interface

interface any
  FUNCTIONTYPE function __pgi_any_v2logical2(x) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x
  logical :: v
  end function
end interface

interface all
  FUNCTIONTYPE function __pgi_all_v2logical2(x) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x
  logical :: v
  end function
end interface

interface count
  FUNCTIONTYPE function __pgi_count_v2logical2(x) bind(C) result(v)
  import V2LOGICAL2
  type(V2LOGICAL2), intent(in) :: x
  integer :: v
  end function
end interface

END MODULE
