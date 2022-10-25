!
!     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
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

module matrix

type :: base_matrix(k,c,r)   
  private
    integer, kind :: k = 4 
    integer, len :: c = 1
    integer, len :: r = 1 
end type base_matrix

type, extends(base_matrix) ::  adj_matrix
  private
    class(*), pointer :: m(:,:) => null()
end type adj_matrix

interface getKind
  module procedure getKind4
  module procedure getKind8
end interface getKind

interface getColumns
  module procedure getNumCols4
  module procedure getNumCols8
end interface getColumns

interface getRows
  module procedure getNumRows4
  module procedure getNumRows8
end interface getRows

interface adj_matrix
   module procedure construct_4   ! kind=4 constructor
   module procedure construct_8   ! kind=8 constructor
end interface adj_matrix

interface assignment(=)
   module procedure m2m4          ! assign kind=4 matrix 
   module procedure a2m4          ! assign kind=4 array  
   module procedure m2m8          ! assign kind=8 matrix 
   module procedure a2m8          ! assign kind=8 array  
   module procedure m2a4          ! assign kind=4 matrix to array
   module procedure m2a8          ! assign kind=8 matrix to array
end interface assignment(=)


contains

  function getKind4(this) result(rslt)
   class(adj_matrix(4,*,*)) :: this
   integer :: rslt
   rslt = this%k
  end function getKind4

 function getKind8(this) result(rslt)
   class(adj_matrix(8,*,*)) :: this
   integer :: rslt
   rslt = this%k
 end function getKind8

  function getNumCols4(this) result(rslt)
   class(adj_matrix(4,*,*)) :: this
   integer :: rslt
   rslt = this%c
  end function getNumCols4

  function getNumCols8(this) result(rslt)
   class(adj_matrix(8,*,*)) :: this
   integer :: rslt
   rslt = this%c
  end function getNumCols8

  function getNumRows4(this) result(rslt)
   class(adj_matrix(4,*,*)) :: this
   integer :: rslt
   rslt = this%r
  end function getNumRows4

  function getNumRows8(this) result(rslt)
   class(adj_matrix(8,*,*)) :: this
   integer :: rslt
   rslt = this%r
  end function getNumRows8


 function construct_4(k,c,r) result(mat)
     integer(4) :: k
     integer :: c
     integer :: r
     class(adj_matrix(4,:,:)),allocatable :: mat

     allocate(adj_matrix(4,c,r)::mat)

  end function construct_4

  function construct_8(k,c,r) result(mat)
     integer(8) :: k
     integer :: c
     integer :: r
     class(adj_matrix(8,:,:)),allocatable :: mat

     allocate(adj_matrix(8,c,r)::mat)

  end function construct_8

  subroutine a2m4(d,s)
   class(adj_matrix(4,:,:)),allocatable :: d
   class(*),dimension(:,:) :: s

   if (allocated(d)) deallocate(d)
   allocate(adj_matrix(4,size(s,1),size(s,2))::d)
   allocate(d%m(size(s,1),size(s,2)),source=s)
 end subroutine a2m4

 subroutine a2m8(d,s)
   class(adj_matrix(8,:,:)),allocatable :: d
   class(*),dimension(:,:) :: s

   if (allocated(d)) deallocate(d)
   allocate(adj_matrix(8,size(s,1),size(s,2))::d)
   allocate(d%m(size(s,1),size(s,2)),source=s)
 end subroutine a2m8

subroutine m2a8(a,this)
class(adj_matrix(8,*,*)) :: this
class(*),allocatable :: a(:,:)
  if (allocated(a)) deallocate(a)
  allocate(a(size(this%m,1),size(this%m,2)),source=this%m)
 end subroutine m2a8

 subroutine m2a4(a,this)
 class(adj_matrix(4,*,*)) :: this
 class(*),allocatable :: a(:,:)
   if (allocated(a)) deallocate(a)
   allocate(a(size(this%m,1),size(this%m,2)),source=this%m)
 end subroutine m2a4

  subroutine m2m4(d,s)
   class(adj_matrix(4,:,:)),allocatable :: d
   class(adj_matrix(4,*,*)) :: s

   if (allocated(d)) deallocate(d)
   allocate(d,source=s)
 end subroutine m2m4

 subroutine m2m8(d,s)
   type(adj_matrix(8,:,:)),allocatable :: d
   type(adj_matrix(8,*,*)) :: s

   if (allocated(d)) deallocate(d)
   allocate(d,source=s)
 end subroutine m2m8


end module matrix


program adj3

  use matrix
  implicit none
  integer(8) :: i,j,k,nerrors

  type(adj_matrix(8,:,:)),allocatable :: adj
  real(8) :: a(2,3)
  real(8),allocatable :: b(:,:)
   
  nerrors = 0
  adj = adj_matrix(INT(8,8),2,4)

  k = 1
  do i=1,2
   do j=1, 3
    a(i,j) = k
    k = k + 1
   enddo
  enddo

  adj = a
  b = adj

  do i=1,2
   do j=1, 3
    if (b(i,j) .ne. a(i,j)) then
      nerrors = nerrors + 1
    endif
   enddo
  enddo

  if (nerrors .ne. 0) then
     print *, "Test FAILED"
  else
     print *, "Test PASSED"
  endif

end program adj3



