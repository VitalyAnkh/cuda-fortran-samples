program derived_type
  implicit none

  type :: t_pair
     integer :: i
     real :: x
  end type

  type:: dim3
      integer(kind=4) :: x, y, z
  end type dim3

  ! All fortran variables must be declared before any executable code
  type(t_pair) :: pair
  type(t_pair) :: pair2

  type(dim3) :: grid, cluster, block

  grid = dim3(1, 1, 1)
  print *, 'grid = ', grid%x, grid%y, grid%z

  pair%i = 1
  pair%x = 3.14
  print *, pair%i, pair%x

  pair2 = t_pair(2, 6.28)
  print *, pair2%i, pair2%x
end program derived_type
