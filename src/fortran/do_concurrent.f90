program do_concurrent
  implicit none

  real, parameter :: pi = 3.14159265
  integer, parameter :: n = 4
  real :: result_sin(n)
  integer :: i

  do concurrent (i = 1:n)
     result_sin(i) = sin(i * pi/4.0)
  end do
end program do_concurrent
