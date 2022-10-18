program string
  implicit none

  character(len=5) :: first_name
  character(len=4) :: last_name

  character(10) :: full_name

  ! String concatenation
  full_name = first_name//' '//last_name

  print *, full_name
end program string
