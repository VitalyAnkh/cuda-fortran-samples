program string_array
  implicit none

  character(len=10), dimension(2) :: keys, vals

  keys = [character(len=10) :: "user", "dbname"]
  vals = [character(len=10) :: "ben", "motivation"]

  call show(keys, vals)

  contains
  subroutine show(akeys, avals)
    character(len=*), intent(in) :: akeys(:), avals(:)
    integer :: i

    do i = 1, size(akeys)
       print *, trim(akeys(i)), ": ", trim(avals(i))
    end do

    if (size(akeys) <= 2) then
       print *, 'Angle is right'
    else
       print *, 'Angle is wrong'
    end if

    do i = 10, 1, -2
       print *, 'i is', ": ", i
    end do

    do i = 1, 100
       if (i > 10) then
          exit
       end if
       print *, i
    end do

    print *, 'After exiting the do loop, i = ', i

  end subroutine show
end program string_array
