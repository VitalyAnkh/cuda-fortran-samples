subroutine square_cube(i, isquare, icube)
    integer, intent (in)  :: i              ! input
    integer, intent (out) :: isquare, icube ! output

    isquare = i**2
    icube   = i**3
end subroutine

program main
    implicit none

    external square_cube ! external subroutine
    integer :: isquare, icube
    integer, allocatable :: my_var(:)
    allocate(my_var(10))

    my_var(:) = 1

    print *, my_var(:)

    call square_cube(4.0, isquare, icube)
    print *, "i,i^2,i^3=", 4, isquare, icube
end program
