subroutine square_cube(i, isquare, icube)
    integer, intent (in)  :: i              ! input
    integer, intent (out) :: isquare, icube ! output

    isquare = i**2
    icube   = i**3
end subroutine

program main
    implicit none

    external square_cube ! external subroutine
    integer :: isq, icub

    call square_cube(4.0, isq, icub)
    print *, "i,i^2,i^3=", 4, isq, icub
end program
