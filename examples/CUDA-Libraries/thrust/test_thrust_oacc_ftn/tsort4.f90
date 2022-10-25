program testsort
call dosort(100)
call dosort(500)
call dosort(1000)
call dosort(5000)
end
!
subroutine dosort(n)
use curand
use openacc
interface isort
    subroutine isortf(array, idx, n, stream) &
                       bind(C,name='thrust_float_sort_wrapper')
    import acc_handle_kind
    real, dimension(*), intent(in) :: array
    integer, dimension(*), intent(out) :: idx
    integer, value :: n
    integer(kind=acc_handle_kind), value :: stream
    end subroutine
end interface
real, allocatable :: a(:), b(:)
integer, allocatable :: idx(:)
type(curandGenerator) :: g
integer(kind=acc_handle_kind) :: istream
allocate(a(n))
allocate(b(n))
allocate(idx(n))
a = 0.0
istream = acc_get_cuda_stream(acc_async_sync)
istat = curandCreateGenerator(g,CURAND_RNG_PSEUDO_XORWOW)
istat = curandSetStream(g,istream)
  istat = curandGenerate(g, a, n)
  call isort(a, idx, n, istream)
!$acc kernels
x = sum(merge(a, 0.0, (idx/2)*2.ne.idx))
y = sum(merge(a, 0.0, (idx/2)*2.eq.idx))
z = sum(a)
!$acc end kernels
istat = curandDestroyGenerator(g)
print *,x,y,z,z-(x+y)
if (z-(x+y).lt.1e-4) then
  print *,"test PASSED"
else
  print *,"test FAILED"
endif
end
