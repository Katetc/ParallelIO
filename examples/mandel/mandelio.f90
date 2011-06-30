subroutine chunk1 (xsegment, ysegment, x1, rank, startx, starty, endx, endy)

implicit none

integer, intent(in)::xsegment, ysegment, rank, x1
integer:: startx, starty, endx, endy

starty = rank/x1
starty = starty*ysegment+1

startx = mod(rank, x1)
startx = startx*xsegment+1

endx = startx + xsegment -1
endy = starty + ysegment-1 


!write (*,*) 'subroutine values for starty, endy, startx, endx',starty,endy, startx, endx

 
end subroutine chunk1


module mandelio
!  use pio
  implicit none
  public :: write_target_netcdf 
contains
  subroutine write_target_netcdf(xsegment, ysegment,x1, rank,ntasks, startx,endx,starty,endy,globalx,globaly,field)
    use netcdf
    use mpi 
    use pio 
   
    implicit none
    
    integer, intent(in) :: rank, ntasks, startx, endx, starty, endy,globalx,globaly
    integer, intent(in), pointer :: field(:,:)
    integer :: xsegment, ysegment, x1
    integer :: ierr, ncid, dimx, dimy, varid, status,i,j,n
    integer :: starts(startx,starty),value
    integer :: strt(2), cnt(2)
    integer stat(MPI_STATUS_SIZE)
    integer :: aggregator, iotype, lcv, iostat, mytask 
    integer, dimension(xsegment*ysegment) :: compdof

    type (iosystem_desc_t):: iosystem
    type (file_desc_t):: file
    type (var_desc_t) :: variableid 
    type (io_desc_t):: iodesc

    value = (endx-startx+1)*(endy-starty+1)
        
    call  PIO_init(rank, MPI_COMM_WORLD, ntasks ,aggregator ,1 ,PIO_rearr_box ,iosystem)

    ierr = PIO_createfile(iosystem,file,PIO_iotype_pnetcdf,'pmandel.nc')

    ierr = PIO_def_dim(file, 'plon', globalx, dimx)

    ierr = PIO_def_dim(file, 'plat', globaly, dimy)
    
    ierr =  PIO_def_var(file, 'pmandelbrot', PIO_int, (/dimx,dimy/), variableid)
    
    ierr = PIO_enddef(file) 
  
    call chunk1(xsegment, ysegment,x1,rank,strt(1),strt(2),endx,endy)
   
   lcv =  1
!   allocate(result(xsegment, ysegment))
 
   do j = starty, endy
     do i = startx, endx
       compdof(lcv) = i + (j-1)*globalx
       lcv = lcv +1  
     end do 
   end do
 
   call PIO_initdecomp(iosystem, PIO_int,(/globalx,globaly/), compdof, iodesc)   
    
   call  PIO_write_darray(file, variableid, iodesc, field, iostat)

   call PIO_closefile(file)
   call  PIO_finalize(iosystem,ierr)

  end subroutine write_target_netcdf

end module mandelio