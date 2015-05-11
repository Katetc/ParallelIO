#include <pio.h>
#include <pio_internal.h>

#define PIO_WRITE_BUFFERING 1
PIO_Offset PIO_BUFFER_SIZE_LIMIT=10485760; // 10MB default limit
bufsize PIO_CNBUFFER_LIMIT=33554432; 
static void *CN_bpool=NULL; 
static PIO_Offset maxusage=0;

 // Changes to PIO_BUFFER_SIZE_LIMIT only apply to files opened after the change
 PIO_Offset PIOc_set_buffer_size_limit(const PIO_Offset limit)
 {
   PIO_Offset oldsize; 
   oldsize = PIO_BUFFER_SIZE_LIMIT;
   if(limit>0)
     PIO_BUFFER_SIZE_LIMIT=limit;
   return(oldsize);
 }

void compute_buffer_init(iosystem_desc_t ios)
{
#ifndef PIO_USE_MALLOC

  if(CN_bpool == NULL){
    CN_bpool = malloc( PIO_CNBUFFER_LIMIT );
    if(CN_bpool==NULL){
      char errmsg[180];
      sprintf(errmsg,"Unable to allocate a buffer pool of size %d on task %d: try reducing PIO_CNBUFFER_LIMIT\n",PIO_CNBUFFER_LIMIT,ios.comp_rank);
      piodie(errmsg,__FILE__,__LINE__);
    }
    bpool( CN_bpool, PIO_CNBUFFER_LIMIT);
    if(CN_bpool==NULL){
      char errmsg[180];
      sprintf(errmsg,"Unable to allocate a buffer pool of size %d on task %d: try reducing PIO_CNBUFFER_LIMIT\n",PIO_CNBUFFER_LIMIT,ios.comp_rank);
      piodie(errmsg,__FILE__,__LINE__);
    }
    bectl(NULL, malloc, free, PIO_CNBUFFER_LIMIT);
  }
#endif
}



 int pio_write_darray_nc(file_desc_t *file, io_desc_t *iodesc, const int vid, void *IOBUF, void *fillvalue)
 {
   iosystem_desc_t *ios;
   var_desc_t *vdesc;
   int ndims;
   int ierr;
   int i;
   int msg;
   int mpierr;
   int dsize;
   MPI_Status status;
   PIO_Offset usage;
   int fndims;
   PIO_Offset tdsize;
#ifdef TIMING
  GPTLstart("PIO:write_darray_nc");
#endif

   tdsize=0;
   ierr = PIO_NOERR;

   ios = file->iosystem;
   if(ios == NULL){
     fprintf(stderr,"Failed to find iosystem handle \n");
     return PIO_EBADID;
   }
   vdesc = (file->varlist)+vid;
  
   if(vdesc == NULL){
     fprintf(stderr,"Failed to find variable handle %d\n",vid);
     return PIO_EBADID;
   }
   ndims = iodesc->ndims;
   msg = 0;

   if(ios->async_interface && ! ios->ioproc){
     if(ios->comp_rank==0) 
       mpierr = MPI_Send(&msg, 1,MPI_INT, ios->ioroot, 1, ios->union_comm);
     mpierr = MPI_Bcast(&(file->fh),1, MPI_INT, ios->compmaster, ios->intercomm);
   }

   ierr = PIOc_inq_varndims(file->fh, vid, &fndims);

   if(ios->ioproc){
     io_region *region;
     int ncid = file->fh;
     int regioncnt;
     int rrcnt;
     void *bufptr;
     void *tmp_buf=NULL;
     int tsize;
     size_t start[fndims];
     size_t count[fndims];
     int buflen, j;

     PIO_Offset *startlist[iodesc->maxregions];
     PIO_Offset *countlist[iodesc->maxregions];
     MPI_Type_size(iodesc->basetype, &tsize);

     region = iodesc->firstregion;

     if(vdesc->record >= 0 && ndims<fndims)
       ndims++;
#ifdef _PNETCDF
     if(file->iotype == PIO_IOTYPE_PNETCDF){
       // make sure we have room in the buffer ;
       flush_output_buffer(file, false, tsize*(iodesc->maxiobuflen));
     }
#endif

     rrcnt=0;
     for(regioncnt=0;regioncnt<iodesc->maxregions;regioncnt++){
       for(i=0;i<ndims;i++){
	 start[i] = 0;
	 count[i] = 0;
       }
       if(region != NULL){
	 bufptr = (void *)((char *) IOBUF+tsize*region->loffset);
	 // this is a record based multidimensional array
	 if(vdesc->record >= 0){
	   start[0] = vdesc->record;
	   for(i=1;i<ndims;i++){
	     start[i] = region->start[i-1];
	     count[i] = region->count[i-1];
	   }
	  if(count[1]>0)
	    count[0] = 1;
	 // Non-time dependent array
	 }else{
	   for( i=0;i<ndims;i++){
	     start[i] = region->start[i];
	     count[i] = region->count[i];
	   }
	 }
       }

       switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
       case PIO_IOTYPE_NETCDF4P:
	 ierr = nc_var_par_access(ncid, vid, NC_COLLECTIVE);
	 if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
	   ierr = nc_put_vara_double (ncid, vid,(size_t *) start,(size_t *) count, (const double *) bufptr); 
	 } else if(iodesc->basetype == MPI_INTEGER){
	   ierr = nc_put_vara_int (ncid, vid, (size_t *) start, (size_t *) count, (const int *) bufptr); 
	 }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
	   ierr = nc_put_vara_float (ncid, vid, (size_t *) start, (size_t *) count, (const float *) bufptr); 
	 }else{
	   fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
	 }
	 break;
       case PIO_IOTYPE_NETCDF4C:
#endif
       case PIO_IOTYPE_NETCDF:
	 {
	   mpierr = MPI_Type_size(iodesc->basetype, &dsize);
	   size_t tstart[ndims], tcount[ndims];
	   if(ios->io_rank==0){

	     for(i=0;i<iodesc->num_aiotasks;i++){
	       if(i==0){	    
		 buflen=1;
		 for(j=0;j<ndims;j++){
		   tstart[j] =  start[j];
		   tcount[j] =  count[j];
		   buflen *= tcount[j];
		   tmp_buf = bufptr;
		 }
	       }else{
		 mpierr = MPI_Send( &ierr, 1, MPI_INT, i, 0, ios->io_comm);  // handshake - tell the sending task I'm ready
		 mpierr = MPI_Recv( &buflen, 1, MPI_INT, i, 1, ios->io_comm, &status);
		 if(buflen>0){
		   mpierr = MPI_Recv( tstart, ndims, MPI_OFFSET, i, ios->num_iotasks+i, ios->io_comm, &status);
		   mpierr = MPI_Recv( tcount, ndims, MPI_OFFSET, i,2*ios->num_iotasks+i, ios->io_comm, &status);
		   tmp_buf = malloc(buflen * dsize);	
		   mpierr = MPI_Recv( tmp_buf, buflen, iodesc->basetype, i, i, ios->io_comm, &status);
		 }
	       }

	       if(buflen>0){
		 if(iodesc->basetype == MPI_INTEGER){
		   ierr = nc_put_vara_int (ncid, vid, tstart, tcount, (const int *) tmp_buf); 
		 }else if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
		   ierr = nc_put_vara_double (ncid, vid, tstart, tcount, (const double *) tmp_buf); 
		 }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
		   ierr = nc_put_vara_float (ncid,vid, tstart, tcount, (const float *) tmp_buf); 
		 }else{
		   fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
		 }
		 if(ierr == PIO_EEDGE){
		   for(i=0;i<ndims;i++)
		     fprintf(stderr,"dim %d start %ld count %ld\n",i,tstart[i],tcount[i]);
		 }
		 if(tmp_buf != bufptr)
		   free(tmp_buf);
	       }
	     }
	   }else if(ios->io_rank < iodesc->num_aiotasks ){
	     buflen=1;
	     for(i=0;i<ndims;i++){
	       tstart[i] = (size_t) start[i];
	       tcount[i] = (size_t) count[i];
	       buflen*=tcount[i];
	       //               printf("%s %d %d %d %d\n",__FILE__,__LINE__,i,tstart[i],tcount[i]);
	     }
	     //	     printf("%s %d %d %d %d %d %d %d %d %d\n",__FILE__,__LINE__,ios->io_rank,tstart[0],tstart[1],tcount[0],tcount[1],buflen,ndims,fndims);
	     mpierr = MPI_Recv( &ierr, 1, MPI_INT, 0, 0, ios->io_comm, &status);  // task0 is ready to recieve
	     mpierr = MPI_Rsend( &buflen, 1, MPI_INT, 0, 1, ios->io_comm);
	     if(buflen>0) {
	       mpierr = MPI_Rsend( tstart, ndims, MPI_OFFSET, 0, ios->num_iotasks+ios->io_rank, ios->io_comm);
	       mpierr = MPI_Rsend( tcount, ndims, MPI_OFFSET, 0,2*ios->num_iotasks+ios->io_rank, ios->io_comm);
	       mpierr = MPI_Rsend( bufptr, buflen, iodesc->basetype, 0, ios->io_rank, ios->io_comm);
	     }
	   }
	   break;
	 }
	 break;
 #endif
 #ifdef _PNETCDF
       case PIO_IOTYPE_PNETCDF:
	 for( i=0,dsize=1;i<ndims;i++){
	   dsize*=count[i];
	 }
	 tdsize += dsize;
	 //	 if(dsize==1 && ndims==2)
	 //	 printf("%s %d %d\n",__FILE__,__LINE__,iodesc->basetype);

	 /*	 if(regioncnt==0){
	   for(i=0;i<iodesc->maxregions;i++){
	     startlist[i] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	     countlist[i] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	   }
	 }
	 */
	 if(dsize>0){
	   //	   printf("%s %d %d %d\n",__FILE__,__LINE__,ios->io_rank,dsize);
	   startlist[rrcnt] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	   countlist[rrcnt] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	   for( i=0; i<fndims;i++){
	     startlist[rrcnt][i]=start[i];
	     countlist[rrcnt][i]=count[i];
	   }
	   rrcnt++;
	 }	 
	 if(regioncnt==iodesc->maxregions-1){
	   // printf("%s %d %d %ld %ld\n",__FILE__,__LINE__,ios->io_rank,iodesc->llen, tdsize);
	   //	   ierr = ncmpi_put_varn_all(ncid, vid, iodesc->maxregions, startlist, countlist, 
	   //			     IOBUF, iodesc->llen, iodesc->basetype);
	   
	   ierr = ncmpi_bput_varn(ncid, vid, rrcnt, startlist, countlist, 
				  IOBUF, iodesc->llen, iodesc->basetype, &(vdesc->request));

	   for(i=0;i<rrcnt;i++){
	     free(startlist[i]);
	     free(countlist[i]);
	   }
	 }
	 break;
#endif
       default:
	 ierr = iotype_error(file->iotype,__FILE__,__LINE__);
       }
       if(region != NULL)
	 region = region->next;
     } //    for(regioncnt=0;regioncnt<iodesc->maxregions;regioncnt++){
   } // if(ios->ioproc)

   ierr = check_netcdf(file, ierr, __FILE__,__LINE__);
#ifdef TIMING
  GPTLstop("PIO:write_darray_nc");
#endif

   return ierr;
 }

int pio_write_darray_multi_nc(file_desc_t *file, const int nvars, const int vid[], 
			      const int iodesc_ndims, MPI_Datatype basetype, const PIO_Offset gsize[],
			      const int maxregions, io_region *firstregion, const PIO_Offset llen,
			      const int maxiobuflen, const int num_aiotasks,
			      void *IOBUF, const int frame[])
 {
   iosystem_desc_t *ios;
   var_desc_t *vdesc;
   int ierr;
   int i;
   int msg;
   int mpierr;
   int dsize;
   MPI_Status status;
   PIO_Offset usage;
   int fndims;
   PIO_Offset tdsize;
   int tsize;
   int ncid;
   tdsize=0;
   ierr = PIO_NOERR;
#ifdef TIMING
  GPTLstart("PIO:write_darray_multi_nc");
#endif

   ios = file->iosystem;
   if(ios == NULL){
     fprintf(stderr,"Failed to find iosystem handle \n");
     return PIO_EBADID;
   }
   vdesc = (file->varlist)+vid[0];
   ncid = file->fh;

   if(vdesc == NULL){
     fprintf(stderr,"Failed to find variable handle %d\n",vid[0]);
     return PIO_EBADID;
   }
   msg = 0;

   if(ios->async_interface && ! ios->ioproc){
     if(ios->comp_rank==0) 
       mpierr = MPI_Send(&msg, 1,MPI_INT, ios->ioroot, 1, ios->union_comm);
     mpierr = MPI_Bcast(&(file->fh),1, MPI_INT, ios->compmaster, ios->intercomm);
   }

   ierr = PIOc_inq_varndims(file->fh, vid[0], &fndims);
   MPI_Type_size(basetype, &tsize);

   if(ios->ioproc){
     io_region *region;
     int regioncnt;
     int rrcnt;
     void *bufptr;
     int buflen, j;
     size_t start[fndims];
     size_t count[fndims];
     int ndims = iodesc_ndims;

     PIO_Offset *startlist[maxregions];
     PIO_Offset *countlist[maxregions];

     ncid = file->fh;
     region = firstregion;

     if(vdesc->record >= 0 && ndims<fndims){
       ndims++;
     }
     rrcnt=0;
     for(regioncnt=0;regioncnt<maxregions;regioncnt++){
       for(i=0;i<ndims;i++){
	 start[i] = 0;
	 count[i] = 0;
       }
       if(region != NULL){
	 // this is a record based multidimensional array
	 if(vdesc->record >= 0){
	   start[0] = frame[0];
	   for(i=1;i<ndims;i++){
	     start[i] = region->start[i-1];
	     count[i] = region->count[i-1];
	   }
	  if(count[1]>0)
	    count[0] = 1;
	 // Non-time dependent array
	 }else{
	   for( i=0;i<ndims;i++){
	     start[i] = region->start[i];
	     count[i] = region->count[i];
	   }
	 }
       }

       switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
       case PIO_IOTYPE_NETCDF4P:
	 for(int nv=0; nv<nvars; nv++){
	   if(vdesc->record >= 0){
	     start[0] = frame[nv];
	   }
	   if(region != NULL){
	     bufptr = (void *)((char *) IOBUF + tsize*(nv*llen + region->loffset));
	   }
	   ierr = nc_var_par_access(ncid, vid[nv], NC_COLLECTIVE);

	   if(basetype == MPI_DOUBLE ||basetype == MPI_REAL8){
	     ierr = nc_put_vara_double (ncid, vid[nv],(size_t *) start,(size_t *) count, (const double *) bufptr); 
	   } else if(basetype == MPI_INTEGER){
	     ierr = nc_put_vara_int (ncid, vid[nv], (size_t *) start, (size_t *) count, (const int *) bufptr); 
	   }else if(basetype == MPI_FLOAT || basetype == MPI_REAL4){
	     ierr = nc_put_vara_float (ncid, vid[nv], (size_t *) start, (size_t *) count, (const float *) bufptr); 
	   }else{
	     fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) basetype);
	   }
	 }
	 break;
       case PIO_IOTYPE_NETCDF4C:
#endif
       case PIO_IOTYPE_NETCDF:
	 {
	   mpierr = MPI_Type_size(basetype, &dsize);
	   size_t tstart[ndims], tcount[ndims];
	   if(ios->io_rank==0){
	     for(int iorank=0;iorank<num_aiotasks;iorank++){
	       if(iorank==0){	    
	         buflen=1;
		 for(j=0;j<ndims;j++){
		   tstart[j] =  start[j];
		   tcount[j] =  count[j];
		   buflen *= tcount[j];
		 }
	       }else{
		 mpierr = MPI_Send( &ierr, 1, MPI_INT, iorank, 0, ios->io_comm);  // handshake - tell the sending task I'm ready
		 mpierr = MPI_Recv( &buflen, 1, MPI_INT, iorank, 1, ios->io_comm, &status);
		 if(buflen>0){
		   mpierr = MPI_Recv( tstart, ndims, MPI_OFFSET, iorank, ios->num_iotasks+iorank, ios->io_comm, &status);
		   mpierr = MPI_Recv( tcount, ndims, MPI_OFFSET, iorank,2*ios->num_iotasks+iorank, ios->io_comm, &status);
		 }
	       }

	       if(buflen>0){
  	         for(int nv=0; nv<nvars; nv++){
		   if(vdesc->record >= 0){
		     tstart[0] = frame[nv];
		   }

		   //		   printf("%s %d %d %d %d %d %ld\n",__FILE__,__LINE__,iodesc->llen, buflen, iodesc->maxiobuflen, nvars, tcount[0]);


		   if(iorank>0){
		     bufptr = malloc(buflen*tsize) ;
		     mpierr = MPI_Recv( bufptr, buflen, basetype, iorank, iorank, ios->io_comm, &status);
		   }else{
		     bufptr = (void *)((char *) IOBUF+ tsize*(nv*llen + region->loffset));
		   }
		   //		   printf("%s %d %d %X %X\n",__FILE__,__LINE__,vid[nv],IOBUF,bufptr);
		   if(basetype == MPI_INTEGER){
		     ierr = nc_put_vara_int (ncid, vid[nv], tstart, tcount, (const int *) bufptr); 
		   }else if(basetype == MPI_DOUBLE || basetype == MPI_REAL8){
		     ierr = nc_put_vara_double (ncid, vid[nv], tstart, tcount, (const double *) bufptr); 
		   }else if(basetype == MPI_FLOAT || basetype == MPI_REAL4){
		     ierr = nc_put_vara_float (ncid,vid[nv], tstart, tcount, (const float *) bufptr); 
		   }else{
		     fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) basetype);
		   }
		   if(iorank>0){
		     free(bufptr);
		   }
		   if(ierr != PIO_NOERR){
		     for(i=0;i<fndims;i++)
		       fprintf(stderr,"vid %d dim %d start %ld count %ld\n",vid[nv],i,tstart[i],tcount[i]);
		   }
	         }
	       }
	     }
	   }else if(ios->io_rank < num_aiotasks ){
	     buflen=1;
	     for(i=0;i<ndims;i++){
	       tstart[i] = (size_t) start[i];
	       tcount[i] = (size_t) count[i];
	       buflen*=tcount[i];
	       //	                      printf("%s %d %d %d %d\n",__FILE__,__LINE__,i,tstart[i],tcount[i]);
	     }
	     //	     printf("%s %d %d %d %d %d %d %d %d %d\n",__FILE__,__LINE__,ios->io_rank,tstart[0],tstart[1],tcount[0],tcount[1],buflen,ndims,fndims);
	     mpierr = MPI_Recv( &ierr, 1, MPI_INT, 0, 0, ios->io_comm, &status);  // task0 is ready to recieve
	     mpierr = MPI_Rsend( &buflen, 1, MPI_INT, 0, 1, ios->io_comm);
	     if(buflen>0) {
	       mpierr = MPI_Rsend( tstart, ndims, MPI_OFFSET, 0, ios->num_iotasks+ios->io_rank, ios->io_comm);
	       mpierr = MPI_Rsend( tcount, ndims, MPI_OFFSET, 0,2*ios->num_iotasks+ios->io_rank, ios->io_comm);

	       for(int nv=0; nv<nvars; nv++){
       	         bufptr = (void *)((char *) IOBUF + tsize*(nv*llen + region->loffset));
		 //		 printf("%d %d %d\n",__LINE__,iodesc->llen,region->loffset);
		 // printf("%d %d %d %d %d %d\n",__LINE__,((int *)bufptr)[0],((int *)bufptr)[1],((int *)bufptr)[2],((int *)bufptr)[3],((int *)bufptr)[4]);
	         mpierr = MPI_Send( bufptr, buflen, basetype, 0, ios->io_rank, ios->io_comm);
	       }
	     }
	   }
	   break;
	 }
	 break;
#endif
#ifdef _PNETCDF
       case PIO_IOTYPE_PNETCDF:
	 for( i=0,dsize=1;i<ndims;i++){
	   dsize*=count[i];
	 }
	 tdsize += dsize;

	 if(dsize>0){
	   //	   printf("%s %d %d %d\n",__FILE__,__LINE__,ios->io_rank,dsize);
	   startlist[rrcnt] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	   countlist[rrcnt] = (PIO_Offset *) calloc(fndims, sizeof(PIO_Offset));
	   for( i=0; i<fndims;i++){
	     startlist[rrcnt][i]=start[i];
	     countlist[rrcnt][i]=count[i];
	   }
	   rrcnt++;
	 }	 
	 if(regioncnt==maxregions-1){
	   //printf("%s %d %d %ld %ld\n",__FILE__,__LINE__,ios->io_rank,iodesc->llen, tdsize);
	   //	   ierr = ncmpi_put_varn_all(ncid, vid, iodesc->maxregions, startlist, countlist, 
	   //			     IOBUF, iodesc->llen, iodesc->basetype);
	   
	   //printf("%s %d %ld \n",__FILE__,__LINE__,IOBUF);
	   for(int nv=0; nv<nvars; nv++){
	     vdesc = (file->varlist)+vid[nv];
	     if(vdesc->record >= 0){
	       for(int rc=0;rc<rrcnt;rc++){
		 startlist[rc][0] = frame[nv];
	       }
	     }
	     bufptr = (void *)((char *) IOBUF + nv*tsize*llen);
	     /*
	     ierr = ncmpi_iput_varn(ncid, vid[nv], rrcnt, startlist, countlist, 
				    bufptr, llen, basetype, &(vdesc->request));
	     */
	     
	     
	     ierr = ncmpi_bput_varn(ncid, vid[nv], rrcnt, startlist, countlist, 
				    bufptr, llen, basetype, &(vdesc->request));
	     
	     //	     printf("%s %d %d %d\n",__FILE__,__LINE__,vid[nv],vdesc->request, llen);
	     if(llen == 0){
	       vdesc->request = PIO_REQ_NULL;
	     }

	     
	   }
	   for(i=0;i<rrcnt;i++){
             //printf("%d %ld %ld %ld %ld\n",i,startlist[i][0],startlist[i][1],countlist[i][0],countlist[i][1]);
	     free(startlist[i]);
	     free(countlist[i]);
	   }
	 }
	 break;
#endif
       default:
	 ierr = iotype_error(file->iotype,__FILE__,__LINE__);
       }
       if(region != NULL)
	 region = region->next;
     } //    for(regioncnt=0;regioncnt<iodesc->maxregions;regioncnt++){
   } // if(ios->ioproc)

   ierr = check_netcdf(file, ierr, __FILE__,__LINE__);
#ifdef TIMING
  GPTLstop("PIO:write_darray_multi_nc");
#endif

   return ierr;
 }

int PIOc_write_darray_multi(const int ncid, const int vid[], const int ioid, const int nvars, const PIO_Offset arraylen, void *array, const int frame[], void *fillvalue[])
 {
   iosystem_desc_t *ios;
   file_desc_t *file;
   io_desc_t *iodesc;
   var_desc_t *vdesc;
   void *iobuf;
   int vsize, rlen;
   int ierr;
   void *fillbuf = NULL;

   ierr = PIO_NOERR;

   file = pio_get_file_from_id(ncid);
   if(file == NULL){
     fprintf(stderr,"File handle not found %d %d\n",ncid,__LINE__);
     return PIO_EBADID;
   }
   if(! (file->mode & PIO_WRITE)){
     fprintf(stderr,"ERROR:  Attempt to write to read-only file\n");
     return PIO_EPERM;
   }
       
   iodesc = pio_get_iodesc_from_id(ioid);
   if(iodesc == NULL){
     print_trace(NULL);
     fprintf(stderr,"iodesc handle not found %d %d\n",ioid,__LINE__);
     return PIO_EBADID;
   }
   iobuf = NULL;

   pioassert(nvars>0,"nvars <= 0",__FILE__,__LINE__);

   ios = file->iosystem;
   //   rlen = iodesc->llen*nvars;
   rlen=0;
   if(iodesc->llen>0){
     rlen = iodesc->maxiobuflen*nvars;
   }
   if(iodesc->rearranger>0){
     if(rlen>0){
       //       printf("rlen = %ld\n",rlen);
       MPI_Type_size(iodesc->basetype, &vsize);	
       iobuf = malloc((size_t) vsize* (size_t) rlen);
       //       iobuf = bget((size_t) vsize* (size_t) rlen);
       if(iobuf==NULL){
	 printf("%s %d %d %ld\n",__FILE__,__LINE__,nvars,vsize*rlen);
	 piomemerror(*ios,(size_t) rlen*(size_t) vsize, __FILE__,__LINE__);
       }
       if(iodesc->needsfill && iodesc->rearranger==PIO_REARR_BOX){
	 if(vsize==4){
	   for(int nv=0;nv < nvars; nv++){
	     for(int i=0;i<iodesc->maxiobuflen;i++){
	       ((float *)iobuf)[i+nv*(iodesc->maxiobuflen)] = ((float *)fillvalue)[nv];
	     }
	   }
	 }else if(vsize==8){
	   for(int nv=0;nv < nvars; nv++){
	     for(int i=0;i<iodesc->maxiobuflen;i++){
	       ((double *)iobuf)[i+nv*(iodesc->maxiobuflen)] = ((double *)fillvalue)[nv];
	     }
	   }
	 }
       }
/*

*/
     }
     //    printf("%s %d rlen = %d 0x%.16X 0x%.16X\n",__FILE__,__LINE__,rlen,array,iobuf); 

     //  }
     /*
     if(vsize==8){
       double asum;
       for(int nv=0;nv<nvars; nv++){
	 asum=0.0;
	 for(int k=0;k<iodesc->ndof;k++){
	   asum += ((double *) array)[k+nv*iodesc->ndof];
	 }
	 printf("%s %d %d %g\n",__FILE__,__LINE__,nv,asum);
       }
     }
     
     */
     
     ierr = rearrange_comp2io(*ios, iodesc, array, iobuf, nvars);

   }else{
     iobuf = array;
   }
   switch(file->iotype){
   case PIO_IOTYPE_PNETCDF:
   case PIO_IOTYPE_NETCDF:
   case PIO_IOTYPE_NETCDF4P:
   case PIO_IOTYPE_NETCDF4C:
     //     printf("%s %d %d %d %d\n",__FILE__,__LINE__,iodesc->holegridsize,vsize,nvars);
     if(iodesc->rearranger == PIO_REARR_SUBSET && iodesc->needsfill &&
	iodesc->holegridsize>0){
       fillbuf = bget(iodesc->holegridsize*vsize*nvars);
       if(vsize==4){
	 for(int nv=0;nv<nvars;nv++){
	   for(int i=0;i<iodesc->holegridsize;i++){
	     ((float *) fillbuf)[i+nv*iodesc->holegridsize] = ((float *) fillvalue)[nv];
	   }
	 
	   //         printf("%s %d %d %d\n",__FILE__,__LINE__,nv,((int *)fillbuf)[0]);
          }
       }else if(vsize==8){
	 for(int nv=0;nv<nvars;nv++){
	   for(int i=0;i<iodesc->holegridsize;i++){
	     ((double *) fillbuf)[i+nv*iodesc->holegridsize] = ((double *) fillvalue)[nv];
	   }
	 }
       }
       //         printf("%s %d fill %d \n",__FILE__,__LINE__,((int *)fillbuf)[0]);
       ierr = pio_write_darray_multi_nc(file, nvars, vid,  
					iodesc->ndims, iodesc->basetype, iodesc->gsize,
					iodesc->maxfillregions, iodesc->fillregion, iodesc->holegridsize,
					iodesc->holegridsize, iodesc->num_aiotasks,
					fillbuf, frame);
       for(int nv=0;nv<nvars;nv++){
	 vdesc = file->varlist+vid[nv];
	 vdesc->fillrequest = vdesc->request;
       }
     }

     ierr = pio_write_darray_multi_nc(file, nvars, vid,  
				      iodesc->ndims, iodesc->basetype, iodesc->gsize,
				      iodesc->maxregions, iodesc->firstregion, iodesc->llen,
				      iodesc->maxiobuflen, iodesc->num_aiotasks,
				      iobuf, frame);
     
   }
   
   /* We cannot free the iobuf and the fillbuf until the flush completes */
   flush_output_buffer(file, false, 0);

   //printf("%s %d %ld\n",__FILE__,__LINE__,iobuf);
   if(iobuf != NULL && iobuf != array){
     //     printf("%s %d 0x%.16X\n",__FILE__,__LINE__,iobuf);
     free(iobuf);
     //     brel(iobuf);
   }
   if(fillbuf != NULL){
       brel(fillbuf);
   }
   return ierr;

 }
#ifdef PIO_WRITE_BUFFERING
 int PIOc_write_darray(const int ncid, const int vid, const int ioid, const PIO_Offset arraylen, void *array, void *fillvalue)
 {
   iosystem_desc_t *ios;
   file_desc_t *file;
   io_desc_t *iodesc;
   var_desc_t *vdesc;
   void *bufptr;
   size_t rlen;
   int ierr;
   MPI_Datatype vtype;
   wmulti_buffer *wmb;
   int tsize;
   int *tptr;
   void *bptr;
   void *fptr;
   bool recordvar;
   bufsize totfree, maxfree;

   ierr = PIO_NOERR;

   file = pio_get_file_from_id(ncid);
   if(file == NULL){
     fprintf(stderr,"File handle not found %d %d\n",ncid,__LINE__);
     return PIO_EBADID;
   }
   if(! (file->mode & PIO_WRITE)){
     fprintf(stderr,"ERROR:  Attempt to write to read-only file\n");
     return PIO_EPERM;
   }
      
   iodesc = pio_get_iodesc_from_id(ioid);
   if(iodesc == NULL){
     fprintf(stderr,"iodesc handle not found %d %d\n",ioid,__LINE__);
     return PIO_EBADID;
   }
   ios = file->iosystem;

  vdesc = (file->varlist)+vid;
  if(vdesc == NULL)
    return PIO_EBADID;

  if(vdesc->record<0){
    recordvar=false;
  }else{
    recordvar=true;
  }
  if(iodesc->ndof != arraylen){
    fprintf(stderr,"ndof=%ld, arraylen=%ld\n",iodesc->ndof,arraylen);
    piodie("ndof != arraylen",__FILE__,__LINE__);
  }
   wmb = &(file->buffer);
   if(wmb->ioid == -1){
     if(recordvar){
       wmb->ioid = ioid;
     }else{
       wmb->ioid = -(ioid);
     }
   }else{
     // separate record and non-record variables
     if(recordvar){
       while(wmb->next != NULL && wmb->ioid!=ioid){
	 if(wmb->next!=NULL)
	   wmb = wmb->next;
       }
       /* flush the previous record before starting a new one. this is collective */
    //   if(wmb->frame != NULL && vdesc->record != wmb->frame[0]){
     //    if(ios->iomaster) printf("%s %d\n",__FILE__,__LINE__);
	// flush_buffer(ncid,wmb);
    //   }
     }else{
       while(wmb->next != NULL && wmb->ioid!= -(ioid)){
	 if(wmb->next!=NULL)
	   wmb = wmb->next;
       }
     }
   }
   if((recordvar && wmb->ioid != ioid) || (!recordvar && wmb->ioid != -(ioid))){
     wmb->next = (wmulti_buffer *) bget((bufsize) sizeof(wmulti_buffer));
     if(wmb->next == NULL){
       piomemerror(*ios,sizeof(wmulti_buffer), __FILE__,__LINE__);
     }
     wmb=wmb->next;
     wmb->next=NULL;
     if(recordvar){
       wmb->ioid = ioid;
     }else{
       wmb->ioid = -(ioid);
     }
     wmb->validvars=0;
     wmb->arraylen=arraylen;
     wmb->vid=NULL;
     wmb->data=NULL;
     wmb->frame=NULL;
     wmb->fillvalue=NULL;
   }

 
    MPI_Type_size(iodesc->basetype, &tsize);
  // At this point wmb should be pointing to a new or existing buffer 
   // so we can add the data
    //     printf("%s %d %X %d %d %d\n",__FILE__,__LINE__,wmb->data,wmb->validvars,arraylen,tsize);
    //    cn_buffer_report(*ios, true);
    bfreespace(&totfree, &maxfree);
    bool needflush = (maxfree <= 1.1*(1+wmb->validvars)*arraylen*tsize );
    MPI_Allreduce(MPI_IN_PLACE, &needflush, 1,  MPI_INT,  MPI_MAX, ios->comp_comm);


    if(needflush ){
      // need to flush first
          printf("%s %d %ld %d %ld %ld\n",__FILE__,__LINE__,maxfree, wmb->validvars, (1+wmb->validvars)*arraylen*tsize,totfree);
            cn_buffer_report(*ios, true);
	
      flush_buffer(ncid,wmb);
    }
    if(arraylen > 0){
      wmb->data = bgetr( wmb->data, (1+wmb->validvars)*arraylen*tsize);
      if(wmb->data == NULL){
	piomemerror(*ios, (1+wmb->validvars)*arraylen*tsize  , __FILE__,__LINE__);
      }
    }
    wmb->vid = (int *) bgetr( wmb->vid,sizeof(int)*( 1+wmb->validvars));
    if(wmb->vid == NULL){
      piomemerror(*ios, (1+wmb->validvars)*sizeof(int)  , __FILE__,__LINE__);
    }
    if(vdesc->record>=0){
      wmb->frame = (int *) bgetr( wmb->frame,sizeof(int)*( 1+wmb->validvars));
      if(wmb->frame == NULL){
	piomemerror(*ios, (1+wmb->validvars)*sizeof(int)  , __FILE__,__LINE__);
      }
    }
    if(iodesc->needsfill){
      wmb->fillvalue = bgetr( wmb->fillvalue,tsize*( 1+wmb->validvars));
      if(wmb->fillvalue == NULL){
	piomemerror(*ios, (1+wmb->validvars)*tsize  , __FILE__,__LINE__);
      }
    }
 

   if(iodesc->needsfill){
     if(fillvalue != NULL){
       memcpy((char *) wmb->fillvalue+tsize*wmb->validvars,fillvalue, tsize); 
     }else{
       vtype = (MPI_Datatype) iodesc->basetype;
       if(vtype == MPI_INTEGER){
	 int fill = PIO_FILL_INT;
	 memcpy((char *) wmb->fillvalue+tsize*wmb->validvars, &fill, tsize); 	     
       }else if(vtype == MPI_FLOAT || vtype == MPI_REAL4){
	 float fill = PIO_FILL_FLOAT;
	 memcpy((char *) wmb->fillvalue+tsize*wmb->validvars, &fill, tsize); 
       }else if(vtype == MPI_DOUBLE || vtype == MPI_REAL8){
	 double fill = PIO_FILL_DOUBLE;
	 memcpy((char *) wmb->fillvalue+tsize*wmb->validvars, &fill, tsize); 
       }else if(vtype == MPI_CHARACTER){
	 char fill = PIO_FILL_CHAR;
	 memcpy((char *) wmb->fillvalue+tsize*wmb->validvars, &fill, tsize); 
       }else{
	 fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",vtype);
       }
    }

   }
 
   wmb->arraylen = arraylen;
   wmb->vid[wmb->validvars]=vid;
   bufptr = (void *)((char *) wmb->data + arraylen*tsize*wmb->validvars);
   if(arraylen>0){
     memcpy(bufptr, array, arraylen*tsize);
   }
   /*
   if(tsize==8){
     double asum=0.0;
     printf("%s %d %d %d %d\n",__FILE__,__LINE__,vid,arraylen,iodesc->ndof);
     for(int k=0;k<arraylen;k++){
       asum += ((double *) array)[k];
     }
     printf("%s %d %d %g\n",__FILE__,__LINE__,vid,asum);
   }
   */

   //   printf("%s %d %d %d %d %X\n",__FILE__,__LINE__,wmb->validvars,wmb->ioid,vid,bufptr);

   if(wmb->frame!=NULL){
     wmb->frame[wmb->validvars]=vdesc->record;
   }
   wmb->validvars++;

			      //   printf("%s %d %d %d %d %d\n",__FILE__,__LINE__,wmb->validvars,iodesc->maxbytes/tsize, iodesc->ndof, iodesc->llen);
   if(wmb->validvars >= iodesc->maxbytes/tsize){
     PIOc_sync(ncid);
   }

   return ierr;

 }
#else
 int PIOc_write_darray(const int ncid, const int vid, const int ioid, const PIO_Offset arraylen, void *array, void *fillvalue)
 {
   iosystem_desc_t *ios;
   file_desc_t *file;
   io_desc_t *iodesc;
   void *iobuf;
   size_t  rlen;
   int tsize;
   int ierr;
   MPI_Datatype vtype;

   ierr = PIO_NOERR;


   file = pio_get_file_from_id(ncid);
   if(file == NULL){
     fprintf(stderr,"File handle not found %d %d\n",ncid,__LINE__);
     return PIO_EBADID;
   }
   iodesc = pio_get_iodesc_from_id(ioid);
   if(iodesc == NULL){
     fprintf(stderr,"iodesc handle not found %d %d\n",ioid,__LINE__);
     return PIO_EBADID;
   }
   iobuf = NULL;

   ios = file->iosystem;

   rlen = iodesc->llen;
   if(iodesc->rearranger>0){
     if(rlen>0){
       MPI_Type_size(iodesc->basetype, &tsize);	
       //       iobuf = bget(tsize*rlen);
       iobuf = malloc((size_t) tsize*rlen);
       if(iobuf==NULL){
	 piomemerror(*ios,rlen*(size_t) tsize, __FILE__,__LINE__);
       }
     }
     //    printf(" rlen = %d %ld\n",rlen,iobuf); 

     //  }


     ierr = rearrange_comp2io(*ios, iodesc, array, iobuf, 1);


   }else{
     iobuf = array;
   }
   switch(file->iotype){
   case PIO_IOTYPE_PNETCDF:
   case PIO_IOTYPE_NETCDF:
   case PIO_IOTYPE_NETCDF4P:
   case PIO_IOTYPE_NETCDF4C:
     ierr = pio_write_darray_nc(file, iodesc, vid, iobuf, fillvalue);
   }

   if(iodesc->rearranger>0 && rlen>0)
     free(iobuf);

   return ierr;

 }
#endif



int pio_read_darray_nc(file_desc_t *file, io_desc_t *iodesc, const int vid, void *IOBUF)
{
  int ierr=PIO_NOERR;
  iosystem_desc_t *ios;
  var_desc_t *vdesc;
  int ndims, fndims;
  MPI_Status status;
  int i;

#ifdef TIMING
  GPTLstart("PIO:read_darray_nc");
#endif
  ios = file->iosystem;
  if(ios == NULL)
    return PIO_EBADID;
  
  vdesc = (file->varlist)+vid;
  
  if(vdesc == NULL)
    return PIO_EBADID;
  
  ndims = iodesc->ndims;
  ierr = PIOc_inq_varndims(file->fh, vid, &fndims);
  if(fndims==ndims) 
    vdesc->record=-1;
  
  if(ios->ioproc){
    io_region *region;
    size_t start[fndims];
    size_t count[fndims];
    size_t tmp_start[fndims];
    size_t tmp_count[fndims];
    size_t tmp_bufsize=1;
    int regioncnt;
    void *bufptr;
    int tsize;

    int rrlen=0;
    PIO_Offset *startlist[iodesc->maxregions];
    PIO_Offset *countlist[iodesc->maxregions];

    // buffer is incremented by byte and loffset is in terms of the iodessc->basetype
    // so we need to multiply by the size of the basetype
    // We can potentially allow for one iodesc to have multiple datatypes by allowing the
    // calling program to change the basetype.   
    region = iodesc->firstregion;
    MPI_Type_size(iodesc->basetype, &tsize);
    if(fndims>ndims){
      ndims++;
      if(vdesc->record<0) 
	vdesc->record=0;
    }
    for(regioncnt=0;regioncnt<iodesc->maxregions;regioncnt++){
         //         printf("%s %d %d %ld %d %d\n",__FILE__,__LINE__,regioncnt,region,fndims,ndims);
      tmp_bufsize=1;
      if(region==NULL || iodesc->llen==0){
	for(i=0;i<fndims;i++){
	  start[i] = 0;
	  count[i] = 0;
	}
	bufptr=NULL;
      }else{       
	if(regioncnt==0 || region==NULL)
	  bufptr = IOBUF;
	else
	  bufptr=(void *)((char *) IOBUF + tsize*region->loffset);
	 
	//		printf("%s %d %d %d %d\n",__FILE__,__LINE__,iodesc->llen - region->loffset, iodesc->llen, region->loffset);
	
	if(vdesc->record >= 0 && fndims>1){
	  start[0] = vdesc->record;
	  for(i=1;i<ndims;i++){
	    start[i] = region->start[i-1];
	    count[i] = region->count[i-1];
	    //	    printf("%s %d %d %ld %ld\n",__FILE__,__LINE__,i,start[i],count[i]); 
	   } 
	  if(count[1]>0)
	    count[0] = 1;
	}else{
	  // Non-time dependent array
	  for(i=0;i<ndims;i++){
	    start[i] = region->start[i];
	    count[i] = region->count[i];
	     // printf("%s %d %d %ld %ld\n",__FILE__,__LINE__,i,start[i],count[i]); 
	  }
	}
      }
       
      switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
      case PIO_IOTYPE_NETCDF4P:
	 if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
	   ierr = nc_get_vara_double (file->fh, vid,start,count, bufptr); 
	 } else if(iodesc->basetype == MPI_INTEGER){
	   ierr = nc_get_vara_int (file->fh, vid, start, count,  bufptr); 
	 }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
	   ierr = nc_get_vara_float (file->fh, vid, start,  count,  bufptr); 
	 }else{
	   fprintf(stderr,"Type not recognized %d in pioc_read_darray\n",(int) iodesc->basetype);
	 }
	break;
      case PIO_IOTYPE_NETCDF4C:
#endif
      case PIO_IOTYPE_NETCDF:
	if(ios->io_rank>0){
	  tmp_bufsize=1;
	  for( i=0;i<fndims; i++){
	    tmp_start[i] = start[i];
	    tmp_count[i] = count[i];
	    tmp_bufsize *= count[i];
	  }
	  MPI_Send( tmp_count, ndims, MPI_OFFSET, 0, ios->io_rank, ios->io_comm);
	  if(tmp_bufsize > 0){
	    MPI_Send( tmp_start, ndims, MPI_OFFSET, 0, ios->io_rank, ios->io_comm);
	    //	    printf("%s %d %d\n",__FILE__,__LINE__,tmp_bufsize);
	    MPI_Recv( bufptr, tmp_bufsize, iodesc->basetype, 0, ios->io_rank, ios->io_comm, &status);
	  }
	  //	  printf("%s %d %d %d %d %d %d %d\n",__FILE__,__LINE__,regioncnt,tmp_start[1],tmp_start[2],tmp_count[1],tmp_count[2], ndims);
	}else if(ios->io_rank==0){
	  for( i=ios->num_iotasks-1; i>=0; i--){
	    if(i==0){
	      for(int k=0;k<fndims;k++)
		tmp_count[k] = count[k];
	      if(regioncnt==0 || region==NULL)
		bufptr = IOBUF;
	      else
		bufptr=(void *)((char *) IOBUF + tsize*region->loffset);
	    }else{
	      MPI_Recv(tmp_count, ndims, MPI_OFFSET, i, i, ios->io_comm, &status);
	    }
	    tmp_bufsize=1;
	    for(int j=0;j<fndims; j++){
	      tmp_bufsize *= tmp_count[j];
	    }
	    //	    printf("%s %d %d %d\n",__FILE__,__LINE__,i,tmp_bufsize);
	    if(tmp_bufsize>0){
	      if(i==0){
		for(int k=0;k<fndims;k++)
		  tmp_start[k] = start[k]; 
	      }else{
		MPI_Recv(tmp_start, ndims, MPI_OFFSET, i, i, ios->io_comm, &status);
	      }		
	      if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
		if(i>0)
		  bufptr = malloc(tmp_bufsize *sizeof(double));
		ierr = nc_get_vara_double (file->fh, vid, tmp_start, tmp_count, bufptr); 
	      }else if(iodesc->basetype == MPI_INTEGER){
		if(i>0)
		  bufptr = malloc(tmp_bufsize *sizeof(int));
		ierr = nc_get_vara_int (file->fh, vid, tmp_start, tmp_count,  bufptr); 	     
	      }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
		if(i>0)
		  bufptr = malloc(tmp_bufsize *sizeof(float));
		ierr = nc_get_vara_float (file->fh, vid, tmp_start, tmp_count,  bufptr); 
	      }else{
		fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
	      }	
	      
	      if(ierr != PIO_NOERR){
		printf("%s %d ",__FILE__,__LINE__);
		for(int j=0;j<fndims;j++)
		  printf(" %ld %ld",tmp_start[j],tmp_count[j]);
		printf("\n");
	      }
	      
	      if(i>0){
		//    printf("%s %d %d %d\n",__FILE__,__LINE__,i,tmp_bufsize);
		MPI_Rsend(bufptr, tmp_bufsize, iodesc->basetype, i, i, ios->io_comm);
		free(bufptr);
	      }
	    }
	  }
	}
	break;
#endif
#ifdef _PNETCDF
      case PIO_IOTYPE_PNETCDF:
	{
	  tmp_bufsize=1;
	  for(int j=0;j<fndims; j++){
	    tmp_bufsize *= count[j];
	  }

	  if(tmp_bufsize>0){
             startlist[rrlen] = (PIO_Offset *) malloc(fndims * sizeof(PIO_Offset));
             countlist[rrlen] = (PIO_Offset *) malloc(fndims * sizeof(PIO_Offset));

	    for(int j=0;j<fndims; j++){
	      startlist[rrlen][j] = start[j];
	      countlist[rrlen][j] = count[j];
	      //	      	      printf("%s %d %d %d %ld %ld %ld\n",__FILE__,__LINE__,iodesc->maxregions, j,start[j],count[j],tmp_bufsize);
	    }
            rrlen++;
	  }
	  if(regioncnt==iodesc->maxregions-1){
	    ierr = ncmpi_get_varn_all(file->fh, vid, rrlen, startlist, 
				      countlist, IOBUF, iodesc->llen, iodesc->basetype);
	    for(i=0;i<rrlen;i++){
	      free(startlist[i]);
	      free(countlist[i]);
	    }
	  }
	}
	break;
#endif
      default:
	ierr = iotype_error(file->iotype,__FILE__,__LINE__);
	 
      }
      if(region != NULL)
	region = region->next;
    } // for(regioncnt=0;...)
  }
  
  ierr = check_netcdf(file, ierr, __FILE__,__LINE__);
#ifdef TIMING
  GPTLstop("PIO:read_darray_nc");
#endif

  return ierr;
}

int PIOc_read_darray(const int ncid, const int vid, const int ioid, const PIO_Offset arraylen, void *array)
{
  iosystem_desc_t *ios;
  file_desc_t *file;
  io_desc_t *iodesc;
  void *iobuf=NULL;
  size_t rlen=0;
  int ierr, tsize;
  MPI_Datatype vtype;
  void *fillval;

  file = pio_get_file_from_id(ncid);

  if(file == NULL){
    fprintf(stderr,"File handle not found %d %d\n",ncid,__LINE__);
    return PIO_EBADID;
  }
  iodesc = pio_get_iodesc_from_id(ioid);
  if(iodesc == NULL){
    fprintf(stderr,"iodesc handle not found %d %d\n",ioid,__LINE__);
    return PIO_EBADID;
  }
  ios = file->iosystem;
  if(ios->iomaster){
    rlen = iodesc->maxiobuflen;
  }else{
    rlen = iodesc->llen;
  }

  if(iodesc->rearranger > 0){
    if(ios->ioproc && rlen>0){
       MPI_Type_size(iodesc->basetype, &tsize);	
       iobuf = bget(((size_t) tsize)*rlen);
       if(iobuf==NULL){
	 piomemerror(*ios,rlen*((size_t) tsize), __FILE__,__LINE__);
       }
     }
    if(iodesc->rearranger == PIO_REARR_SUBSET){
      // need to prefill fillvalue
      nc_type nctype;
      int errmethod, ierr;
      errmethod = PIOc_Set_File_Error_Handling(ncid, PIO_BCAST_ERROR);
      ierr = PIOc_inq_att(ncid, vid, "_FillValue", &nctype, NULL);
      PIOc_Set_File_Error_Handling(ncid, errmethod);
      if(ierr == PIO_NOERR){
	switch(nctype){
	case NC_INT:
	case NC_FLOAT:
	  fillval = bget(4);
	  PIOc_get_att_int(ncid, vid, "_FillValue", (int *) fillval);
	  for(int i=0;i<arraylen;i++){
	    ((int *) array)[i]= *((int *) fillval);
	  }
	  break;
	case NC_DOUBLE:
	  fillval = bget(8);
	  PIOc_get_att_double(ncid, vid, "_FillValue", (double *) fillval);
	  for(int i=0;i<arraylen;i++){
	    ((double *) array)[i]= *((double *) fillval);
	  }
	  break;
	default:
	  piodie("Unrecognized _Fillvalue type in read_darray",__FILE__,__LINE__);
	}
	brel(fillval);
      }else{  // use  the default netcdf fill value
       vtype = (MPI_Datatype) iodesc->basetype;
       if(vtype == MPI_INTEGER){
	 for(int i=0;i<arraylen;i++){
	   ((int *) array)[i]= PIO_FILL_INT;
	 }
       }else if(vtype == MPI_FLOAT || vtype == MPI_REAL4){
	 for(int i=0;i<arraylen;i++){
	   ((int *) array)[i]= PIO_FILL_FLOAT;
	 }
       }else if(vtype == MPI_DOUBLE || vtype == MPI_REAL8){
	 for(int i=0;i<arraylen;i++){
	   ((int *) array)[i]= PIO_FILL_DOUBLE;
	 }
       }else{
	  piodie("Unrecognized _Fillvalue type in read_darray",__FILE__,__LINE__);
       }
      }

    }

  }else{
    iobuf = array;
  }
  


  switch(file->iotype){
  case PIO_IOTYPE_PNETCDF:
  case PIO_IOTYPE_NETCDF:
  case PIO_IOTYPE_NETCDF4P:
  case PIO_IOTYPE_NETCDF4C:
    ierr = pio_read_darray_nc(file, iodesc, vid, iobuf);
  }
  if(iodesc->rearranger > 0){
    ierr = rearrange_io2comp(*ios, iodesc, iobuf, array);

    if(rlen>0)
      brel(iobuf);
  }

  return ierr;

}

int flush_output_buffer(file_desc_t *file, bool force, PIO_Offset addsize)
{
  var_desc_t *vardesc;
  int ierr=PIO_NOERR;
#ifdef _PNETCDF
//  if(file->nreq==0)
//    return ierr;
  PIO_Offset usage;

  if(file->iosystem->ioproc){

#ifdef TIMING
  GPTLstart("PIO:flush_output_buffer");
#endif

    ierr = ncmpi_inq_buffer_usage(file->fh, &usage);

    if(!force){
      usage += addsize;
      MPI_Allreduce(MPI_IN_PLACE, &usage, 1,  MPI_OFFSET,  MPI_MAX, 
		    file->iosystem->io_comm);
    }
    if(usage > maxusage){
      maxusage = usage;
    }
    if(force || usage>=PIO_BUFFER_SIZE_LIMIT){
      int status[PIO_MAX_VARS];
      int request[PIO_MAX_VARS];
      int i, nreq;
      var_desc_t *vard;
      nreq=0;
      /* BGQ onesided optimization requires that the list of requests in a single call to wait
	 be associated with a contiguous list of variable ids */ 
      for(i=0;i<PIO_MAX_VARS;i++){
	vard = file->varlist+i;
	if(vard->request != NC_REQ_NULL){
	  if(vard->request == PIO_REQ_NULL){
	    request[nreq++] = NC_REQ_NULL;
	  }else{
	    request[nreq++] = vard->request;
	    if(vard->fillrequest != NC_REQ_NULL){
	      request[nreq++] = vard->fillrequest;
	    }
	  }
	  //	  printf("%s %d %d %d %d\n",__FILE__,__LINE__,i,nreq,vard->request);
	  vard->request = NC_REQ_NULL;  //too eager?
	}else if(nreq > 0){
	  /*	  printf("%s %d ",__FILE__,__LINE__);	
	  for(int j=0;j<nreq;j++)
	    printf("%d ",request[j]);
	  printf("\n");
	  */
	  ierr = ncmpi_wait_all(file->fh,nreq, request,status);
	  for(int j=0; j<nreq; j++)
	    request[j]=NC_REQ_NULL;
	  nreq=0;
	
	}
      }
      if(nreq>0){
	ierr = ncmpi_wait_all(file->fh,nreq, request,status);
      }
    }
#ifdef TIMING
  GPTLstop("PIO:flush_output_buffer");
#endif
  }

#endif
  return ierr;
}

void cn_buffer_report(iosystem_desc_t ios, bool collective)
{

  if(CN_bpool != NULL){
    long bget_stats[5];
    long bget_mins[5];
    long bget_maxs[5];

    bstats(bget_stats, bget_stats+1,bget_stats+2,bget_stats+3,bget_stats+4);
    if(collective){
      MPI_Reduce(bget_stats, bget_maxs, 5, MPI_LONG, MPI_MAX, 0, ios.comp_comm);
      MPI_Reduce(bget_stats, bget_mins, 5, MPI_LONG, MPI_MIN, 0, ios.comp_comm);
      if(ios.compmaster){
	printf("PIO: Currently allocated buffer space %ld %ld\n",bget_mins[0],bget_maxs[0]);
	printf("PIO: Currently available buffer space %ld %ld\n",bget_mins[1],bget_maxs[1]);
	printf("PIO: Current largest free block %ld %ld\n",bget_mins[2],bget_maxs[2]);
	printf("PIO: Number of successful bget calls %ld %ld\n",bget_mins[3],bget_maxs[3]);
	printf("PIO: Number of successful brel calls  %ld %ld\n",bget_mins[4],bget_maxs[4]);
	//	print_trace(stdout);
      }
    }else{
      printf("%d: PIO: Currently allocated buffer space %ld \n",ios.union_rank,bget_stats[0]) ;
      printf("%d: PIO: Currently available buffer space %ld \n",ios.union_rank,bget_stats[1]);
      printf("%d: PIO: Current largest free block %ld \n",ios.union_rank,bget_stats[2]);
      printf("%d: PIO: Number of successful bget calls %ld \n",ios.union_rank,bget_stats[3]);
      printf("%d: PIO: Number of successful brel calls  %ld \n",ios.union_rank,bget_stats[4]);
    }
  }
}

void free_cn_buffer_pool(iosystem_desc_t ios)
{
#ifndef PIO_USE_MALLOC
  if(CN_bpool != NULL){
    cn_buffer_report(ios, true);
    bpoolrelease(CN_bpool);
    //    free(CN_bpool);
    CN_bpool=NULL;
  }
#endif
}

void flush_buffer(int ncid, wmulti_buffer *wmb)
{
  if(wmb->validvars>0){
      PIOc_write_darray_multi(ncid, wmb->vid,  wmb->ioid, wmb->validvars, wmb->arraylen, wmb->data, wmb->frame, wmb->fillvalue);
      wmb->validvars=0;
      brel(wmb->vid);
      wmb->vid=NULL;
      brel(wmb->data);
      wmb->data=NULL;
      if(wmb->fillvalue != NULL)
	brel(wmb->fillvalue);
      if(wmb->frame != NULL)
	brel(wmb->frame);
      wmb->fillvalue=NULL;
      wmb->frame=NULL;
    }
}    
  
void compute_maxaggregate_bytes(const iosystem_desc_t ios, io_desc_t *iodesc)
{
  int maxbytesoniotask=INT_MAX;
  int maxbytesoncomputetask=INT_MAX;
  int maxbytes;
  
  // printf("%s %d %d %d\n",__FILE__,__LINE__,iodesc->maxiobuflen, iodesc->ndof);

  if(ios.ioproc && iodesc->maxiobuflen>0){
     maxbytesoniotask = PIO_BUFFER_SIZE_LIMIT/ iodesc->maxiobuflen;
  }
  if(ios.comp_rank>=0 && iodesc->ndof>0){
    maxbytesoncomputetask = PIO_CNBUFFER_LIMIT/iodesc->ndof;
  }
  maxbytes = min(maxbytesoniotask,maxbytesoncomputetask);

  //  printf("%s %d %d %d\n",__FILE__,__LINE__,maxbytesoniotask, maxbytesoncomputetask);

  MPI_Allreduce(MPI_IN_PLACE, &maxbytes, 1, MPI_INT, MPI_MIN, ios.union_comm);
  iodesc->maxbytes=maxbytes;
  //  printf("%s %d %d %d\n",__FILE__,__LINE__,iodesc->maxbytes,iodesc->maxiobuflen);
    
}
