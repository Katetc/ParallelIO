////
/// @file pio_rearrange.c
/// @author Jim Edwards
/// @date 2014
/// @brief Code to map IO to model decomposition
///
/// 
/// 
/// 
/// @see  http://code.google.com/p/parallelio/
////
#include <pio.h>
#include <pio_internal.h>
#include <limits.h>
/** internal variable used for debugging */
int tmpioproc=-1;  


/** @internal
 ** Convert an index into a list of dimensions. E.g., for index 4 into a
 ** array defined as a[3][2], will return 1 1.
 ** @endinternal
*/
void idx_to_dim_list(const int ndims, const int gdims[], const PIO_Offset idx,
                     PIO_Offset dim_list[])
{
  int i, curr_idx, next_idx;
  curr_idx = idx;
  // Easiest to start from the right and move left.
  for (i = ndims-1; i >= 0; --i) {
    // This way of doing div/mod is slightly faster than using "/" and "%".
    next_idx = curr_idx / gdims[i];
    dim_list[i] = curr_idx - (next_idx*gdims[i]);
    curr_idx = next_idx;
  }
}
/**
 ** @internal
 ** Expand a region along dimension dim, by incrementing count[i] as much as
 ** possible, consistent with the map.
 **
 ** Once max_size is reached, the map is exhausted, or the next entries fail
 ** to match, expand_region updates the count and calls itself with the next
 ** outermost dimension, until the region has been expanded as much as
 ** possible along all dimensions.
 ** @endinternal
 */
void expand_region(const int dim, const int gdims[], const int maplen,
                   const PIO_Offset map[], const int region_size,
                   const int region_stride, const int max_size[],
                   PIO_Offset count[])
{
  int i, j, test_idx, expansion_done;
  // Precondition: maplen >= region_size (thus loop runs at least once).

  // Flag used to signal that we can no longer expand the region along
  // dimension dim.
  expansion_done = 0;

  // Expand no greater than max_size along this dimension.
  for (i = 1; i <= max_size[dim]; ++i) {
    // Count so far is at least i.
    count[dim] = i;

    // Now see if we can expand to i+1 by checking that the next
    // region_size elements are ahead by exactly region_stride.
    // Assuming monotonicity in the map, we could skip this for the
    // innermost dimension, but it's necessary past that because the
    // region does not necessarily comprise contiguous values.
    for (j = 0; j < region_size; ++j) {
      test_idx = j + i*region_size;
      // If we have exhausted the map, or the map no longer matches,
      // we are done, break out of both loops.
      if (test_idx >= maplen || map[test_idx] != map[j] + i*region_stride) {
        expansion_done = 1;
        break;
      }
    }
    if (expansion_done) break;
  }

  // Move on to next outermost dimension if there are more left, else return.
  if (dim > 0) {
    expand_region(dim-1, gdims, maplen, map, region_size*count[dim],
                  region_stride*gdims[dim], max_size, count);
  }
}
/**
 ** @internal
 ** Set start and count so that they describe the first region in map.
 ** @endinternal
 */
PIO_Offset find_region(const int ndims, const int gdims[],
                       const int maplen, const PIO_Offset map[],
                       PIO_Offset start[], PIO_Offset count[])
{
  int dim;
  int max_size[ndims];
  PIO_Offset regionlen=1;
  // Preconditions (which might be useful to check/assert):
  //   ndims is > 0
  //   maplen is > 0
  //   all elements of map are inside the bounds specified by gdims
  // The map array is 1 based, but calculations are 0 based
  idx_to_dim_list(ndims, gdims, map[0]-1, start);

  for (dim = 0; dim < ndims; ++dim) {
    // Can't expand beyond the array edge.
    max_size[dim] = gdims[dim] - start[dim];
  }

  // For each dimension, figure out how far we can expand in that dimension
  // while staying contiguous in the input array.
  //
  // Start with the innermost dimension (ndims-1), and it will recurse
  // through to the outermost dimensions.
  expand_region(ndims-1, gdims, maplen, map, 1, 1, max_size, count);

  for(dim=0;dim<ndims;dim++)
    regionlen*=count[dim];
  return(regionlen);

}

/**
 ** @internal
** Convert a global coordinate value into a local array index
** @endinternal
*/
PIO_Offset coord_to_lindex(const int ndims, const PIO_Offset lcoord[], const PIO_Offset count[])
{
  PIO_Offset lindex=0;
  PIO_Offset stride=1;

  for(int i=ndims-1; i>=0; i--){
    lindex += lcoord[i]*stride;
    stride = stride*count[i];
  }
  return lindex;

}

/**
 ** @internal
** Compute the max io buffersize needed for a given variable
** @endinternal
*/
void compute_maxIObuffersize(MPI_Comm io_comm, io_desc_t *iodesc)
{
  PIO_Offset iosize, totiosize;
  int i;
  io_region *region;

  //  compute the max io buffer size, for conveneance it is the combined size of all regions
  totiosize=0;
  region = iodesc->firstregion;
  while(region != NULL){
    if(region->count[0]>0){
      iosize=1;
      for(i=0;i<iodesc->ndims;i++)
	iosize*=region->count[i];
      totiosize+=iosize;
    }
    region = region->next;
  }
  // Share the max io buffer size with all io tasks
#ifndef _MPISERIAL
  CheckMPIReturn(MPI_Allreduce(&totiosize, &(iodesc->maxiobuflen), 1, MPI_OFFSET, MPI_MAX, io_comm),__FILE__,__LINE__);
#endif
  iodesc->maxiobuflen = totiosize;
  
}
/**
 ** @internal
** Create the derived MPI datatypes used for comp2io and io2comp transfers
** @endinternal
*/
int create_mpi_datatypes(const MPI_Datatype basetype,const int msgcnt,const PIO_Offset dlen, const PIO_Offset mindex[],const int mcount[],
			 int *mfrom, MPI_Datatype mtype[])
{
  PIO_Offset bsizeT[msgcnt];
  int pos;
  int ii;
  PIO_Offset i8blocksize;
  int blocksize;
  PIO_Offset *lindex;

  pioassert(dlen>=0,"dlen < 0",__FILE__,__LINE__);

#ifdef _MPISERIAL
  mtype[0] = basetype * blocksize;
#else
  if(mindex != NULL){
    lindex = (PIO_Offset *) malloc(dlen * sizeof(PIO_Offset));
    memcpy(lindex, mindex, (size_t) (dlen*sizeof(PIO_Offset)));
  }else{
    lindex = NULL;
  }
  bsizeT[0]=0;
  mtype[0] = MPI_DATATYPE_NULL;
  pos = 0;
  ii = 0;
  if(msgcnt>0){
    if(mfrom == NULL){
      for(int i=0;i<msgcnt;i++){
	if(mcount[i]>0){
	  bsizeT[ii] = GCDblocksize(mcount[i], lindex+pos);
	  ii++;
	  pos+=mcount[i];

	}
      }
      blocksize = (int) lgcd_array(ii ,bsizeT);
    }else{
      blocksize=1;
    }
    
    pos = 0;
    for(int i=0;i< msgcnt; i++){
      if(mcount[i]>0){
	int len = mcount[i]/blocksize;
	int displace[len];
	if(blocksize==1){
	  if(mfrom == NULL){
	    for(int j=0;j<len;j++)
	      displace[j] = (int) (lindex[pos+j]);
	  }else{
	    int k=0;
	    for(int j=0;j<dlen;j++)
	      if(mfrom[j]==i)
		displace[k++] = (int) (lindex[j]);
	  }
	    
	}else{
	  for(int j=0;j<mcount[i];j++)
	    (lindex+pos)[j]++;
	  for(int j=0;j<len;j++){
	    displace[j]= ((lindex+pos)[j*blocksize]-1);
	  }
	}

	CheckMPIReturn(MPI_Type_create_indexed_block(len, blocksize, displace, basetype, mtype+i),__FILE__,__LINE__);
	CheckMPIReturn(MPI_Type_commit(mtype+i), __FILE__,__LINE__);
	pos+=mcount[i];

      }
    }

  }
  if(lindex != NULL){  
    free(lindex);
  }
#endif
  return PIO_NOERR;

}

/**
 ** @internal
** Create the derived MPI datatypes used for comp2io and io2comp transfers
** @endinternal
*/
int define_iodesc_datatypes(const iosystem_desc_t ios, io_desc_t *iodesc,const int indim)
{
  int i;
  MPI_Datatype derivedtype;

  if(indim > 1){
    CheckMPIReturn(MPI_Type_contiguous(indim, iodesc->basetype, &derivedtype),__FILE__,__LINE__);
    CheckMPIReturn(MPI_Type_commit(&derivedtype), __FILE__,__LINE__);
  }else{
    derivedtype = iodesc->basetype;
  }


  if(ios.ioproc){
    if(iodesc->rtype==NULL){
      int ntypes = iodesc->nrecvs;
      iodesc->rtype = (MPI_Datatype *) malloc(ntypes * sizeof(MPI_Datatype));
      for(i=0; i<ntypes; i++){
        iodesc->rtype[i] = MPI_DATATYPE_NULL;
      }
      iodesc->num_rtypes = ntypes;

      if(iodesc->rearranger==PIO_REARR_SUBSET){
	create_mpi_datatypes(derivedtype, iodesc->nrecvs, iodesc->llen, iodesc->rindex, iodesc->rcount, iodesc->rfrom, iodesc->rtype);
      }else{
	create_mpi_datatypes(derivedtype, iodesc->nrecvs, iodesc->llen, iodesc->rindex, iodesc->rcount, NULL, iodesc->rtype);
      }
#ifndef _MPISERIAL
      /*      if(tmpioproc==95)     {
	MPI_Aint lb;
	MPI_Aint extent;
	for(i=0;i<ntypes;i++){
	  MPI_Type_get_extent(iodesc->rtype[i], &lb, &extent);
	  printf("%s %d %d %d %d \n",__FILE__,__LINE__,i,lb,extent);
	  
	}
	}
      */
#endif
    }
  }


  if(iodesc->stype==NULL){
    int ntypes;
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      ntypes = 1;
    else
      ntypes = ios.num_iotasks;


    //  printf("COMP: %d\n",ntypes);


    iodesc->stype = (MPI_Datatype *) malloc(ntypes * sizeof(MPI_Datatype));
    for(i=0; i<ntypes; i++){
      iodesc->stype[i] = MPI_DATATYPE_NULL;
    }
    iodesc->num_stypes = ntypes;

    create_mpi_datatypes(derivedtype, ntypes, iodesc->ndof, iodesc->sindex, iodesc->scount, NULL, iodesc->stype);
#ifndef _MPISERIAL
    /*    if(tmpioproc==95)   {
      MPI_Aint lb;
      MPI_Aint extent;
      for(i=0;i<ntypes;i++){
	MPI_Type_get_extent(iodesc->stype[i], &lb, &extent);
	printf("%s %d %d %d %d \n",__FILE__,__LINE__,i,lb,extent);
      }
      }
    */
#endif
  }
  if(indim > 1){
    CheckMPIReturn(MPI_Type_free(&derivedtype), __FILE__,__LINE__);
  }


  return PIO_NOERR;

}


/**
 ** @internal
**  Completes the mapping for the box rearranger
** @endinternal
*/

int compute_counts(const iosystem_desc_t ios, io_desc_t *iodesc, const int maplen, 
		   const int dest_ioproc[], const PIO_Offset dest_ioindex[], MPI_Comm mycomm)
{

  int i;
  int iorank;

  int rank;
  int ntasks;

  MPI_Comm_rank(mycomm, &rank);
  MPI_Comm_size(mycomm, &ntasks);


  MPI_Datatype sr_types[ntasks];
  int send_counts[ntasks];
  int send_displs[ntasks];
  int recv_counts[ntasks];
  int recv_displs[ntasks];
  int *recv_buf=NULL;
  int nrecvs;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int ierr;
  int io_comprank;
  int ioindex;
  int tsize;
  int numiotasks;
  PIO_Offset s2rindex[iodesc->ndof];


  
  if(iodesc->rearranger==PIO_REARR_BOX)
    numiotasks = ios.num_iotasks;
  else
    numiotasks=1;

  iodesc->scount = (int *) calloc(numiotasks,sizeof(int));

  // iodesc->scount is the amount of data sent to each task from the current task
  for(i=0;i<maplen; i++){
    if(dest_ioindex[i] >= 0){
      (iodesc->scount[dest_ioproc[i]])++;
    }
  }

  //  for(i=0;i<ios.num_iotasks;i++)
  //   printf("iodesc->scount = %d\n",iodesc->scount[i]);

  for(i=0;i<ntasks;i++){
    send_counts[i] = 0;
    send_displs[i] = 0;
    recv_counts[i] = 0;
    recv_displs[i] = 0;
    sr_types[i] = MPI_INT;
  }
  for(i=0;i<numiotasks;i++){
    int io_comprank;
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      io_comprank=0;
    else
      io_comprank = ios.ioranks[i];
    send_counts[io_comprank] = 1;
    send_displs[io_comprank] = i*sizeof(int);
  }

  if(ios.ioproc){
    recv_buf = (int *) malloc(ntasks * sizeof(int));
    for(i=0;i<ntasks;i++){
      recv_buf[i] = 0;
      recv_counts[i] = 1;
      recv_displs[i] = i*sizeof(int);
    }
  }
  //  for(i=0;i<numiotasks;i++)
  //  printf("%s %d %d\n",__FILE__,__LINE__,iodesc->scount[i]);

  // Share the iodesc->scount from each compute task to all io tasks
  ierr = pio_swapm( iodesc->scount, send_counts, send_displs, sr_types, 
                    recv_buf,  recv_counts, recv_displs, sr_types,
		    mycomm, false, false, maxreq);

  nrecvs = 0;
  if(ios.ioproc){
    //       printf("recv_buf = ");
    for(i=0;i<ntasks; i++){
      //     printf(" %d ",recv_buf[i]);
      if(recv_buf[i] != 0)
	nrecvs++;
    }
    // printf("\n");

    iodesc->rcount = (int *) calloc(max(1,nrecvs),sizeof(int));
    iodesc->rfrom = (int *) calloc(max(1,nrecvs),sizeof(int));
    

    nrecvs = 0;
    for(i=0;i<ntasks; i++){
      if(recv_buf[i] != 0){
	iodesc->rcount[nrecvs] = recv_buf[i];
	iodesc->rfrom[nrecvs] = i;
	nrecvs++;
      }

    }
    free(recv_buf);
  }

  iodesc->nrecvs = nrecvs;
  if(iodesc->sindex == NULL)
    iodesc->sindex = (PIO_Offset *) calloc(iodesc->ndof,sizeof(PIO_Offset));


  int tempcount[numiotasks];
  int spos[numiotasks];

  spos[0]=0;
  tempcount[0]=0;
  for(i=1;i<numiotasks;i++){
    spos[i] = spos[i-1] + iodesc->scount[i-1];
    tempcount[i]=0;
  }

  for(i=0;i<maplen;i++){
    iorank =dest_ioproc[i]; 
    ioindex = dest_ioindex[i];
    if(iorank > -1){
      // this should be moved to create_box
      if(iodesc->rearranger==PIO_REARR_BOX)
	iodesc->sindex[spos[iorank]+tempcount[iorank]] = i;

      s2rindex[spos[iorank]+tempcount[iorank]] = ioindex;
      (tempcount[iorank])++;
    }
  }
    //    printf("%s %d %d %d %d %d\n",__FILE__,__LINE__,iodesc->llen,iodesc->ndof, maplen,spos[0]+tempcount[0]);

  for(i=0;i<ntasks;i++){
    send_counts[i] = 0;
    send_displs[i]  = 0;
    recv_counts[i] = 0;
    recv_displs[i]   =0;
  }
#ifndef _MPISERIAL
  MPI_Type_size(MPI_OFFSET, &tsize);
#else
  tsize = sizeof(long long);
#endif
  for(i=0; i<ntasks; i++){
    sr_types[i] = MPI_OFFSET;
  }

  for(i=0;i<numiotasks;i++){
    if(iodesc->rearranger==PIO_REARR_BOX){
      io_comprank = ios.ioranks[i];
    }else{
      io_comprank=0;
    }
    send_counts[io_comprank] = iodesc->scount[i];
    if(send_counts[io_comprank]>0)
      send_displs[io_comprank]  = spos[i]*tsize ;
  }

  if(ios.ioproc){
    for(i=0;i<nrecvs;i++)
      recv_counts[iodesc->rfrom[i]] = iodesc->rcount[i];
    recv_displs[0] = 0;
    for(i=1;i<nrecvs;i++)
      recv_displs[iodesc->rfrom[i]] = recv_displs[iodesc->rfrom[i-1]]+iodesc->rcount[i-1]*tsize;
    if(iodesc->llen>0)
      iodesc->rindex = (PIO_Offset *) calloc(iodesc->llen,sizeof(PIO_Offset));
  }
  //   printf("%d rbuf_size %d\n",ios.comp_rank,rbuf_size);


  // s2rindex is the list of indeces on each compute task
  /*        
  printf("%d s2rindex: ", ios.comp_rank);
  for(i=0;i<iodesc->ndof;i++)
    printf("%ld ",s2rindex[i]);
  printf("\n");
  */
  //  printf("%s %d %ld\n",__FILE__,__LINE__,iodesc->llen);
  //  printf("%s %d %d %d %d %d %d %d\n",__FILE__,__LINE__,send_counts[0],recv_counts[0],send_displs[0],recv_displs[0],sr_types[0],iodesc->llen);
  ierr = pio_swapm( s2rindex, send_counts, send_displs, sr_types, 
		    iodesc->rindex, recv_counts, recv_displs, sr_types,
  		    mycomm, false, false, 0);
  // printf("%s %d\n",__FILE__,__LINE__);

  //  rindex is an array of the indices of the data to be sent from
  //  this io task to each compute task. 
  /* 
  if(ios.ioproc){
    printf("%d rindex: ",ios.io_rank);
    for(int j=0;j<iodesc->llen;j++)
      printf(" %ld ",iodesc->rindex[j]);
    printf("\n");
  }
  */
  return ierr;

}

/** 
 ** @internal
 ** Moves data from compute tasks to IO tasks.
 ** @endinternal
 **
 */


int rearrange_comp2io(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int indim)
{

  bool handshake=false;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int ntasks;
  int niotasks;
  int *scount = iodesc->scount;

  int i, tsize;
  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;
  MPI_Comm mycomm;
  
  if(iodesc->rearranger == PIO_REARR_BOX){
    mycomm = ios.union_comm;
    niotasks = ios.num_iotasks;
    if(niotasks == ios.num_comptasks)
      maxreq  = 0;
  }else{
    mycomm = iodesc->subset_comm;
    niotasks = 1;
  }  
  MPI_Comm_size(mycomm, &ntasks);

#ifdef _MPISERIAL
  /* in mpiserial iodesc->basetype is the byte length of the basic type */
  for(i=0;i<iodesc->llen; i++){
    memcpy((char *) rbuf+ iodesc->rindex[i],(char *) sbuf + iodesc->sindex[i], (size_t) (iodesc->basetype * indim));
  }
#else
  define_iodesc_datatypes(ios, iodesc, indim);
  
  sendcounts = (int *) malloc(ntasks*sizeof(int));
  recvcounts = (int *) malloc(ntasks*sizeof(int));
  sdispls = (int *) malloc(ntasks*sizeof(int));
  rdispls = (int *) malloc(ntasks*sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));

  for(i=0;i<ntasks;i++){
    sendcounts[i] = 0;
    recvcounts[i] = 0; 
    sdispls[i] = 0; 
    rdispls[i] = 0;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
    sendtypes[ i ] =  MPI_DATATYPE_NULL;
  }


  if(ios.ioproc && iodesc->nrecvs>0){
    //    printf("%d: rindex[%d] %d\n",ios.comp_rank,0,iodesc->rindex[0]);
    for( i=0;i<iodesc->nrecvs;i++){
      if(iodesc->rtype[i] != MPI_DATATYPE_NULL){
	if(iodesc->rearranger==PIO_REARR_SUBSET){
	  recvcounts[ i ] = 1;
	  recvtypes[ i ] = iodesc->rtype[i];
	}else{
	  recvcounts[ iodesc->rfrom[0] ] = 1;
	  recvtypes[ iodesc->rfrom[0] ] = iodesc->rtype[0];
	  rdispls[ iodesc->rfrom[0] ] = 0;
	  //    printf("%d: rindex[%d] %d\n",ios.comp_rank,0,iodesc->rindex[0]);
	  for( i=1;i<iodesc->nrecvs;i++){
	    recvcounts[ iodesc->rfrom[i] ] = 1;
	    recvtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];
	    
	  }
	}
      }
      //   printf("%d: rindex[%d] %d\n",ios.comp_rank,i,iodesc->rindex[i]);
      
    }
  }

  for( i=0;i<niotasks; i++){
    int io_comprank = ios.ioranks[i];
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      io_comprank=0;
    //    printf("scount[%d]=%d\n",i,scount[i]);
    if(scount[i] > 0) {
      sendcounts[io_comprank]=1;
      sendtypes[io_comprank]=iodesc->stype[i];
    }else{
      sendcounts[io_comprank]=0;
    }
  }      

  // Data in sbuf on the compute nodes is sent to rbuf on the ionodes
  pio_swapm( sbuf,  sendcounts, sdispls, sendtypes,
	     rbuf, recvcounts, rdispls, recvtypes, 
	     mycomm, handshake, isend, maxreq);
  

  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);
#endif
  return PIO_NOERR;
}
/** 
 ** @internal
 ** Moves data from compute tasks to IO tasks.
 ** @endinternal
 **
 */

int rearrange_io2comp(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int indim)
{
  

  bool handshake=false;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  //  int maxreq = -1;
  MPI_Comm mycomm;

  int ntasks ;
  int niotasks;
  int *scount = iodesc->scount;

  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;

  int i, tsize;
  if(iodesc->rearranger==PIO_REARR_BOX){
    mycomm = ios.union_comm;
    niotasks = ios.num_iotasks;
  }else{
    mycomm = iodesc->subset_comm;
    niotasks=1;
  }
  MPI_Comm_size(mycomm, &ntasks);

#ifdef _MPISERIAL
  for(i=0;i<iodesc->llen;i++){
    memcpy((char *) rbuf+ iodesc->sindex[i],(char *)sbuf + iodesc->rindex[i], (size_t) (iodesc->basetype * indim));
  }
#else  
  define_iodesc_datatypes(ios, iodesc, indim);

  sendcounts = (int *) calloc(ntasks,sizeof(int));
  recvcounts = (int *) calloc(ntasks,sizeof(int));
  sdispls = (int *) calloc(ntasks,sizeof(int));
  rdispls = (int *) calloc(ntasks,sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));


  for( i=0;i< ntasks;i++){
    sendtypes[ i ] = MPI_DATATYPE_NULL;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
  }
  if(ios.ioproc){
    for( i=0;i< iodesc->nrecvs;i++){
      if(iodesc->rtype[i] != MPI_DATATYPE_NULL){
	if(iodesc->rearranger==PIO_REARR_SUBSET){
	  sendcounts[ i ] = 1;
	  sendtypes[ i ] = iodesc->rtype[i];
	}else{
	  sendcounts[ iodesc->rfrom[i] ] = 1;
	  sendtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];
	}
      }
    }
  }

  for( i=0;i<niotasks; i++){
    int io_comprank = ios.ioranks[i];
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      io_comprank=0;
    if(scount[i] > 0 && iodesc->stype[i] != MPI_DATATYPE_NULL) {
      recvcounts[io_comprank]=1;
      recvtypes[io_comprank]=iodesc->stype[i];
    }
  } 
  
  //
  // Data in sbuf on the ionodes is sent to rbuf on the compute nodes
  //
  //  printf("%s %d \n",__FILE__,__LINE__);

  pio_swapm( sbuf,  sendcounts, sdispls, sendtypes,
	     rbuf, recvcounts, rdispls, recvtypes, 
	     mycomm, handshake,isend, maxreq);

  //  printf("%s %d \n",__FILE__,__LINE__);

  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);
#endif
 
  return PIO_NOERR;

}
/** 
 ** @internal
 ** The box rearranger computes a mapping between IO tasks and compute tasks such that the data
 ** on io tasks can be written with a single call to the underlying netcdf library.   This 
 ** may involve an all to all rearrangement in the mapping, but should minimize data movement in 
 ** lower level libraries
 ** @endinternal
 **
 */

int box_rearrange_create(const iosystem_desc_t ios,const int maplen, const PIO_Offset compmap[], const int gsize[],
			 const int ndims, io_desc_t *iodesc)
{
  int ierr=PIO_NOERR;
  int nprocs = ios.num_comptasks;
  int nioprocs = ios.num_iotasks;
  PIO_Offset gstride[ndims];
  PIO_Offset iomap;
  PIO_Offset start[ndims], count[ndims];
  int  tsize, i, j, k, llen;
  MPI_Datatype dtype;
  int dest_ioproc[maplen];
  PIO_Offset dest_ioindex[maplen];
  int sndlths[nprocs]; 
  int sdispls[nprocs];
  int recvlths[nprocs];
  int rdispls[nprocs];
  MPI_Datatype dtypes[nprocs];
  PIO_Offset iomaplen[nioprocs];

  iodesc->rearranger = PIO_REARR_BOX;

  iodesc->ndof = maplen;
  gstride[ndims-1]=1;
  for(int i=ndims-2;i>=0; i--)
    gstride[i]=gstride[i+1]*gsize[i+1];

#ifndef _MPISERIAL
  MPI_Type_size(MPI_OFFSET, &tsize);
#endif

  for(i=0; i< maplen; i++){
    dest_ioproc[i] = -1;
    dest_ioindex[i] = -1;
  }
  for(i=0;i<nprocs;i++){
    sndlths[i] = 0;
    sdispls[i] = 0;
    recvlths[i] = 0;
    rdispls[i] = 0;
    dtypes[i] = MPI_OFFSET;
  }
  iodesc->llen=0;
  if(ios.ioproc){
    for( i=0;i<nprocs;i++){
      sndlths[ i ] = 1;
    }
    iodesc->llen=1;
    for(i=0;i<ndims;i++)
      iodesc->llen *= iodesc->firstregion->count[i];
  }

  for( i=0;i<nioprocs; i++){
    int io_comprank = ios.ioranks[i];
    recvlths[ io_comprank ] = 1;
    rdispls[ io_comprank ] = i*tsize;
  }      

  //  The length of each iomap
  //  iomaplen = calloc(nioprocs, sizeof(PIO_Offset));
  pio_swapm(&(iodesc->llen), sndlths, sdispls, dtypes,
	    iomaplen, recvlths, rdispls, dtypes, 	
	    ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);
  /*
  printf("%s %d %d\n",__FILE__,__LINE__,nioprocs);
  for(i=0; i<nioprocs; i++){
    printf("%ld ",iomaplen[i]);
  }
  printf("\n");
  */
  for(i=0; i<nioprocs; i++){

    if(iomaplen[i]>0){
      int io_comprank = ios.ioranks[i];
      for( j=0; j<nprocs ; j++){
	sndlths[ j ] = 0;
	sdispls[ j ] = 0;
	rdispls[ j ] = 0;
	recvlths[ j ] = 0;
	if(ios.union_rank == io_comprank)
	  sndlths[ j ] = ndims;
      }
      recvlths[ io_comprank ] = ndims;
      
      // The count from iotask i is sent to all compute tasks
      
      pio_swapm(iodesc->firstregion->count,  sndlths, sdispls, dtypes,
		count, recvlths, rdispls, dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);
      
      // The start from iotask i is sent to all compute tasks
      pio_swapm(iodesc->firstregion->start,  sndlths, sdispls, dtypes,
		start, recvlths, rdispls, dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);

      for(k=0;k<maplen;k++){
	PIO_Offset gcoord[ndims], lcoord[ndims];
	bool found=true;
	// The compmap array is 1 based but calculations are 0 based
	idx_to_dim_list(ndims, gsize, compmap[k]-1, gcoord);


	for(j=0;j<ndims;j++){
	  if(gcoord[j] >= start[j] && gcoord[j] < start[j]+count[j]){
	    lcoord[j] = gcoord[j] - start[j];
	  }else{
	    found = false;
	    break;
	  }
	}
	if(found){
	  dest_ioindex[k] = coord_to_lindex(ndims, lcoord, count);
	  dest_ioproc[k] = i;
	}
      }
    }
  }

  for(k=0; k<maplen; k++){
    //       printf("%s %d %d %d\n",__FILE__,__LINE__,dest_ioproc[k],dest_ioindex[k]);

    if(dest_ioproc[k] < 0 && compmap[k]>0){
      fprintf(stderr,"No destination found for compmap[%d] = %ld\n",k,compmap[k]);
      piodie(" ",__FILE__,__LINE__);
    }
  }

  compute_counts(ios, iodesc, maplen, dest_ioproc, dest_ioindex, ios.union_comm);

  if(ios.ioproc){
    compute_maxIObuffersize(ios.io_comm, iodesc);
  }
  return PIO_NOERR;
}
/** 
 ** @internal
 ** compare offsets is used by the sort in the subset rearrange
 ** @endinternal
 */
int compare_offsets(const void *a,const void *b) 
{
  mapsort *x = (mapsort *) a;
  mapsort *y = (mapsort *) b;
  return (int) (x->iomap - y->iomap);
}    

/**
 ** @internal
 ** Each region is a block of output which can be represented in a single call to the underlying 
 ** netcdf library.  This can be as small as a single data point, but we hope we've aggrigated better than that. 
 ** @endinternal
 */
void get_start_and_count_regions(const MPI_Comm io_comm, io_desc_t *iodesc, const int gdims[],const PIO_Offset map[])
{
  int i;
  int nmaplen;
  int regionlen;
  io_region *region, *prevregion;
  int ndims=iodesc->ndims;

  nmaplen = 0;
  region = iodesc->firstregion;
  while(map[nmaplen++]<=0);
  nmaplen--;
  region->loffset=nmaplen;

  iodesc->maxregions = 1;
  prevregion=NULL;

  while(nmaplen < iodesc->llen){
    // Here we find the largest region from the current offset into the iomap
    // regionlen is the size of that region and we step to that point in the map array
    // until we reach the end 
    for(i=0;i<ndims;i++){
      region->count[i]=1;
    }

    regionlen = find_region(ndims, gdims, iodesc->llen-nmaplen, 
				  map+nmaplen, region->start, region->count);

    pioassert(region->start[0]>=0,"failed to find region",__FILE__,__LINE__);
    

    
    nmaplen = nmaplen+regionlen;
    if(region->next==NULL && nmaplen<iodesc->llen){
      region->next = alloc_region(iodesc->ndims);
      // The offset into the local array buffer is the sum of the sizes of all of the previous regions (loffset) 
      region=region->next;
      region->loffset = nmaplen;
      // The calls to the io library are collective and so we must have the same number of regions on each
      // io task iodesc->maxregions will be the total number of regions on this task 
      iodesc->maxregions++;
    }
  }

  // pad maxregions on all tasks to the maximum and use this to assure that collective io calls are made.
#ifndef _MPISERIAL  
  MPI_Allreduce(MPI_IN_PLACE,&(iodesc->maxregions), 1, MPI_INTEGER, MPI_MAX, io_comm);
#endif
}

/** 
 ** @internal
 ** The subset rearranger needs a mapping from compute tasks to IO task, the only requirement is 
 ** that each compute task map to one and only one IO task.   This mapping groups by mpi task id
 ** others are possible and may be better for certain decompositions
 ** @endinternal
 **
 */
void default_subset_partition(const iosystem_desc_t ios, io_desc_t *iodesc)
{
  int taskratio = ios.num_comptasks/ios.num_iotasks;
  int color;
  int key;

  /* Create a new comm for each subset group with the io task in rank 0 and
     only 1 io task per group */

  if(ios.ioproc){
    key = 0;
    color= ios.io_rank;
  }else{
    key=max(1,ios.comp_rank%taskratio+1);
    color = min(ios.num_iotasks-1,ios.comp_rank/taskratio);
  }

  MPI_Comm_split(ios.comp_comm, color, key, &(iodesc->subset_comm));

}

/** 
 ** @internal
 ** The subset rearranger computes a mapping between IO tasks and compute tasks such that each compute
 ** task communicates with one and only one IO task.  
 ** @endinternal
 **
 */

int subset_rearrange_create(const iosystem_desc_t ios,const int maplen, PIO_Offset compmap[], 
			    const int gsize[], const int ndims, io_desc_t *iodesc)
{

  int taskratio;

  int i, j, jlast;
  bool hs=false;
  bool isend=false;
  PIO_Offset *iomap=NULL;
  int ierr = PIO_NOERR;
  mapsort *map=NULL;
  PIO_Offset totalgridsize;
  PIO_Offset *srcindex=NULL;


  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int rank, ntasks, rcnt;
  size_t pio_offset_size=sizeof(PIO_Offset);


  tmpioproc = ios.io_rank;


  /* subset partitions each have exactly 1 io task which is task 0 of that subset_comm */ 
  /* TODO: introduce a mechanism for users to define partitions */
  default_subset_partition(ios, iodesc);
  iodesc->rearranger = PIO_REARR_SUBSET;

  MPI_Comm_rank(iodesc->subset_comm, &rank);
  MPI_Comm_size(iodesc->subset_comm, &ntasks);

  if(ios.ioproc){
    pioassert(rank==0,"Bad io rank in subset create",__FILE__,__LINE__);
  }else{
    pioassert(rank>0 && rank<ntasks,"Bad comp rank in subset create",__FILE__,__LINE__);
  }
  totalgridsize=1;
  for(i=0;i<ndims;i++)
    totalgridsize*=gsize[i];

  rcnt = 0;
  iodesc->ndof = maplen;
  if(ios.ioproc){
    iodesc->rcount = (int *) malloc(ntasks *sizeof(int));
    rcnt = 1;
  } 
  iodesc->scount = (int *) calloc(1,sizeof(int));

  for(i=0;i<maplen;i++){
    pioassert(compmap[i]>=0 && compmap[i]<=totalgridsize, "Compmap value out of bounds",__FILE__,__LINE__);
    if(compmap[i]>0){
      (iodesc->scount[0])++;
    }
  }
  if(iodesc->scount[0]>0)
    iodesc->sindex = (PIO_Offset *) calloc(iodesc->scount[0],pio_offset_size); 

  j=0;
  for(i=0;i<maplen;i++){
    if(compmap[i]>0){
      iodesc->sindex[j++]=i;

    }
  }

  // Pass the reduced maplen (without holes) from each compute task to its associated IO task
  //  printf("%s %d %ld\n",__FILE__,__LINE__,iodesc->scount);
  
  pio_fc_gather( (void *) iodesc->scount, 1, MPI_INT,
		 (void *) iodesc->rcount, rcnt, MPI_INT, 
		 0, iodesc->subset_comm, maxreq);

  iodesc->llen = 0;

  int rdispls[ntasks];
  int recvlths[ntasks];

  if(ios.ioproc){
    for(i=0;i<ntasks;i++){
      iodesc->llen+=iodesc->rcount[i];
      rdispls[i]=0;
      recvlths[i]= iodesc->rcount[ i ];
      if(i>0)
	rdispls[i] = rdispls[i-1]+ iodesc->rcount[ i-1 ];
    }
    //    printf("%s %d %ld %d %d\n",__FILE__,__LINE__,iodesc,iodesc->llen,maplen);

      
    if(iodesc->llen>0){
      srcindex = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
    }
  }else{
    for(i=0;i<ntasks;i++){
      recvlths[i]=0;
      rdispls[i]=0;
    }
  }
  // Pass the sindex from each compute task to its associated IO task

  pio_fc_gatherv((void *) iodesc->sindex, iodesc->scount[0], PIO_OFFSET,
		 (void *) srcindex, recvlths, rdispls, PIO_OFFSET,  
		 0, iodesc->subset_comm, maxreq);

  if(ios.ioproc){
    map = (mapsort *) malloc(iodesc->llen * sizeof(mapsort));    
    iomap = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
  }

  // Now pass the compmap, skipping the holes

  PIO_Offset *shrtmap;
  if(maplen>iodesc->scount[0]){
    shrtmap = (PIO_Offset *) calloc(iodesc->scount[0],pio_offset_size);
    j=0;
    for(i=0;i<maplen;i++)
      if(compmap[i]>0)
	shrtmap[j++]=compmap[i];
  }else{
    shrtmap = compmap;
  }

  pio_fc_gatherv((void *) shrtmap, iodesc->scount[0], PIO_OFFSET,
		 (void *) iomap, recvlths, rdispls, PIO_OFFSET,  
		 0, iodesc->subset_comm, maxreq);


  if(shrtmap != compmap)
    free(shrtmap);

  if(ios.ioproc){
    int pos=0;
    int k=0;
    mapsort *mptr;
    for(i=0;i<ntasks;i++){
      for(j=0;j<iodesc->rcount[i];j++){
	mptr = map+k;
	mptr->rfrom = i;
	mptr->soffset = srcindex[pos+j];
	mptr->iomap = iomap[pos+j];
	k++;
      }
      pos += iodesc->rcount[i];
    }
    // sort the mapping, this will transpose the data into IO order        
    qsort(map, iodesc->llen, sizeof(mapsort), compare_offsets); 

    iodesc->rindex = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
    iodesc->rfrom = (int *) calloc(iodesc->llen,sizeof(int));
  }
  int cnt[ntasks];
  int sndlths[ntasks];
  int sdispls[ntasks];
  MPI_Datatype dtypes[ntasks];
  for(i=0;i<ntasks;i++){
    cnt[i]=rdispls[i];

    /* offsets to swapm are in bytes */
    //    rdispls[i]*=pio_offset_size;
    sdispls[i]=0;
    sndlths[i]=0;
    dtypes[i]=PIO_OFFSET;
  }
  sndlths[0]=iodesc->scount[0];
  mapsort *mptr;
  for(i=0;i<iodesc->llen;i++){
    mptr = map+i;
    iodesc->rfrom[i] = mptr->rfrom;
    iodesc->rindex[i]=i;
    iomap[i] = mptr->iomap;
    srcindex[ (cnt[iodesc->rfrom[i]])++   ]=mptr->soffset;
  }

  CheckMPIReturn(MPI_Scatterv((void *) srcindex, recvlths, rdispls, PIO_OFFSET, 
			      (void *) iodesc->sindex, iodesc->scount[0],  PIO_OFFSET,
			      0, iodesc->subset_comm),__FILE__,__LINE__);


  /*  
  pio_swapm((void *) srcindex, recvlths, rdispls, dtypes, 
	    (void *) iodesc->sindex, sndlths, sdispls, dtypes, 
	    iodesc->subset_comm, hs, isend, maxreq);
  */
  if(ios.ioproc){

    /*
    printf("%s %d %d\n",__FILE__,__LINE__,ios.io_rank);
    for(i=0;i<iodesc->llen;i++)
      printf("%d ",iomap[i]);
    printf("\n");
    */
    get_start_and_count_regions(ios.io_comm,iodesc,gsize,iomap);
  
    if(iomap != NULL)
      free(iomap);
    
    if(map != NULL)
      free(map);

    if(srcindex != NULL)
      free(srcindex);
    
    compute_maxIObuffersize(ios.io_comm, iodesc);

    iodesc->nrecvs=ntasks;
  }

  return ierr;


}
  
    
  
