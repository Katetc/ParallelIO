/** @file 
 * Support functions.
 */
#include <config.h>
#if PIO_ENABLE_LOGGING
#include <stdarg.h>
#include <unistd.h>
#endif /* PIO_ENABLE_LOGGING */
#include <pio.h>
#include <pio_internal.h>

#include <execinfo.h>
#define versno 2001

#if PIO_ENABLE_LOGGING
#define MAX_LOG_MSG 1024
#define MAX_RANK_STR 12
#define ERROR_PREFIX "ERROR: " 
int pio_log_level = 0;
int my_rank;
FILE *LOG_FILE;
#endif /* PIO_ENABLE_LOGGING */

/** Return a string description of an error code. If zero is passed, a
 * null is returned.
 *
 * @param pioerr the error code returned by a PIO function call. 
 * @param errmsg Pointer that will get the error message. It will be
 * PIO_MAX_NAME chars or less.
 *
 * @return 0 on success
 */
int
PIOc_strerror(int pioerr, char *errmsg)
{

    /* System error? */
    if(pioerr > 0)
    {
	const char *cp = (const char *)strerror(pioerr);
	if (cp)
	    strncpy(errmsg, cp, PIO_MAX_NAME);
	else
	    strcpy(errmsg, "Unknown Error");
    }
    else if (pioerr == PIO_NOERR)
    {
	strcpy(errmsg, "No error");
    }
    else if (pioerr <= NC2_ERR && pioerr >= NC4_LAST_ERROR)     /* NetCDF error? */
    {
#if defined( _PNETCDF) || defined(_NETCDF)
	strncpy(errmsg, nc_strerror(pioerr), NC_MAX_NAME);
#else /* defined( _PNETCDF) || defined(_NETCDF) */
	strcpy(errmsg, "NetCDF error code, PIO not built with netCDF.");
#endif /* defined( _PNETCDF) || defined(_NETCDF) */
    }
    else
    {
	/* Handle PIO errors. */
	switch(pioerr) {
	case PIO_EBADIOTYPE:
	    strcpy(errmsg, "Bad IO type");
	    break;
	default:
	    strcpy(errmsg, "unknown PIO error");
	}
    }

    return PIO_NOERR;
}

/** Set the logging level. Set to -1 for nothing, 0 for errors only, 1
 * for important logging, and so on. Log levels below 1 are only
 * printed on the io/component root. If the library is not built with
 * logging, this function does nothing. */
int PIOc_set_log_level(int level)
{
#if PIO_ENABLE_LOGGING
    char log_filename[NC_MAX_NAME];
    
    printf("setting log level to %d\n", level);
    pio_log_level = level;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    sprintf(log_filename, "pio_log_%d.txt", my_rank);
    LOG_FILE = fopen(log_filename, "w");
#endif /* PIO_ENABLE_LOGGING */
    return PIO_NOERR;
}

#if PIO_ENABLE_LOGGING
/** This function prints out a message, if the severity of the message
   is lower than the global pio_log_level. To use it, do something
   like this:
   
   pio_log(0, "this computer will explode in %d seconds", i);

   After the first arg (the severity), use the rest like a normal
   printf statement. Output will appear on stdout.
   This function is heavily based on the function in section 15.5 of
   the C FAQ. 
*/
void 
pio_log(int severity, const char *fmt, ...)
{
   va_list argp;
   int t;
   char msg[MAX_LOG_MSG];
   char *ptr = msg;
   char rank_str[MAX_RANK_STR];

   /* If the severity is greater than the log level, we don't print
      this message. */
   if (severity > pio_log_level)
      return;

   /* If the severity is 0, only print on rank 0. */
   if (severity < 1 && my_rank != 0)
       return;

   /* If the severity is zero, this is an error. Otherwise insert that
      many tabs before the message. */
   if (!severity)
   {
       strcpy(ptr, ERROR_PREFIX);
       ptr += strlen(ERROR_PREFIX);
   }
   for (t = 0; t < severity; t++)
       strcpy(ptr++, "\t");

   /* Show the rank. */
   sprintf(rank_str, "%d ", my_rank);
   strcpy(ptr, rank_str);
   ptr += strlen(rank_str);
   
   /* Print out the variable list of args with vprintf. */
   va_start(argp, fmt);
   vsprintf(ptr, fmt, argp);
   va_end(argp);
   
   /* Put on a final linefeed. */
   ptr = msg + strlen(msg);
   strcpy(ptr, "\n\0");

   /* Send message to stdout. */
   fprintf(stdout, "%s", msg);

   /* Send message to log file. */
   fprintf(LOG_FILE, "%s", msg);

   /* Ensure an immediate flush of stdout. */
   fflush(stdout);
   fflush(LOG_FILE);
}
#endif /* PIO_ENABLE_LOGGING */

static pio_swapm_defaults swapm_defaults;
bool PIO_Save_Decomps=false;
/**
 ** @brief Get PIO environment variables
 **
 **/
void pio_get_env(void)
{
  char *envptr;
  extern bufsize PIO_CNBUFFER_LIMIT;
  envptr = getenv("PIO_Save_Decomps");
  
  if(envptr != NULL && (strcmp(envptr,"true")==0)){
    PIO_Save_Decomps=true;
  }
  swapm_defaults.nreqs = 0;
  swapm_defaults.handshake=false;
  swapm_defaults.isend=false;

  envptr = getenv("PIO_SWAPM");
  if(envptr != NULL){
    char *token = strtok(envptr, ":");
    
    swapm_defaults.nreqs = atoi(token);

    token = strtok(NULL, ":");

    if((token!=NULL) && strcmp(token,"t")==0){
      swapm_defaults.handshake = true;
    }
    token = strtok(NULL, ":");
    
    if((token!=NULL) && strcmp(token,"t")==0){
      swapm_defaults.isend = true;
    }
    //printf("nreqs %d handshake %d isend %d\n",swapm_defaults.nreqs, swapm_defaults.handshake, swapm_defaults.isend);
  }
  envptr = getenv("PIO_CNBUFFER_LIMIT");
  if(envptr != NULL){
    int mult=1;
    if(strchr(envptr,"M") != NULL){
      mult = 1000000;
    }else if(strchr(envptr,"K") != NULL){
      mult = 1000;
    }
    PIO_CNBUFFER_LIMIT=(bufsize) atoll(envptr)*mult;
    
  }

 

}


     
/* Obtain a backtrace and print it to stderr. */
void print_trace (FILE *fp)
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;
  
  if(fp==NULL)
    fp = stderr;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);
  
  fprintf (fp,"Obtained %zd stack frames.\n", size);
     
  for (i = 0; i < size; i++)
    fprintf (fp,"%s\n", strings[i]);
     
  free (strings);
}

void piomemerror(iosystem_desc_t ios, size_t req, char *fname, const int line){
  char msg[80];
  sprintf(msg,"out of memory requesting: %ld",req);
  cn_buffer_report(ios,false);
  piodie(msg,fname,line);
}


void piodie(const char *msg,const char *fname, const int line){
  fprintf(stderr,"Abort with message %s in file %s at line %d\n",msg,fname,line);
  
  print_trace(stderr);
#ifdef MPI_SERIAL
  abort();
#else
  MPI_Abort(MPI_COMM_WORLD, -1);
#endif
}


void pioassert(_Bool expression, const char *msg, const char *fname, const int line)
{
#ifndef NDEBUG
  if(! expression){
    piodie(msg,fname,line);
  }
#endif  

}

/** Handle MPI errors. An error message is sent to stderr, then the
  check_netcdf() function is called with PIO_EIO.

  @param file pointer to the file_desc_t info. May be NULL, in which
  case check_netcdf() is not called.
  @param mpierr the MPI return code to handle
  @param filename the name of the code file where error occured.
  @param line the line of code where error occured.
  @return PIO_NOERR for no error, otherwise PIO_EIO.
 */
int check_mpi(file_desc_t *file, const int mpierr, const char *filename,
	       const int line)
{
    if (mpierr)
    {
	char errstring[MPI_MAX_ERROR_STRING];
	int errstrlen;

	/* If we can get an error string from MPI, print it to stderr. */
	if (!MPI_Error_string(mpierr, errstring, &errstrlen))
	    fprintf(stderr, "MPI ERROR: %s in file %s at line %d\n",
		    errstring, filename, line);

	/* Handle all MPI errors as PIO_EIO. */
	if (file)
	    check_netcdf(file, PIO_EIO, filename, line);
	return PIO_EIO;
    }
    return PIO_NOERR;
}

/** Check the result of a netCDF API call. 
 * 
 * @param file pointer to the PIO structure describing this file.
 * @param status the return value from the netCDF call.
 * @param fname the name of the code file. 
 * @param line the line number of the netCDF call in the code. 
 * 
 * @return the error code
*/
int check_netcdf(file_desc_t *file, int status, const char *fname, const int line){
  iosystem_desc_t *ios;
  int ierr;
  char errstr[160];
  ios = file->iosystem;
  ierr = PIO_NOERR;

  switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
  case PIO_IOTYPE_NETCDF4P:
  case PIO_IOTYPE_NETCDF4C:
#endif
  case PIO_IOTYPE_NETCDF:
    if(ios->iomaster){
      if(status != NC_NOERR && (ios->error_handler == PIO_INTERNAL_ERROR))
         piodie(nc_strerror(status),fname,line);	
//	fprintf(stderr,"NETCDF ERROR: %s %s %d\n",nc_strerror(status),fname,line);
    }
    if(ios->error_handler == PIO_INTERNAL_ERROR){
      if(status != NC_NOERR)	
	MPI_Abort(MPI_COMM_WORLD,status);
      // abort
    }else if(ios->error_handler==PIO_BCAST_ERROR){
      ierr =MPI_Bcast(&status, 1, MPI_INTEGER, ios->ioroot, ios->my_comm);
    }
    break;
#endif
#ifdef _PNETCDF
  case PIO_IOTYPE_PNETCDF:
    if(status != NC_NOERR && (ios->error_handler == PIO_INTERNAL_ERROR)) {
      //	  fprintf(stderr,"PNETCDF ERROR: %s %s %d\n",ncmpi_strerror(status),fname,line);
      piodie(ncmpi_strerror(status),fname,line);
    }
    if(ios->error_handler==PIO_BCAST_ERROR){
      ierr = MPI_Bcast(&status, 1, MPI_INTEGER, ios->ioroot, ios->my_comm);
    }
    break;
#endif
  default:
    ierr = iotype_error(file->iotype,__FILE__,__LINE__);
  } 
  return status;
}

int iotype_error(const int iotype, const char *fname, const int line)
{
    fprintf(stderr, "ERROR: iotype %d not defined in build %s %d\n", iotype, fname,line);
    return(PIO_EBADIOTYPE);
}

io_region *alloc_region(const int ndims)
{
  io_region *region;

  region = (io_region *) bget(sizeof(io_region));
  region->start = (PIO_Offset *) bget(ndims* sizeof(PIO_Offset));
  region->count = (PIO_Offset *) bget(ndims* sizeof(PIO_Offset));
  region->loffset = 0;
  region->next=NULL;
  for(int i=0;i< ndims; i++){
    region->start[i] = 0;
    region->count[i] = 0;
  }
  return region;
}

io_desc_t *malloc_iodesc(const int piotype, const int ndims)
{
  io_desc_t *iodesc;
  iodesc = (io_desc_t *) bget(sizeof(io_desc_t));

  if(iodesc == NULL)
    fprintf(stderr,"ERROR: allocation error \n");

  switch(piotype){
  case PIO_REAL:			
    iodesc->basetype=MPI_FLOAT;
    break;
  case PIO_DOUBLE:
    iodesc->basetype=MPI_DOUBLE;
    break;
  case PIO_CHAR:
    iodesc->basetype=MPI_CHAR;
    break;
  case PIO_INT:   
  default:
    iodesc->basetype = MPI_INTEGER;
    break;
  }    
  iodesc->rearranger = 0;
  iodesc->maxregions=1;
  iodesc->rfrom = NULL;
  iodesc->scount = NULL;
  iodesc->rtype = NULL;
  iodesc->stype = NULL;
  iodesc->num_stypes = 0;
  iodesc->sindex = NULL;
  iodesc->rindex = NULL;
  iodesc->rcount = NULL;
  iodesc->ioid=-1;
  iodesc->llen=0;
  iodesc->maxiobuflen=0;
  iodesc->holegridsize=0;
  iodesc->maxbytes=0;
  iodesc->ndims = ndims;
  iodesc->firstregion = alloc_region(ndims);
  iodesc->fillregion = NULL;

  iodesc->handshake=swapm_defaults.handshake;
  iodesc->isend=swapm_defaults.isend;
  iodesc->max_requests=swapm_defaults.nreqs;

  return iodesc;
}

void free_region_list(io_region *top)
{
  io_region *ptr, *tptr;

  ptr = top;
  while(ptr != NULL) {
    if(ptr->start != NULL)
      brel(ptr->start);
    if(ptr->count != NULL)
      brel(ptr->count);
    tptr=ptr;
    ptr=ptr->next;
    brel(tptr);      
  }

}


int PIOc_freedecomp(int iosysid, int ioid)
{
  iosystem_desc_t *ios;
  io_desc_t *iodesc;
  int i;

  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;

  iodesc = pio_get_iodesc_from_id(ioid);
  if(iodesc == NULL)
    return PIO_EBADID;

  if(iodesc->rfrom != NULL)
    brel(iodesc->rfrom);
  if(iodesc->rtype != NULL){
    for(i=0; i<iodesc->nrecvs; i++){
      if(iodesc->rtype[i] != MPI_DATATYPE_NULL){
        MPI_Type_free(iodesc->rtype+i);
      }
    }
    brel(iodesc->rtype);
  }
  if(iodesc->stype != NULL){
    for(i=0; i<iodesc->num_stypes; i++){
      if(iodesc->stype[i] != MPI_DATATYPE_NULL){
        MPI_Type_free(iodesc->stype+i);
      }
    }
    iodesc->num_stypes = 0;
    brel(iodesc->stype);
  }
  if(iodesc->scount != NULL)
    brel(iodesc->scount);
  if(iodesc->rcount != NULL)
    brel(iodesc->rcount);
  if(iodesc->sindex != NULL)
    brel(iodesc->sindex);
  if(iodesc->rindex != NULL)
    brel(iodesc->rindex);


  if(iodesc->firstregion != NULL)
    free_region_list(iodesc->firstregion);

  if(iodesc->rearranger == PIO_REARR_SUBSET){
    MPI_Comm_free(&(iodesc->subset_comm));
  }

  return pio_delete_iodesc_from_list(ioid);


}

int PIOc_readmap(const char file[], int *ndims, int *gdims[], PIO_Offset *fmaplen, PIO_Offset *map[], const MPI_Comm comm)
{
  int npes, myrank;  
  int rnpes, rversno;
  int j;
  int *tdims;
  PIO_Offset *tmap;
  MPI_Status status;
  PIO_Offset maplen;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);
  
  if(myrank == 0) {
    FILE *fp = fopen(file, "r");
    if(fp==NULL)
      piodie("Failed to open dof file",__FILE__,__LINE__);
    
    fscanf(fp,"version %d npes %d ndims %d\n",&rversno, &rnpes,ndims);

    if(rversno != versno)
      piodie("Attempt to read incompatable map file version",__FILE__,__LINE__);

    if(rnpes < 1 || rnpes > npes)
      piodie("Incompatable pe count in map file ",__FILE__,__LINE__);

    MPI_Bcast(&rnpes, 1, MPI_INT, 0, comm);
    MPI_Bcast(ndims, 1, MPI_INT, 0, comm);
    tdims = (int *) calloc((*ndims), sizeof(int));
    for(int i=0;i<(*ndims);i++)
      fscanf(fp,"%d ",tdims+i);

    MPI_Bcast(tdims, (*ndims), MPI_INT, 0, comm);

    for(int i=0; i< rnpes; i++){
      fscanf(fp,"%d %ld",&j,&maplen);
      if( j != i)  // Not sure how this could be possible
	piodie("Incomprehensable error reading map file ",__FILE__,__LINE__);

      tmap = (PIO_Offset *) malloc(maplen*sizeof(PIO_Offset));
      for(j=0;j<maplen;j++)
	fscanf(fp,"%ld ",tmap+j);
      
      if(i>0){
	MPI_Send(&maplen, 1, PIO_OFFSET, i, i+npes, comm);
	MPI_Send(tmap, maplen, PIO_OFFSET, i, i, comm);
	free(tmap);
      }else{
	*map = tmap;
	*fmaplen = maplen;
      }
    }
    fclose(fp);
  }else{
    MPI_Bcast(&rnpes, 1, MPI_INT, 0, comm);
    MPI_Bcast(ndims, 1, MPI_INT, 0, comm);
    tdims = (int *) calloc((*ndims), sizeof(int));
    MPI_Bcast(tdims, (*ndims), MPI_INT, 0, comm);

    if(myrank<rnpes){
      MPI_Recv(&maplen, 1, PIO_OFFSET, 0, myrank+npes, comm, &status);
      tmap = (PIO_Offset *) malloc(maplen*sizeof(PIO_Offset));
      MPI_Recv(tmap, maplen, PIO_OFFSET, 0, myrank, comm, &status);
      *map = tmap;
    }else{
      tmap=NULL;
      maplen=0;
    }
    *fmaplen = maplen;
  }      
  *gdims = tdims;
  return PIO_NOERR;
}

int PIOc_readmap_from_f90(const char file[],int *ndims, int *gdims[], PIO_Offset *maplen, PIO_Offset *map[], const int f90_comm)
{
  int ierr;

  ierr = PIOc_readmap(file, ndims, gdims, maplen, map, MPI_Comm_f2c(f90_comm));

  return(ierr);
}


int PIOc_writemap(const char file[], const int ndims, const int gdims[], PIO_Offset maplen, PIO_Offset map[], const MPI_Comm comm)
{
  int npes, myrank;
  PIO_Offset *nmaplen;
  MPI_Status status;
  int i;
  PIO_Offset *nmap;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);
  if(myrank==0)
    nmaplen = (PIO_Offset *) malloc(npes*sizeof(PIO_Offset));
  else
    nmaplen = NULL;

  //  printf("maplen[%d] = %ld\n",myrank,maplen);


  MPI_Gather(&maplen, 1, PIO_OFFSET, nmaplen, 1, PIO_OFFSET, 0, comm);

  if(myrank==0){
    FILE *fp;


    fp = fopen( file, "w");
    if(fp==NULL){
      fprintf(stderr,"Failed to open file %s to write\n",file);
      return PIO_EIO;
    }
    fprintf(fp,"version %d npes %d ndims %d \n",versno, npes,ndims);
    for(i=0;i<ndims;i++){
      fprintf(fp,"%d ",gdims[i]);
    }
    fprintf(fp,"\n");    
    fprintf(fp,"0 %ld\n",nmaplen[0]);
    for( i=0;i<nmaplen[0];i++)
      fprintf(fp,"%ld ",map[i]);
    fprintf(fp,"\n");    
    for( i=1; i<npes;i++){
      nmap = (PIO_Offset *) malloc(nmaplen[i] * sizeof(PIO_Offset));

      MPI_Send(&i, 1, MPI_INT, i, npes+i, comm);
      MPI_Recv(nmap, nmaplen[i], PIO_OFFSET, i, i, comm, &status);

      fprintf(fp,"%d %ld\n",i,nmaplen[i]);
      for(int j=0;j<nmaplen[i];j++)
	fprintf(fp,"%ld ",nmap[j]);
      fprintf(fp,"\n");

      free(nmap);
      
    }

    fprintf(fp,"\n");
    print_trace(fp);

    fclose(fp);
  }else{
    MPI_Recv(&i, 1, MPI_INT, 0, npes+myrank, comm, &status);
    MPI_Send(map, maplen, PIO_OFFSET, 0, myrank, comm);
  }
  return PIO_NOERR;
}

int PIOc_writemap_from_f90(const char file[], const int ndims, const int gdims[], 
			   const PIO_Offset maplen, const PIO_Offset map[], const int f90_comm)
{
  // printf("%s %d %s %ld\n",__FILE__,__LINE__,file,maplen);
  return(PIOc_writemap(file, ndims, gdims, maplen, map, MPI_Comm_f2c(f90_comm)));
}
/*
#ifdef BGQ
#include <stdio.h>
#include <spi/include/kernel/memory.h>
void print_memusage()
{
  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;

  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  printf("Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024), (double)heapavail/(1024*1024));
  printf("Allocated stack: %.2f MB, avail. stack: %.2f MB\n", (double)stack/(1024*1024), (double)stackavail/(1024*1024));
  printf("Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));

  return; 
}
#endif
*/
