/* C program to run some basic performance tests for the ParallelIO
   library. 

   Ed Hartnett 11/19/15
*/
#include <stdio.h>
#include <mpi.h>
#include <mpe.h>
#include <math.h>
#include <pio.h>

#define NDIMS 1
static const int LEN = 16;
static const int VAL = 42;
static const int ERR_CODE = 99;

/* Some error codes for when things go wrong. */
#define ERR_FILE 1
#define ERR_DUMB 2
#define ERR_ARG 3
#define ERR_MPI 4
#define ERR_MPITYPE 5
#define ERR_LOGGING 6
#define ERR_UPDATE 7
#define ERR_CALC 8
#define ERR_COUNT 9
#define ERR_WRITE 10
#define ERR_SWAP 11
#define ERR_INIT 12
#define ERR_MEM 13
#define ERR_CHECK 14

/* Error handling code derived from an MPI example here: 
   http://www.dartmouth.edu/~rc/classes/intro_mpi/mpi_error_functions.html */
#define MPIERR(e) do {                                                  \
    MPI_Error_string(e, err_buffer, &resultlen);			\
    printf("MPI error, line %d, file %s: %s\n", __LINE__, __FILE__, err_buffer); \
    MPI_Finalize();							\
    return 2;								\
  } while (0) 

#define ERR(e) do {				\
    MPI_Finalize();				\
    return e;					\
  } while (0) 

/* global err buffer for MPI. */
int resultlen;
char err_buffer[MPI_MAX_ERROR_STRING];

/* These are for the event numbers array used to log various events in
 * the program with the MPE library, which produces output for the
 * Jumpshot program. */
#define NUM_EVENTS 7
#define START 0
#define END 1
#define INIT 0
#define UPDATE 1
#define WRITE 2
#define SWAP 3
#define COMM 4
#define CALCULATE 5 
#define INGEST 6

/* This will set up the MPE logging event numbers. */
int
init_logging(int my_rank, int event_num[][NUM_EVENTS])
{
  /* Get a bunch of event numbers. */
  event_num[START][INIT] = MPE_Log_get_event_number();
  event_num[END][INIT] = MPE_Log_get_event_number();
  event_num[START][UPDATE] = MPE_Log_get_event_number();
  event_num[END][UPDATE] = MPE_Log_get_event_number();
  event_num[START][INGEST] = MPE_Log_get_event_number();
  event_num[END][INGEST] = MPE_Log_get_event_number();
  event_num[START][COMM] = MPE_Log_get_event_number();
  event_num[END][COMM] = MPE_Log_get_event_number();
  event_num[START][CALCULATE] = MPE_Log_get_event_number();
  event_num[END][CALCULATE] = MPE_Log_get_event_number();
  event_num[START][WRITE] = MPE_Log_get_event_number();
  event_num[END][WRITE] = MPE_Log_get_event_number();
  event_num[START][SWAP] = MPE_Log_get_event_number();
  event_num[END][SWAP] = MPE_Log_get_event_number();

  /* You should track at least initialization and partitioning, data
   * ingest, update computation, all communications, any memory
   * copies (if you do that), any output rendering, and any global
   * communications. */
  if (!my_rank)
    {
      MPE_Describe_state(event_num[START][INIT], event_num[END][INIT], "init", "yellow");
      MPE_Describe_state(event_num[START][INGEST], event_num[END][INGEST], "ingest", "red");
      MPE_Describe_state(event_num[START][UPDATE], event_num[END][UPDATE], "update", "green");
      MPE_Describe_state(event_num[START][CALCULATE], event_num[END][CALCULATE], "calculate", "orange");
      MPE_Describe_state(event_num[START][WRITE], event_num[END][WRITE], "write", "purple");
      MPE_Describe_state(event_num[START][COMM], event_num[END][COMM], "reduce", "blue");
      MPE_Describe_state(event_num[START][SWAP], event_num[END][SWAP], "swap", "pink");
    }
  return 0;
}

int init_pio(int my_rank, int p, PIO_Offset arrIdxPerPe, int ndim, int *dimLen,
	     int *iodescNCells, PIO_Offset *compdof, int verbose, int *pioIoSystem) {   
  int ret;
  
  /* Create PIO IO system. */
  if (!my_rank && verbose)
    printf("creating PIO IO system...\n");
  if ((ret = PIOc_Init_Intracomm(MPI_COMM_WORLD, p, 1, 1, PIO_REARR_SUBSET, pioIoSystem)))
    ERR(ret);
  if (!my_rank && verbose)
    printf("*pioIoSystem = %d\n", *pioIoSystem);
  
  /* Set up PIO decomposition. */
  PIO_Offset ista = my_rank * arrIdxPerPe;
  PIO_Offset isto = ista + (arrIdxPerPe - 1);
  PIO_Offset localVal = ista;
  for (int i = 0; i < arrIdxPerPe; i++) {
    compdof[i] = localVal;
    localVal++;
  }
  
  /* Create PIO decomposition. */
  if (!my_rank && verbose)
    printf("creating PIO decomposition, arrIdxPerPe = %d\n", (int)arrIdxPerPe);
  if ((ret = PIOc_InitDecomp(*pioIoSystem, PIO_INT, ndim, dimLen, arrIdxPerPe,
			     compdof, iodescNCells, NULL, NULL, NULL)))
    ERR(ret);

  /* Everything worked! */
  return 0;
}

int check_file(int my_rank, PIO_Offset arrIdxPerPe, int iosysid, char *filename,
	       int iotype, int ndims, int *dimlen, int verbose, int *expected) {
  int ncid2;
  int iptype;
  int *read_buffer;
  int ret;

  /* Allocate memory to read the data into. */
  if (!(read_buffer = (int *)malloc(arrIdxPerPe * sizeof(int))))
    ERR(ERR_MEM);
  
  /* Reopen the netCDF file. */
  if ((ret = PIOc_openfile(iosysid, &ncid2, &iotype, filename, NC_NOWRITE)))
    ERR(ret);
   
  /* Check file metadata. */
  int ndims_in_file, nvars_in_file, ngatts_in_file, unlimdimid;
  if ((ret = PIOc_inq(ncid2, &ndims_in_file, &nvars_in_file, &ngatts_in_file,
		      &unlimdimid)))
    ERR(ret);
  if (verbose)
    printf("File %s has %d dims, %d vars, %d global atts, and unlimdimid = %d\n",
	   filename, ndims_in_file, nvars_in_file, ngatts_in_file, unlimdimid);
  if (ndims_in_file != ndims || nvars_in_file != 1 || ngatts_in_file != 0 ||
      unlimdimid != -1)
    ERR(ERR_CHECK);

  /* Read data. */
  PIO_Offset start[ndims];
  PIO_Offset count[ndims];
  start[0] = my_rank * arrIdxPerPe;
  count[0] = arrIdxPerPe;
  if (verbose)
    printf("About to read for processor %d, start[0]=%d count[0]=%d\n",
	   my_rank, start[0], count[0]);
  if ((ret = PIOc_get_vara_int(ncid2, 0, start, count, read_buffer)))
    ERR(ret);
  if (verbose)
    printf("Data read on processor %d.\n", my_rank);

  /* Check data. */
  for (int i = 0; i < count[0]; i++)
    printf("processor %d checking read_buffer[%d] (%d) agains expected[%d] (%d)\n",
	   my_rank, i, read_buffer[i], i + my_rank * arrIdxPerPe, expected[i + my_rank * arrIdxPerPe]);
  for (int i = 0; i < count[0]; i++) 
    if (read_buffer[i] != expected[i + my_rank * arrIdxPerPe])
      ERR(ERR_CHECK);
  if (verbose)
    printf("Data read and checked on processor %d.\n", my_rank);

  /* Close file again. */
  if ((ret = PIOc_closefile(ncid2)))
    ERR(ret);

  /* Free memory. */
  free(read_buffer);
   
  /* Everything worked! */
  return 0;
}  

/* Write the test file. */
int create_file(int my_rank, PIO_Offset arrIdxPerPe, int iosysid, char *filename,
	       int iotype, int ndims, int *dimlen, int verbose, int *data_buffer) {
  int ncid2;
  int iptype;
  int ret;
  int pioDimId, pioVarId;
  int pioFileDesc;

  if (verbose)
    printf("About to create file %s.\n", filename);
  /* Create the netCDF file. */
  if ((ret = PIOc_createfile(iosysid, &pioFileDesc, &iotype,
			     filename, PIO_CLOBBER)))
    ERR(ret);
  
  /* Define metadata. */
  if ((ret = PIOc_def_dim(pioFileDesc, "x", (PIO_Offset)dimlen[0], &pioDimId)))
    ERR(ret);
  if ((ret = PIOc_def_var(pioFileDesc, "foo", PIO_INT, 1, &pioDimId, &pioVarId)))
    ERR(ret);
  if ((ret = PIOc_enddef(pioFileDesc)))
    ERR(ret);

  /* Write data to file. */
  PIO_Offset start[NDIMS], count[NDIMS];
  start[0] = my_rank * arrIdxPerPe;
  count[0] = arrIdxPerPe;
  if (verbose) {
    printf("About to write for processor %d, start[0]=%d count[0]=%d\n",
	   my_rank, start[0], count[0]);
    for(int i = start[0]; i < count[0]; i++)
      printf("rank %d data_buffer[%d]=%d\n", my_rank, i, data_buffer[i]);
  }
  if ((ret = PIOc_put_vara_int(pioFileDesc, pioVarId, start, count, data_buffer)))
    ERR(ret);

  /* Close the netCDF output file. */
  if (!my_rank && verbose)
    printf("closing file...\n");
  if ((ret = PIOc_closefile(pioFileDesc)))
    ERR(ret);

  /* Everything worked! */
  return 0;
}

int main(int argc, char* argv[]) {
  int p, my_rank;
  int event_num[2][NUM_EVENTS];
  int verbose = 1;
  int pioIoSystem;
  int dimlen[NDIMS];
  int iodescNCells;
  dimlen[0] = LEN;
  PIO_Offset *compdof = NULL;
  PIO_Offset arrIdxPerPe;
  int *readBuffer = NULL;
  char filename[] = "perftest.nc";
  int iotype = PIO_IOTYPE_NETCDF;
  int *data_buffer;
  int ret;

  /* Initialize MPI. */
  MPI_Init(&argc, &argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  /* Learn my rank and the total number of processors. */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  arrIdxPerPe = LEN/p;

  /* Turn off comments in all but task 0. */
  /* if (verbose && my_rank) */
  /*   verbose = 0; */
  if (verbose)
    printf("initialized for %d processors, %d array index per processor.\n", p, arrIdxPerPe);

  /* Allocate memory for data. */
  if (!(data_buffer = (int *)malloc(arrIdxPerPe * sizeof(int))))
    ERR(ERR_MEM);
  for (int i = 0; i < arrIdxPerPe; i++) 
    data_buffer[i] = my_rank * 1000 + i;

  /* Allocate array for PIO decomposition data. */
  if (!(compdof = (PIO_Offset *)malloc(arrIdxPerPe * sizeof(PIO_Offset))))
    ERR(ERR_MEM);

  /* Intialize MPE logging. */
  if ((ret = MPE_Init_log()))
    ERR(ret);
  if (init_logging(my_rank, event_num))
    ERR(ERR_LOGGING);

  /* Put a barrier here so every process starts file initialization
     at the same time. */
  if ((ret = MPI_Barrier(MPI_COMM_WORLD)))
    MPIERR(ret);

  /* Start MPE logging for file initialization. */
  if ((ret = MPE_Log_event(event_num[START][INIT], 0, "start init")))
    MPIERR(ret);

  /* Initialize the PIO system. */
  if ((ret = init_pio(my_rank, p, arrIdxPerPe, NDIMS, dimlen, &iodescNCells, compdof,
		      verbose, &pioIoSystem)))
    ERR(ret);

  /* We are done with initialization. */
  if ((ret = MPE_Log_event(event_num[END][INIT], 0, "end init")))
    MPIERR(ret);

  /* Create the file. */
  if ((ret = create_file(my_rank, arrIdxPerPe, pioIoSystem, filename, iotype,
			NDIMS, dimlen, verbose, data_buffer)))
    ERR(ret);
  
  /* Check that the output file is correct. */
  /* if ((ret = check_file(my_rank, arrIdxPerPe, pioIoSystem, filename, iotype, */
  /* 			NDIMS, dimlen, verbose, data_buffer))) */
  /*   ERR(ret); */
   
  /* Clean up PIO decomp. */
  if (verbose)
    printf("cleaning up PIO decomposition...\n");
  if ((ret = PIOc_freedecomp(pioIoSystem, iodescNCells)))
    ERR(ret);
  free(compdof);

  /* Free memory used for data. */
  free(data_buffer);

  /* Clean up PIO IO system. */
  if (verbose)
    printf("cleaning up PIO IO system...\n");
  if ((ret = PIOc_finalize(pioIoSystem)))
    ERR(ret);

  /* Shut down MPI. */
  if (verbose)
    printf("task %d finalizing...\n", my_rank);
  MPI_Finalize();
   
  return 0;
}
