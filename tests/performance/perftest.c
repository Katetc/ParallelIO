/* C program to run some basic performance tests for the ParallelIO
   library. 

   Ed Hartnett 11/19/15
*/
#include <stdio.h>
#include <mpi.h>
#include <mpe.h>
#include <math.h>
#include <pio.h>

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

/* Error handling code derived from an MPI example here: 
   http://www.dartmouth.edu/~rc/classes/intro_mpi/mpi_error_functions.html */
#define MPIERR(e) do {                                                  \
      MPI_Error_string(e, err_buffer, &resultlen);                      \
      printf("MPI error, line %d, file %s: %s\n", __LINE__, __FILE__, err_buffer); \
      MPI_Finalize();                                                   \
      return 2;                                                         \
   } while (0) 

#define ERR(e) do {                                                     \
      MPI_Finalize();                                                   \
      return e;                                                         \
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

int init_pio(int my_rank, int p, int ndim, int *dimLen, int *iodescNCells, PIO_Offset **compdof,
	     int **dataBuffer, int **readBuffer, int verbose, int *pioIoSystem) {   
  int ret;
  
  /* Create PIO IO system. */
  if (!my_rank && verbose)
    printf("creating PIO IO system...\n");
  if ((ret = PIOc_Init_Intracomm(MPI_COMM_WORLD, p, 1, 1, PIO_REARR_SUBSET, pioIoSystem)))
    ERR(ret);
  if (!my_rank && verbose)
    printf("*pioIoSystem = %d\n", *pioIoSystem);
  
  /* Set up PIO decomposition. */
  PIO_Offset arrIdxPerPe = LEN/p;
  PIO_Offset ista = my_rank * arrIdxPerPe;
  PIO_Offset isto = ista + (arrIdxPerPe - 1);
  *dataBuffer = (int *)malloc(arrIdxPerPe * sizeof(int));
  *readBuffer = (int *)malloc(arrIdxPerPe * sizeof(int));
  *compdof = (PIO_Offset *)malloc(arrIdxPerPe * sizeof(PIO_Offset));
  PIO_Offset localVal = ista;
  for (int i = 0; i < arrIdxPerPe; i++) {
    *dataBuffer[i] = my_rank + 42;
    *compdof[i] = localVal;
    *readBuffer[i] = 99;
    localVal++;
  }
  
  /* Create PIO decomposition. */
  if (!my_rank && verbose)
    printf("creating PIO decomposition, arrIdxPerPe = %d\n", (int)arrIdxPerPe);
  if ((ret = PIOc_InitDecomp(*pioIoSystem, PIO_INT, ndim, dimLen, arrIdxPerPe,
			     *compdof, iodescNCells, NULL, NULL, NULL)))
    ERR(ret);

  /* Everything worked! */
  return 0;
}

#define NDIM 1

int main(int argc, char* argv[]) {
   int p, my_rank;
   int event_num[2][NUM_EVENTS];
   int verbose = 1;
   int pioIoSystem;
   int dimLen[NDIM];
   int iodescNCells;
   dimLen[0] = LEN;
   PIO_Offset *compdof = NULL;
   int *dataBuffer = NULL;
   int *readBuffer = NULL;
   int ret;

   /* Initialize MPI. */
   if (!my_rank && verbose)
     printf("now initializing...\n");
   MPI_Init(&argc, &argv);
   MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

   /* Learn my rank and the total number of processors. */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   if (!my_rank && verbose)
     printf("initialized for %d processors.\n", p);

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
   if ((ret = init_pio(my_rank, p, NDIM, dimLen, &iodescNCells, &compdof,
		       &dataBuffer, &readBuffer, verbose, &pioIoSystem)))
     ERR(ret);

/*    /\* Create the file. *\/ */
/*    if (!my_rank && verbose) */
/*      printf("creating file...\n"); */
/*    int pioFileDesc; */
/*    int iotype = PIO_IOTYPE_NETCDF; */
/*    if ((ret = PIOc_createfile(pioIoSystem, &pioFileDesc, &iotype, */
/* 			      "perftest.nc", PIO_CLOBBER))) */
/*      ERR(ret); */

/*    /\* Define metadata. *\/ */
/*    int pioDimId, pioVarId; */
/*    if ((ret = PIOc_def_dim(pioFileDesc, "x", (PIO_Offset)dimLen[0], &pioDimId))) */
/*      ERR(ret); */
/*    if ((ret = PIOc_def_var(pioFileDesc, "foo", PIO_INT, 1, &pioDimId, &pioVarId))) */
/*      ERR(ret); */
/*    if ((ret = PIOc_enddef(pioFileDesc))) */
/*      ERR(ret); */

/*    /\* Create some data. *\/ */
/* #define DATA_LEN 3000    */
/*    if (!my_rank && verbose) */
/*      printf("doing calculations...\n"); */
/*    double data[DATA_LEN]; */
/*    for (int i = 0; i < DATA_LEN; i++) */
/*      data[i] = sqrt(i); */

   /* Write data to file. */

   /* We are done with initialization. */
   if ((ret = MPE_Log_event(event_num[END][INIT], 0, "end init")))
     MPIERR(ret);

   /* /\* Close the netCDF output file. *\/ */
   /* if (!my_rank && verbose) */
   /*   printf("closing file...\n"); */
   /* if ((ret = PIOc_closefile(pioFileDesc))) */
   /*   ERR(ret); */

   /* Clean up PIO decomp. */
   if (!my_rank && verbose)
     printf("cleaning up PIO decomposition...\n");
   if ((ret = PIOc_freedecomp(pioIoSystem, iodescNCells)))
     ERR(ret);
   /*   free(compdof);*/
   /* free(dataBuffer); */
   /* free(readBuffer); */

   /* Clean up PIO IO system. */
   if (!my_rank && verbose)
     printf("cleaning up PIO IO system...\n");
   if ((ret = PIOc_finalize(pioIoSystem)))
     ERR(ret);

   /* Shut down MPI. */
   MPI_Finalize();
   
   return 0;
}
