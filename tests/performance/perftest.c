/* @file
 * 
 * C program to run some basic performance tests for the ParallelIO
 *  library. 
 *
 * @author Ed Hartnett 
 * 11/19/15
 */
#include <stdio.h>
#include <mpi.h>
#include <mpe.h>
#include <math.h>
#include <pio.h>

/** Number of dimensions for output variable. */
#define NDIMS 2

/** Length of the X dimension. */
static const int X_LEN = 4;

/** Length of the y dimension. */
static const int Y_LEN = 4;

/** Starting value for sample data. */
static const int VAL = 42;

/** Error code returned if there is a problem. */
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
	return 2;							\
    } while (0) 

#define ERR(e) do {				\
	MPI_Finalize();				\
	return e;				\
    } while (0) 

/** Lentgh of error buffer, needed for MPI error handler. */
int resultlen;

/** Global err buffer for MPI. */
char err_buffer[MPI_MAX_ERROR_STRING];

/** Number of events for the MPE library to measure. */
#define NUM_EVENTS 6

/** Start an MPE event. */
#define START 0

/** End an MPE event. */
#define END 1

/** Event for initialization of library. */
#define INIT 0

/** Event for creation of sample file. */
#define CREATE 1

/** Event for caluclations. */
#define CALCULATE 2

/** Event for writing data to sample file. */
#define WRITE 3

/** Event for reading data from sample file. */
#define READ 4

/** Event for cleanup of resourses. */
#define CLEANUP 5

/** Set up the MPE event numbers array. This array is used to log
 * various events in the program with the MPE library, which produces
 * output for the Jumpshot program. 
 * @param my_rank: rank of processor.
 * @param event_num: 2D array used to hold the event numbers MPE
 * needs.
 *
 * @returns: 0 for success, non-zero for error. 
 */
int
init_logging(int my_rank, int event_num[][NUM_EVENTS])
{
    /* Get a bunch of event numbers. */
    event_num[START][INIT] = MPE_Log_get_event_number();
    event_num[END][INIT] = MPE_Log_get_event_number();
    event_num[START][CREATE] = MPE_Log_get_event_number();
    event_num[END][CREATE] = MPE_Log_get_event_number();
    event_num[START][CALCULATE] = MPE_Log_get_event_number();
    event_num[END][CALCULATE] = MPE_Log_get_event_number();
    event_num[START][WRITE] = MPE_Log_get_event_number();
    event_num[END][WRITE] = MPE_Log_get_event_number();
    event_num[START][READ] = MPE_Log_get_event_number();
    event_num[END][READ] = MPE_Log_get_event_number();
    event_num[START][CLEANUP] = MPE_Log_get_event_number();
    event_num[END][CLEANUP] = MPE_Log_get_event_number();

    if (!my_rank)
    {
	MPE_Describe_state(event_num[START][INIT], event_num[END][INIT], "init", "yellow");
	MPE_Describe_state(event_num[START][CREATE], event_num[END][CREATE], "create", "red");
	MPE_Describe_state(event_num[START][CALCULATE], event_num[END][CALCULATE], "calculate", "orange");
	MPE_Describe_state(event_num[START][WRITE], event_num[END][WRITE], "write", "purple");
	MPE_Describe_state(event_num[START][READ], event_num[END][READ], "read", "blue");
	MPE_Describe_state(event_num[START][CLEANUP], event_num[END][CLEANUP], "cleanup", "pink");
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
    PIO_Offset ista = my_rank * arrIdxPerPe + 1;
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
	       int iotype, int iodescNCells, int ndims, int *dimlen, int verbose,
	       int *expected) {
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
    if (verbose)
	printf("About to read for processor %d\n", my_rank);
    if ((ret = PIOc_read_darray(ncid2, 0, iodescNCells, arrIdxPerPe, read_buffer)))
	ERR(ret);

    if (verbose)
	printf("Data read on processor %d.\n", my_rank);

    /* Check data. */
    for (int i = 0; i < arrIdxPerPe; i++)
    	if (read_buffer[i] != expected[i])
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

/** Create the output file, including metadata. */
int create_file(int my_rank, PIO_Offset arrIdxPerPe, int iosysid, int iodescNCells,
		char *filename, int iotype, int ndims, int *dimlen, int verbose,
		int *data_buffer, int *ncid) {
    int ncid2;
    int iptype;
    int ret;
    int pioDimId, pioVarId;

    if (verbose)
	printf("rank %d About to create file %s.\n", my_rank, filename);
    /* Create the netCDF file. */
    if ((ret = PIOc_createfile(iosysid, ncid, &iotype, filename, PIO_CLOBBER)))
	ERR(ret);
  
    /* Define metadata. */
    if ((ret = PIOc_def_dim(*ncid, "x", (PIO_Offset)dimlen[0], &pioDimId)))
	ERR(ret);
    if ((ret = PIOc_def_var(*ncid, "foo", PIO_INT, 1, &pioDimId, &pioVarId)))
	ERR(ret);
    if ((ret = PIOc_enddef(*ncid)))
	ERR(ret);

    /* Everything worked! */
    return 0;
}

/** Write data to the output file. */
int write_data(int my_rank, PIO_Offset arrIdxPerPe, int iosysid, int iodescNCells,
	       char *filename, int iotype, int ndims, int *dimlen, int verbose,
	       int *data_buffer, int ncid) {
    int ncid2;
    int iptype;
    int ret;
    int pioDimId, pioVarId;

    /* Write data to file. */
    if (verbose) 
	printf("About to write for processor %d\n", my_rank);
    if ((ret = PIOc_write_darray(ncid, pioVarId, iodescNCells,
				 (PIO_Offset)arrIdxPerPe, data_buffer, NULL)))
	ERR(ret);
      
    /* Close the netCDF output file. */
    if (!my_rank && verbose)
	printf("closing file...\n");
    if ((ret = PIOc_closefile(ncid)))
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
    PIO_Offset *compdof = NULL;
    PIO_Offset arrIdxPerPe;
    int *readBuffer = NULL;
    char filename[] = "perftest.nc";
    int iotype = PIO_IOTYPE_NETCDF;
    int *data_buffer;
    int ncid;
    int ret;

    /* Specify the lengths of each dimension. */
    dimlen[0] = X_LEN;
    dimlen[1] = Y_LEN;

    /* Initialize MPI. */
    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    /* Learn my rank and the total number of processors. */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    arrIdxPerPe = (X_LEN * Y_LEN)/p;

    /* Turn off comments in all but task 0. */
    /* if (verbose && my_rank) */
    /*   verbose = 0; */
    if (verbose)
	printf("rank %d initialized for %d processors, %d array index per processor.\n",
	       my_rank, p, arrIdxPerPe);

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

    /* Start MPE logging for initialization. */
    if ((ret = MPE_Log_event(event_num[START][INIT], 0, "start init")))
	MPIERR(ret);

    /* Initialize the PIO system. */
    if ((ret = init_pio(my_rank, p, arrIdxPerPe, NDIMS, dimlen, &iodescNCells, compdof,
			verbose, &pioIoSystem)))
	ERR(ret);

    /* End MPE logging for initialization. */
    if ((ret = MPE_Log_event(event_num[END][INIT], 0, "end init")))
	MPIERR(ret);

    /* Start MPE logging for file creation. */
    if ((ret = MPE_Log_event(event_num[CREATE][INIT], 0, "start create")))
	MPIERR(ret);

    /* Create the file. */
    if ((ret = create_file(my_rank, arrIdxPerPe, pioIoSystem, iodescNCells, filename,
			   iotype, NDIMS, dimlen, verbose, data_buffer, &ncid)))
	ERR(ret);
  
    /* End MPE logging for file creation. */
    if ((ret = MPE_Log_event(event_num[END][CREATE], 0, "end create")))
	MPIERR(ret);

    /* Start MPE logging for writing. */
    if ((ret = MPE_Log_event(event_num[START][WRITE], 0, "start write")))
	MPIERR(ret);

    /* Write data to the file. */
    if ((ret = write_data(my_rank, arrIdxPerPe, pioIoSystem, iodescNCells, filename,
			  iotype, NDIMS, dimlen, verbose, data_buffer, ncid)))
	ERR(ret);
  
    /* End MPE logging for writing. */
    if ((ret = MPE_Log_event(event_num[END][WRITE], 0, "end write")))
	MPIERR(ret);

    /* Start MPE logging for reading. */
    if ((ret = MPE_Log_event(event_num[START][READ], 0, "start read")))
	MPIERR(ret);

    /* Check that the output file is correct. */
    if ((ret = check_file(my_rank, arrIdxPerPe, pioIoSystem, filename, iotype,
			  iodescNCells, NDIMS, dimlen, verbose, data_buffer)))
      ERR(ret);
   
    /* End MPE logging for reading. */
    if ((ret = MPE_Log_event(event_num[END][READ], 0, "end read")))
	MPIERR(ret);

    /* Start MPE logging for cleanup. */
    if ((ret = MPE_Log_event(event_num[START][CLEANUP], 0, "start cleanup")))
	MPIERR(ret);

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

    /* End MPE logging for initialization. */
    if ((ret = MPE_Log_event(event_num[END][CLEANUP], 0, "end cleanup")))
	MPIERR(ret);

    /* Shut down MPI. */
    if (verbose)
	printf("task %d finalizing...\n", my_rank);
    MPI_Finalize();

    if (verbose)
	printf("SUCCESS!\n");
   
    return 0;
}
