#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#include<mpi.h>
#include<math.h>

/* ************************************************************** */
/* This code makes use of the Master Worker MPI topology and      */
/* combines MPI with OpenMP. The program must be run with a       */
/* minimum of 2 processes i.e. 1 Master and 1 at least 1 worker   */
/* The number of threads created by each processor is fixed at    */
/* 2 but the value can be modified depending upon system specs    */
/* ============================================================== */

/* -------------------------------------------------------------- */
/* Structure Definitions ........................................ */

// Structure for saving data of pictures 
typedef struct {
	int ID;
	int size;
	int* pixels;
} Picture;

// Structure for saving data of objects
typedef struct {
	int ID;
	int size;
	int* pixels;
} Object;

/* -------------------------------------------------------------- */
/* Function Definitions ......................................... */
FILE* ReadPics(int NumPics, FILE* InPtr, Picture* PICS);
FILE* ReadObjs(int NumObjs, FILE* InPtr, Object* OBJS);
void ProcessPICS(int MyRank, int CommSize, int NumPics, int LocalNumPics, int NumObjs, Picture* PICS, Picture* LocalPICS, Object* OBJS, double MatchThreshold);
void PrintResults(int MyRank, int CommSize, char** argv, int NumPics);
void PrintTime(int MyRank, int CommSize, char** argv, double ActualTime);

/* ===== MAIN ===================================================== */
// Process 0 is the Master, the other processes are Workers
int main(int argc, char* argv[])
{
	// Catch make errors
	// If there is an error, correct make file
	if (argc != 4)
	{
		printf("USE LIKE THIS: mpiexec -n <number of processes> ./obj_det input_file_name output_file_name.txt time_file_name.txt\n");
		exit(-1);
	}

	// Define variables
	double MatchThreshold;			// Matching threshold given in input file
	int NumPics;					// Number of pictures given in input file	
	int NumObjs;					// Number of objects given in input file
	Picture* PICS = NULL;			// Array for storing all pictures
	Object* OBJS = NULL;			// Array for storing all objects
	int LocalNumPics = 0;			// Number of pictures to be processed by each worker
	Picture* LocalPICS = NULL;		// Array used by workers for storing data of pictures assigned			
	int CommSize, MyRank;			// For storing processor rank and communicator size

	// Set up MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &CommSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

	// Only Master processor reads the input file
	if (MyRank == 0) {
		// Open input file for reading
		FILE* InPtr = fopen(argv[1], "r");
		if (InPtr == NULL) {
			printf("Could not open file %s\n", argv[1]);
			exit(-2);
		}
		// Scan and store data in respective variables
		fscanf(InPtr, "%lf", &MatchThreshold);
		fscanf(InPtr, "%d", &NumPics);

		// Broadcast required data to Worker Processes
		MPI_Bcast(&MatchThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&NumPics, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Allocate memory according to the number of pictures
		PICS = (Picture*)malloc(NumPics * sizeof(Picture));

		// Read all information of pictures
		InPtr = ReadPics(NumPics, InPtr, PICS);

		// Read Objects Now
		fscanf(InPtr, "%d", &NumObjs);
		MPI_Bcast(&NumObjs, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Allocate memory as per the number of objects
		OBJS = (Object*)malloc(NumObjs * sizeof(Object));

		// Read all information of objects
		InPtr = ReadObjs(NumObjs, InPtr, OBJS);

		// All processes need to know the object data
		// Therefore, that has to be sent to workers as well
		for (int i = 0; i < NumObjs; i++) {
			MPI_Bcast(&OBJS[i].ID, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&OBJS[i].size, 1, MPI_INT, 0, MPI_COMM_WORLD);
			int ObjSize = OBJS[i].size * OBJS[i].size;
			MPI_Bcast(OBJS[i].pixels, ObjSize, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	// Worker Processes
	else
	{
		// Redeive data from master
		MPI_Bcast(&MatchThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&NumPics, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Determine the local number of pictures that this worker will have to process
		int quotient, remainder;
		quotient = NumPics / (CommSize - 1);
		remainder = NumPics % (CommSize - 1);
		if (MyRank <= remainder)
			LocalNumPics = quotient + 1;
		else
			LocalNumPics = quotient;

		// Allocate memory for objects that will be received from Master
		MPI_Bcast(&NumObjs, 1, MPI_INT, 0, MPI_COMM_WORLD);
		OBJS = (Object*)malloc(NumObjs * sizeof(Object));

		// Receive information of all objects
		for (int i = 0; i < NumObjs; i++) {
			MPI_Bcast(&OBJS[i].ID, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&OBJS[i].size, 1, MPI_INT, 0, MPI_COMM_WORLD);
			int ObjSize = OBJS[i].size * OBJS[i].size;
			OBJS[i].pixels = (int*)malloc(ObjSize * sizeof(int));
			MPI_Bcast(OBJS[i].pixels, ObjSize, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	// Start measuring time
	double start = MPI_Wtime();

	// Process the pictures
	ProcessPICS(MyRank, CommSize, NumPics, LocalNumPics, NumObjs, PICS, LocalPICS, OBJS, MatchThreshold);
	
	// Stop measuring time
	double end = MPI_Wtime();

	// Write results to output file
	PrintResults(MyRank, CommSize, argv, NumPics);

	// Measure Execution Time
	double ProcessorTime = end - start;
	double ActualTime;

	// Actual time is taken as the average time taken by all processes
	// So, first we sum using MPI Reduce
	MPI_Reduce(&ProcessorTime, &ActualTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// Write elapsed time to Time output file
	PrintTime(MyRank, CommSize, argv, ActualTime);

	// Close MPI
	MPI_Finalize();
	return 0;
}
/* End Main ----------------------------------------------------- */

/* ============================================================== */
/* -------------------------------------------------------------- */
/* Function Explanations ........................................ */
FILE* ReadPics(int NumPics, FILE* InPtr, Picture* PICS)
{
	// Read all information of pictures
	for (int i = 0; i < NumPics; i++) {
		fscanf(InPtr, "%d", &PICS[i].ID);		// Read ID
		fscanf(InPtr, "%d", &PICS[i].size);		// Read Size

		// Allocate memory to this picture's pixels
		// Each picture is square so
		int PicSize = PICS[i].size * PICS[i].size;
		PICS[i].pixels = (int*)malloc(PicSize * sizeof(int));

		// Read this picture's pixels
		for (int j = 0; j < PICS[i].size; j++) {
			for (int k = 0; k < PICS[i].size; k++) {
				fscanf(InPtr, "%d", &PICS[i].pixels[(j * PICS[i].size) + k]);
			}
		}
	}
	return InPtr;
}
/* --------------------------------------------------------------- */
FILE* ReadObjs(int NumObjs, FILE* InPtr, Object* OBJS)
{
	for (int i = 0; i < NumObjs; i++) {
		fscanf(InPtr, "%d", &OBJS[i].ID);		// Read ID
		fscanf(InPtr, "%d", &OBJS[i].size);		// Read Size

		// Allocate memory to this object's pixels
		// Each object is square so
		int ObjSize = OBJS[i].size * OBJS[i].size;
		OBJS[i].pixels = (int*)malloc(ObjSize * sizeof(int));

		// Read this object's pixels
		for (int j = 0; j < OBJS[i].size; j++) {
			for (int k = 0; k < OBJS[i].size; k++) {
				fscanf(InPtr, "%d", &OBJS[i].pixels[(j * OBJS[i].size) + k]);
			}
		}
	}
	return InPtr;
}
/* ----------------------------------------------------------------------------------------------------------- */
void ProcessPICS(int MyRank, int CommSize, int NumPics, int LocalNumPics, int NumObjs, Picture* PICS, Picture* LocalPICS, Object* OBJS, double MatchThreshold)
{
	// Set the number of threads
	// The value can be adjusted as desired
	int thread_count = 2;

	int PicCount = 0;

	if (MyRank == 0) {	
		// While there are pictures left to assign
		while (PicCount < NumPics) {
			
			/* Stop signal is set to 1 when all pictures have been processed */
			int stop_signal = 0;

			/* Send the stop signal to workers */
			for (int i = 1; i < CommSize; i++)
				MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

			/* Prepare buffer for receiving signal and rank of any "ready to work" worker process */
			/* Index 0 is for storing signal value of WorkerIsFree */
			/* Index 1 is for storing rank of signal sending process */
			int signal_rank[2];
			signal_rank[0] = 0;
			signal_rank[1] = 0;
			int WorkerIsFree = signal_rank[0];

			/* Prepare buffer for receiving signal and rank of other worker processes */
			int temp_rank[2];
			temp_rank[0] = 0;
			temp_rank[1] = 0;

			/* A flag that is set to 1 for processes that are not assigned work */
			int no_work_for_you = 0;

			/* Wait until a worker process is available for work */
			while (WorkerIsFree != 1) {
				MPI_Recv(signal_rank, 2, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				WorkerIsFree = signal_rank[0];
			}
			/* Send no_work_for_you flag info to the process that is assigned work */
			MPI_Send(&no_work_for_you, 1, MPI_INT, signal_rank[1], 0, MPI_COMM_WORLD);

			/* Send a Picture to the worker that is assigned work */
			MPI_Send(&PICS[PicCount].ID, 1, MPI_INT, signal_rank[1], 0, MPI_COMM_WORLD);
			MPI_Send(&PICS[PicCount].size, 1, MPI_INT, signal_rank[1], 1, MPI_COMM_WORLD);
			int PicSize = PICS[PicCount].size * PICS[PicCount].size;
			MPI_Send(PICS[PicCount].pixels, PicSize, MPI_INT, signal_rank[1], 2, MPI_COMM_WORLD);
			PicCount++;

			/* Receive signal messages from other processes that have not been assigned work */
			for (int rank = 1; rank < CommSize; rank++)	{
				/* Set their no_work flag to 1 */
				no_work_for_you = 1;
				if (rank != signal_rank[1])	{
					/* Receive their signal info and send them the work flag value */
					MPI_Recv(temp_rank, 2, MPI_INT, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Send(&no_work_for_you, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
				}
			}
		}
		/* Stop signal is set to 1 when there is no more available data and sent
		   to all other processes */
		int stop_signal = 1;
		for (int i = 1; i < CommSize; i++)
			MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	}
	else {
		/* Initialize value of stop signal to 0 initially */
		int stop_signal = 0;
		while (stop_signal != 1)	/* while work is available */
		{
			/* Receive the value of stop signal from master */			
			MPI_Recv(&stop_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			/* If work is available */
			if (stop_signal == 0)
			{
				int signal_rank[2];
				signal_rank[0] = 1;
				signal_rank[1] = MyRank;

				/* Send signal and rank information to master */
				MPI_Send(signal_rank, 2, MPI_INT, 0, 1, MPI_COMM_WORLD);

				/* Receive work flag info from master */
				int no_work_for_you;
				MPI_Recv(&no_work_for_you, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				/* If there is work to do */
				if (no_work_for_you != 1)
				{				
					Picture Pic;
					MPI_Recv(&Pic.ID, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&Pic.size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					int PicSize = Pic.size * Pic.size;
					Pic.pixels = (int*)malloc(PicSize * sizeof(int));
					MPI_Recv(Pic.pixels, PicSize, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					// For recording the number of objects matched
					int NumObjsMatched = 0;

					// Array for storing the results (initialized to 0)
					// Results[0] strores Picture ID
					// Results [10] stores final NumObjsMatched
					// Results [1 to 9] are allocated for storing up to 3 objects matched
					// Each object's ID, x and y coordinates are saved
					int Results[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
					Results[0] = Pic.ID;

					// Search for matches from all objects
					for (int j = 0; j < NumObjs; j++) {
						// Check to see if an object has already been matched once
						// Check is set to 1 if current object is matched once
						int check = 0;

						// Scan object placement in picture in parallel
						// Each worker process forks out threads = num_threads
#						pragma omp parallel for num_threads(thread_count)
						for (int k = 0; k <= Pic.size - OBJS[j].size; k++) {
							if (check != 1) {
								for (int l = 0; l <= Pic.size - OBJS[j].size; l++) {
									if (check != 1) {
										int x_coordinate = k;		// X coordinate
										int y_coordinate = l;		// Y coordinate
										double matching = 0;		// Calculated Matching
										int index = 0;				// Object pixel index

										// Calculate matching at current x and y coordinate
										// matching = SUM(abs(p-o)/p)/ (Obj Size * Obj Size)
										for (int m = x_coordinate; m < (x_coordinate + OBJS[j].size); m++) {
											for (int n = y_coordinate; n < (y_coordinate + OBJS[j].size); n++) {
												int p = (Pic.pixels[(m * Pic.size) + n]);
												int o = OBJS[j].pixels[index];
												double diff = abs((double)(p - o) / (double)p);
												matching += diff;
												index++;
											}
										}
										matching = matching / (OBJS[j].size * OBJS[j].size);

										// Check to see if calculated value of matching is less than given threshold
										// If YES, store this object's data in results array
										if (matching < MatchThreshold) {
											Results[(NumObjsMatched * 3) + 1] = OBJS[j].ID;
											Results[(NumObjsMatched * 3) + 2] = x_coordinate;
											Results[(NumObjsMatched * 3) + 3] = y_coordinate;
											NumObjsMatched++;
#											pragma omp critical									
												check = 1;
										}
									}
								}
							}
						}
					}
					Results[10] = NumObjsMatched;
					// Send the results to Master
					MPI_Send(Results, 11, MPI_INT, 0, 4, MPI_COMM_WORLD);
				}
			}
		}
	}
}
/* ----------------------------------------------------------------- */
void PrintResults(int MyRank, int CommSize, char** argv, int NumPics)
{
	// Master will collect results and write to output file
	if (MyRank == 0) {
		FILE* OutPtr = fopen(argv[2], "w");	
		// Receive Data
		for (int i = 0; i < NumPics; i++) {
			int RecvResults[11];
			MPI_Recv(RecvResults, 11, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Write to output file
			fprintf(OutPtr, "Picture %d:  ", RecvResults[0]);

			// First determine how many objects were matched for this picture
			int NumObjsMatched = RecvResults[10];
			if (NumObjsMatched < 3)
				fprintf(OutPtr, "No three different objects were found\n\n");
			else {
				fprintf(OutPtr, "Found Objects: %d Position(%d, %d), ", RecvResults[1], RecvResults[2], RecvResults[3]);
				fprintf(OutPtr, "%d Position(%d, %d), ", RecvResults[4], RecvResults[5], RecvResults[6]);
				fprintf(OutPtr, "%d Position(%d, %d)\n\n", RecvResults[7], RecvResults[8], RecvResults[9]);
			}
		}	
		fclose(OutPtr);
	}
}
/* ------------------------------------------- */
void PrintTime(int MyRank, int CommSize, char** argv, double ActualTime)
{
	// Master Will write to time file
	if (MyRank == 0) {
		FILE* TimePtr = fopen(argv[3], "w");
		printf("Done! Output Files Created\n\n");
		fprintf(TimePtr, "%f seconds", ActualTime / CommSize);
		fclose(TimePtr);
	}
}