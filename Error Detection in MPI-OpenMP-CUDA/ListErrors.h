#pragma once

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>

#include "MPI_state.h"
#include "CUDA_state.h"


// Function Declarations
// ------------------------------------------------
void CreateOutputFiles(vector <returnMessage> messagesData, vector <grammar_MPI_Send> mpiSendData,
	vector <grammar_MPI_Recv> mpiRecvData, vector <grammar_MPI_Isend> mpiIsendData,
	vector <grammar_MPI_Irecv> mpiIrecvData, vector <grammar_MPI_Bcast> mpiBcastData,
	vector <grammar_MPI_Reduce> mpiReduceData, vector <grammar_cudaMalloc> cudaMallocData,
	vector <grammar_cudaMemcpy> cudaMemcpyData);
