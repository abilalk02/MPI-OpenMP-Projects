#pragma once

#define MAX_LINES 200
#define MAX_WORDS_PER_LINE 100

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <vector>

#include "helper.h"

using namespace std;

/* Structure definitions */
/* -------------------------------------------- */
// Structure for storing grammar of MPI_Init
typedef struct {
	int initLineNumber;
	string argC;
	string argV;
} grammar_MPI_Init;

// Structure for storing grammar of MPI_Send
typedef struct {
	int sendID;
	int sendLineNumber;
	string sendBuffer;
	string sendCount;
	string sendDatatype;
	string sendDestination;
	string sendTag;
	string sendCommunicator;
	int correspondingReceive;
} grammar_MPI_Send;

// Structure for storing grammar of MPI_Recv
typedef struct {
	int recvID;
	int recvLineNumber;
	string recvBuffer;
	string recvCount;
	string recvDatatype;
	string recvSource;
	string recvTag;
	string recvCommunicator;
	string recvStatus;
	int correspondingSend;
} grammar_MPI_Recv;

// Structure for storing grammar of MPI_Bcast
typedef struct {
	int bcastID;
	int bcastLineNumber;
	string bcastBuffer;
	string bcastCount;
	string bcastDatatype;
	string bcastRoot;
	string bcastCommunicator;
} grammar_MPI_Bcast;

// Structure for storing grammar of MPI_Reduce
typedef struct {
	int reduceID;
	int reduceLineNumber;
	string reduceSendbuf;
	string reduceRecvbuf;
	string reduceCount;
	string reduceDatatype;
	string reduceOperator;
	string reduceRoot;
	string reduceCommunicator;
} grammar_MPI_Reduce;

// Structure for storing grammar of MPI_Comm_rank
typedef struct {
	int commrankLineNumber;
	string commrankCommunicator;
	string commrankRank;
} grammar_MPI_Comm_rank;

// Structure for storing grammar of MPI_Isend
typedef struct {
	int IsendID;
	int IsendLineNumber;
	string IsendBuffer;
	string IsendCount;
	string IsendDatatype;
	string IsendDestination;
	string IsendTag;
	string IsendCommunicator;
	string IsendRequest;
	int correspondingIReceive;
} grammar_MPI_Isend;

// Structure for storing grammar of MPI_Irecv
typedef struct {
	int IrecvID;
	int IrecvLineNumber;
	string IrecvBuffer;
	string IrecvCount;
	string IrecvDatatype;
	string IrecvSource;
	string IrecvTag;
	string IrecvCommunicator;
	string IrecvRequest;
	int correspondingISend;
} grammar_MPI_Irecv;

/* Function declarations */
/* ---------------------------------------------- */
string Get_Rank_Variable(string mpiCommrankRank);
returnMessage Check_MPI_Send(grammar_MPI_Send mpiSend, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank, string senderRank);
returnMessage Check_MPI_Recv(grammar_MPI_Recv mpiRecv, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int startLine, string mpiRank, string receiverRank);
returnMessage Check_MPI_Isend(grammar_MPI_Isend mpiIsend, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank, string senderRank);
returnMessage Check_MPI_Irecv(grammar_MPI_Irecv mpiIrecv, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int startLine, string mpiRank, string receiverRank);
//returnMessage Check_MPI_Bcast(grammar_MPI_Bcast mpiBcast, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank);