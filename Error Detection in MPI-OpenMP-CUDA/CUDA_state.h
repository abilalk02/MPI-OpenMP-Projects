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
#include <cctype>

#include "helper.h"

using namespace std;

/* Structure definitions */
/* -------------------------------------------- */
// Structure for storing grammar of blockGridDims
typedef struct {
	int dimsLineNumber;
	string xDim;
	string yDim;
	string zDim;
} grammar_blockGridDims;

// Structure for storing grammar of cudaMalloc
typedef struct {
	int mallocLineNumber;
	string devPtr;
	string size;
} grammar_cudaMalloc;

// Structure for storing grammar of cudaMemcpy
typedef struct {
	int memCpyLineNumber;
	string destination;
	string source;
	string count;
	string kind;
} grammar_cudaMemcpy;

/* Function declarations */
/* ---------------------------------------------- */
string Identify_Kernel_Func(string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int kernelLineNumber);
string Identify_Tid_Var(string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int kernelLineNumber, int numLines);
returnMessage detectRaceCondCuda(string tidVariable, string codeLines[], int kernelLineNumber, int mainLine);
returnMessage isCorrect(string argOfKernel, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int callLineNumber);