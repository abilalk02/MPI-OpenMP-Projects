#pragma once

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <sstream>

using namespace std;

/* Function declarations */
string readSourceCode();
char* subString(const char* str, int left, int right);
bool isKeywordCpp(const char* str);
bool isKeywordMPI(const char* str);
//bool isKeywordOpenMP(const char* str);
//bool isKeywordCUDA(const char* str);
bool isIdentifier(const char* str);
bool isDelimiter(const char str);
bool isOperator(const char str);
bool isInteger(const char* str);
bool isNumber(const char* str);
void Tokenize(const char* str);