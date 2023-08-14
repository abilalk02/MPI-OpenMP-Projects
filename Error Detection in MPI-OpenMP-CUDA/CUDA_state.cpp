#include "CUDA_state.h"

// Function for identifying the name of the kernel
string Identify_Kernel_Func(string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int kernelLineNumber)
{
	string kernelLine = codeLines[kernelLineNumber];
	string kernelFuncName = codeWords[kernelLineNumber][2];
	return kernelFuncName;
}
// Function for identifying the CUDA thread Id variable
string Identify_Tid_Var(string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int kernelLineNumber, int numLines)
{
	// Identify the code line containing the thread ID variable
	string tidVarLine = "NULL";
	for (int i = kernelLineNumber; i < numLines; i++)
	{
		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
		{
			if (codeWords[i][j] == "threadIdx.x")
			{
				tidVarLine = codeLines[i];
				break;
			}
		}
		if (tidVarLine != "NULL")
			break;
	}
	// Find the position of '='
	size_t equalPos = tidVarLine.find('=');
	if (tidVarLine[equalPos - 1] == ' ')
		equalPos -= 1;

	// Find the position of the last space before '='
	size_t lastSpacePos = tidVarLine.find_last_of(" \t", equalPos - 1);

	// Extract the word before '='
	string tidVariable = tidVarLine.substr(lastSpacePos + 1, equalPos - lastSpacePos - 1);

	// return tid variable
	return tidVariable;
}
// Function for detecting race condition in CUDA kernel
returnMessage detectRaceCondCuda(string tidVariable, string codeLines[], int kernelLineNumber, int mainLine)
{
	returnMessage message;
	bool isRace = false;
	for (int i = kernelLineNumber + 1; i < mainLine; i++)
	{
		string line = codeLines[i];
		// Remove whitespace characters from the line
		line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

		// Check if the modified string contains "[tidVariable+" or "[tidVariable-"
		string searchString1 = "[" + tidVariable + "+";
		string searchString2 = "[" + tidVariable + "-";
		size_t found1 = line.find(searchString1);
		size_t found2 = line.find(searchString2);

		if (found1 != std::string::npos || found2 != std::string::npos)
		{
			message.errorLine = i;
			message.Error = "\t(CUDA) RACE CONDITION: Using values that are possibly updated by other threads";
			message.Check = false;
			message.Potential = false;
			isRace = true;
			break;
		}
	}
	if (isRace == false)
	{
		message.errorLine = kernelLineNumber;
		message.Error = "\t(CUDA) No race condition";
		message.Check = true;
		message.Potential = false;
	}
	return message;
}
// Function for checking if the variable is an array or pointer
returnMessage isCorrect(string argOfKernel, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int callLineNumber)
{
	returnMessage message;

	// Find the declaration of this kernel argument
	bool checkDeclaration = false;
	int argLineNumber = 0;
	for (int i = 1; i < callLineNumber; i++)
	{
		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
		{
			if (codeWords[i][j] == argOfKernel)
			{
				checkDeclaration = true;
				argLineNumber = i;
				break;
			}
		}
		if (checkDeclaration == true)
			break;
	}
	if (checkDeclaration == true)
	{
		string argLine = codeLines[argLineNumber];
		// Check if it is an array or pointer
		bool hasPointer = false;

		// Find the position of the argument in the line
		size_t argPos = argLine.find(argOfKernel);

		// Check if the '*' character is before the argument
		for (size_t k = 0; k < argPos; k++) {
			if (argLine[k] == '*') {
				hasPointer = true;
				break;
			}
		}
		// If the argument is indeed a pointer
		// its memory must be allocated on device
		if (hasPointer == true)
		{
			bool found = false;
			for (int l = argLineNumber + 1; l < callLineNumber; l++)
			{
				for (int m = 0; m < MAX_WORDS_PER_LINE; m++)
				{
					if (codeWords[l][m] == argOfKernel || codeWords[l][m] == "&" + argOfKernel + ",")
					{
						// find cudaMalloc
						bool foundMalloc = false;
						for (int n = 0; n < MAX_WORDS_PER_LINE; n++)
						{
							if (codeWords[l][n] == "cudaMalloc")
							{
								foundMalloc = true;
								break;
							}
						}
						// If the line contains a CUDA malloc
						if (foundMalloc == true)
						{
							found = true;
							message.errorLine = callLineNumber;
							message.Error = "\t(CUDA) Correct memory allocation on device";
							message.Check = true;	
							message.Potential = false;
						}					
						break;
					}
				}
				if (found == true)
					break;
			}
			if (found == false)
			{
				message.errorLine = callLineNumber;
				message.Error = "\t(CUDA) Failure to allocate required memory on device for Kernel argument";
				message.Check = false;
				message.Potential = false;
			}
		}
		else
		{
			message.errorLine = callLineNumber;
			message.Error = "\t(CUDA) No need to allocate memory on device for this variable";
			message.Check = true;		
			message.Potential = false;
		}
		return message;
	}
	else
	{
		message.errorLine = callLineNumber;
		message.Error = "\t(CUDA) Kernel Argument declaration not found";
		message.Check = false;
		message.Potential = false;
		return message;
	}
}