#include "OMP_state.h"

// Function for checking whether omp for loop is being called correctly
bool checkOmpFor(string codeLines[], int ompForLineNumber)
{
	string word = "for";
	string nextLine = codeLines[ompForLineNumber + 1];

	// Remove any preceding spaces
	size_t startPos = nextLine.find_first_not_of(" \t");
	
	bool startsWithFor = (nextLine.substr(startPos, word.length()) == word);
	return startsWithFor;
}
// Function for checking whether omp parallel is being called correctly
bool checkOmpParallel(string codeLines[], int ompParallelLineNumber)
{
	string word = "{";
	string nextLine = codeLines[ompParallelLineNumber + 1];

	// Remove any preceding spaces
	size_t startPos = nextLine.find_first_not_of(" \t");

	bool startsWithBracket = (nextLine.substr(startPos, word.length()) == word);
	return startsWithBracket;
}