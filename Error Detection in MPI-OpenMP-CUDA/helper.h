#pragma once

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <regex>

using namespace std;

/* Helper structures */
/* ------------------------------------------------------ */
typedef struct
{
	int errorLine;
	string Error;
	bool Check;
	bool Potential;
} returnMessage;

typedef struct
{
	int count;
	string type;
} SpaceOrTab;

/* Function declarations */
/* ------------------------------------------------------ */
vector <string> separateAtCommas(string &codeLine);
void printErrorMessages(vector <returnMessage> messagesData);
string findMPIRank(string rankLine);
SpaceOrTab countSpacesTabs(string codeLine);
int findLoopSize(string codeLine);
