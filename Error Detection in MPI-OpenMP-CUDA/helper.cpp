#include "helper.h"

/* Helper functions definitions */
/* ---------------------------------------------------- */
// Function for separating words within brackets at commas
vector <string> separateAtCommas(string& codeLine)
{
	// Vector for storing the comma separated words
	vector <string> separatedWords;

	// Find the position of the opening and closing brackets
	size_t leftBracket = codeLine.find('(');
	size_t rightBracket = codeLine.rfind(')');

    // Extract the substring between the brackets
    string substring = codeLine.substr(leftBracket + 1, rightBracket - leftBracket - 1);

    // Use stringstream to tokenize the substring based on commas
    stringstream ss(substring);
    string word;

    // Separate the substring at commas
    while (getline(ss, word, ',')) 
    {
        // Remove leading and trailing whitespace from each word
        size_t start = word.find_first_not_of(" \t");
        size_t end = word.find_last_not_of(" \t");
        if (start != string::npos && end != string::npos) 
        {
            // Add the word to the separated words vector
            separatedWords.push_back(word.substr(start, end - start + 1));
        }
    }
    // Return the separated words vector
    return separatedWords;
}

// Function for printing out the error messages
void printErrorMessages(vector <returnMessage> messagesData)
{
    int vecLength = size(messagesData);
    for (int i = 0; i < vecLength; i++)
    {
        if (messagesData[i].Check == false)
        {
            cout << "ERROR (Line " << messagesData[i].errorLine << ")" << endl;
            cout << messagesData[i].Error << endl << endl;
        }
    }
}

// Function for finding the required MPI rank
string findMPIRank(string rankLine)
{
    // Check if the line contains "=="
    size_t found = rankLine.find("==");
    string matchRankLine;

    // If line contains "=="
    if (found != std::string::npos)
    {
        // Extract the substring between the brackets
        size_t leftBracket = rankLine.find('(');
        size_t rightBracket = rankLine.find(')');
        string substring = rankLine.substr(leftBracket + 1, rightBracket - leftBracket - 1);

        // Create a new string by removing any spaces in the substring
        for (char c : substring)
        {
            if (!std::isspace(c)) {
                matchRankLine += c;
            }
        }
    }
    else
        matchRankLine = "NULL";

    // Return the matchRank string
    return matchRankLine;
}

// Function for counting spaces of tabs in a code line
SpaceOrTab countSpacesTabs(string codeLine)
{
    SpaceOrTab numSpacesTabs;
    int spaces = 0, tabs = 0;
    
    // Check if leading characters are spaces
    for (char ch : codeLine) 
    {
        if (ch == ' ') {
            spaces++;
        }
        else {
            break;
        }
    }
    // Check if leading characters are tabs
    for (char ch : codeLine)
    {
        if (ch == '\t') {
            tabs++;
        }
        else {
            break;
        }
    }
    // If leading characters are spaces
    if (spaces != 0)
    {
        numSpacesTabs.count = spaces;
        numSpacesTabs.type = "spaces";
    }
    else
    {
        numSpacesTabs.count = tabs;
        numSpacesTabs.type = "tabs";
    }
    return numSpacesTabs;
}

// Function for finding the size of a for loop
int findLoopSize(string codeLine)
{
    // Regular expression pattern to match the loop condition
    regex loopPattern(R"(for\s*\(.*\s*=\s*\d+;\s*\w+\s*<\s*(\d+);\s*.*\))");

    // For matching results
    smatch match;
    int loopSize = 1;

    if (regex_search(codeLine, match, loopPattern) && match.size() == 2) {
        // Extract the loop size value from the match
        loopSize = stoi(match[1]);
    }
    return loopSize;
}
