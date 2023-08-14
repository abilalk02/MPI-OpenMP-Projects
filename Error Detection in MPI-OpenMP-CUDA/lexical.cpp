#include "lexical.h"

/* Function definitions */
/* ------------------------------------------------------ */

/* Function for reading the input file and storing it
   character by character */
string readSourceCode()
{
    // Variables for reading the input file
    std::ifstream inFile("inputCode.txt");
    std::string buffer;
    std::string sourceString;

    // Store the input file as a single string
    while (std::getline(inFile, buffer))
        sourceString += (buffer + '\n');

    // Return the string
    return sourceString;
}

/* For isolating sub-strings from the main source code */
char* subString(const char* str, int left, int right)
{
    char* subStr = (char*)malloc(sizeof(char) * (right - left + 2));
    for (int i = left; i <= right; i++)
        subStr[i - left] = str[i];
    subStr[right - left + 1] = '\0';
    return (subStr);
}

/* Function for detecting keywords of C++ */
bool isKeywordCpp(const char* str)
{
    if (!strcmp(str, "if") || !strcmp(str, "else") || !strcmp(str, "while")
        || !strcmp(str, "do") || !strcmp(str, "break") || !strcmp(str, "continue")
        || !strcmp(str, "int") || !strcmp(str, "double") || !strcmp(str, "float")
        || !strcmp(str, "return") || !strcmp(str, "char") || !strcmp(str, "case")
        || !strcmp(str, "sizeof") || !strcmp(str, "long") || !strcmp(str, "short")
        || !strcmp(str, "typedef") || !strcmp(str, "switch") || !strcmp(str, "unsigned")
        || !strcmp(str, "void") || !strcmp(str, "static") || !strcmp(str, "struct")
        || !strcmp(str, "goto") || !strcmp(str, "main") || !strcmp(str, "using")
        || !strcmp(str, "namespace") || !strcmp(str, "FILE") || !strcmp(str, "string")
        || !strcmp(str, "ifstream") || !strcmp(str, "ostream") || !strcmp(str, "const")
        || !strcmp(str, "#include") || !strcmp(str, "iostream") || !strcmp(str, "fstream")
        || !strcmp(str, "string") || !strcmp(str, "string.h") || !strcmp(str, "stdio.h")
        || !strcmp(str, "stdlib.h") || !strcmp(str, "math.h") || !strcmp(str, "conio.h"))
        return (true);
    else
        return (false);
}

/* Function for detecting keywords of MPI */
bool isKeywordMPI(const char* str)
{
    if (!strcmp(str, "mpi.h") || !strcmp(str, "time.h") || !strcmp(str, "MPI_Init")
        || !strcmp(str, "MPI_Comm_size") || !strcmp(str, "MPI_Comm_rank")
        || !strcmp(str, "MPI_COMM_WORLD") || !strcmp(str, "MPI_Finalize")
        || !strcmp(str, "MPI_Send") || !strcmp(str, "MPI_Recv")
        || !strcmp(str, "MPI_Scatter") || !strcmp(str, "MPI_Gather"))
        return (true);
    else
        return (false);
}

/* Function for detecting delimiters */
bool isDelimiter(const char ch)
{
    if (ch == ' ' || ch == '+' || ch == '-' || ch == '*' ||
        ch == '/' || ch == ',' || ch == ';' || ch == '=' ||
        ch == '(' || ch == ')' || ch == '[' || ch == ']' ||
        ch == '<' || ch == '>' || ch == '{' || ch == '}' ||
        ch == '\n')
        return (true);
    else
        return (false);
}

/* Function for detecting operators */
bool isOperator(const char ch)
{
    if (ch == '+' || ch == '-' || ch == '*' ||
        ch == '/' || ch == '>' || ch == '<' ||
        ch == '=')
        return (true);
    else
        return (false);
}

/* Function or detecting identifiers */
bool isIdentifier(const char* str)
{
    if (str[0] == '0' || str[0] == '1' || str[0] == '2' ||
        str[0] == '3' || str[0] == '4' || str[0] == '5' ||
        str[0] == '6' || str[0] == '7' || str[0] == '8' ||
        str[0] == '9' || isDelimiter(str[0]) == true)
        return (false);
    else
        return (true);
}

/* Function for detecting an integer value */
bool isInteger(const char* str)
{
    int len = strlen(str);
    if (len == 0)
        return (false);

    for (int i = 0; i < len; i++) {
        if ((str[i] != '0' && str[i] != '1' && str[i] != '2' && str[i] != '3'
            && str[i] != '4' && str[i] != '5' && str[i] != '6' && str[i] != '7'
            && str[i] != '8' && str[i] != '9') || (str[i] == '-' && i > 0))
            return (false);
    }
    return (true);
}

/* Function for detecting a real number */
bool isNumber(const char* str)
{
    int i, len = strlen(str);
    bool hasDecimal = false;
    if (len == 0)
        return (false);

    for (i = 0; i < len; i++) {
        if ((str[i] != '0' && str[i] != '1' && str[i] != '2' && str[i] != '3'
            && str[i] != '4' && str[i] != '5' && str[i] != '6' && str[i] != '7'
            && str[i] != '8' && str[i] != '9' && str[i] != '.') || (str[i] == '-'
                && i > 0))
            return (false);

        // Detect a decimal point
        if (str[i] == '.')
            hasDecimal = true;
    }
    return (hasDecimal);
}

/* Function for tokenizing the source code and identifying
   various components of the code */
void Tokenize(const char* str)
{
    // Variables used for keeping track of words in code
    int left = 0, right = 0;

    // Record the total number of characters in the source code
    int length = strlen(str);

    // While (there are characters to read)
    while (right <= length && left <= right)
    {
        // Check if the character is a delimiter
        if (isDelimiter(str[right]) == false)
            right++;

        if (isDelimiter(str[right]) == true && left == right)
        {
            // Check to see if the delimiter is an operator
            if (isOperator(str[right]) == true)
                printf("Valid operator : %c\n", str[right]);
            right++;
            left = right;
        }
        else if ((isDelimiter(str[right]) == true && left != right) || (right == length && left != right))
        {
            // Isolate the sub-string
            char* subStr = subString(str, left, right - 1);

            // If sub-string is a C++ keyword
            if (isKeywordCpp(subStr) == true)
                printf("Valid keyword : %s\n", subStr);

            // If sub-string is a MPI keyword
            else if (isKeywordMPI(subStr) == true)
                printf("Valid MPI keyword : %s\n", subStr);

            // If sub-string is an integer
            else if (isInteger(subStr) == true)
                printf("Valid Integer : %s\n", subStr);

            // If sub-string is a real number
            else if (isNumber(subStr) == true)
                printf("Real Number : %s\n", subStr);

            // If sub-string is an identifier
            else if (isIdentifier(subStr) == true && isDelimiter(str[right - 1]) == false)
                printf("Valid Identifier : %s\n", subStr);

            // If sub-string is an invalid identifier
            else if (isIdentifier(subStr) == false && isDelimiter(str[right - 1]) == false)
                printf("Invalid Identifier : %s\n", subStr);
            left = right;
        }
    }
    return;
}