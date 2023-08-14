
#include "lexical.h"
#include "MPI_state.h"
#include "CUDA_state.h"
#include "OMP_state.h"
#include "ListErrors.h"
#include "dynamic.h"

/* MAIN */
/* ----------------------------------------------------- */
int main(void)
{
    // Array for storing lines of code
    string codeLines[MAX_LINES];

    // 2D array for storing words of code
    string codeWords[MAX_LINES][MAX_WORDS_PER_LINE];

    // Variables for recording data of lines
    int numLines = 0, mainLine = 0, startLine = 0, kernelArgs = 0;

    // Variables for storing the IDs of mp messages;
    int sendID = 1, recvID = 1, bcastID = 1, reduceID = 1, IsendID = 1, IrecvID = 1;

    // Variables for ensuring correct headers are included and initialization is done
    bool mpiHeaderCheck = false, mpiInitCheck = false;
    bool cudaHeaderCheck = false;
    bool ompHeaderCheck = false;

    // Variables for storing variables of MPI, CUDA and OpenMP
    string mpiRank = "NULL";
    string kernelFuncName = "NULL";
    string ompLockVariable = "NULL";
    string cudaTidVariable = "NULL";

    // Open the input source code file
    std::ifstream inFile("inputCode.txt");
    std::string buffer;
    std::string sourceString;

    // Vector for storing the return messages
    vector <returnMessage> messagesData;

    // Vectors for storing data of various commands
    vector <grammar_MPI_Send> mpiSendData;
    vector <grammar_MPI_Recv> mpiRecvData;
    vector <grammar_MPI_Isend> mpiIsendData;
    vector <grammar_MPI_Irecv> mpiIrecvData;
    vector <grammar_MPI_Bcast> mpiBcastData;
    vector <grammar_MPI_Reduce> mpiReduceData;
    vector <grammar_blockGridDims> cudaDimsData;
    vector <grammar_cudaMalloc> cudaMallocData;
    vector <grammar_cudaMemcpy> cudaMemcpyData;
    vector <grammar_omp_init_lock> ompLockData;


    // If file opened successfully
    if (inFile.is_open())
    {
        // Read the file line by line
        while (std::getline(inFile, buffer))
        {
            string word;                        // For isolating each word
            sourceString += (buffer + '\n');    // For storing the entire input code as a single string
            
            // Use string stream to read lines word by word separating them at specified delimiters
            istringstream iss(buffer);
            int wordIndex = 0;                 
            while (iss >> word)
            {
                size_t pos = 0;
                string delimiters = " +-*/;()[]{}=<>:";     // Specified delimiters

                // Find the first delimiter to separate the word
                while ((pos = word.find_first_of(delimiters)) != string::npos) {
                    codeWords[numLines][wordIndex] = word.substr(0, pos);
                    word = word.substr(pos + 1);
                    wordIndex++;
                }

                // Find the starting line of the code
                if (word == "#include")
                    startLine = numLines;

                // Find the main line of the code
                if (word == "main")
                    mainLine = numLines;

                // Add the word to the 2D words array
                codeWords[numLines][wordIndex] = word;
                wordIndex++;
            }
            // Add the line to 1D line array
            codeLines[numLines] = buffer;
            numLines++;
        }
        // Close the input file
        inFile.close();
    }
    // Print an error if unable to open file
    else
        cout << "Error: Unable to opern input file" << endl;

    cout << endl << "Numer of lines = " << numLines << endl;
    cout << "Main Line = " << mainLine << endl;
    cout << "Start Line = " << startLine << endl;

    // Search the code for various keywords
    for (int i = 0; i < numLines; i++) {  
        for (int j = 0; j < MAX_WORDS_PER_LINE; j++) 
        {
            // Check header files
            if (codeWords[i][j] == "mpi.h")
            {
                cout << endl << "(MPI) mpi.h header file found at line " << i << endl << endl;
                mpiHeaderCheck = true;
            }
            else if (codeWords[i][j] == "omp.h")
            {
                cout << "(OpenMP) omp.h header file found at line " << i << endl << endl;
                ompHeaderCheck = true;
            }
            else if (codeWords[i][j] == "cuda.h")
            {
                cout << "(CUDA) cuda.h header file found at line " << i << endl << endl;
                cudaHeaderCheck = true;
            }
            // If MPI_Init is found
            else if (codeWords[i][j] == "MPI_Init")
            {
                if (mpiHeaderCheck == true)
                {
                    int initLineNumber = i;
                    cout << "MPI_Init found at line " << initLineNumber << endl << endl;

                    // Extract the code line containing MPI_Init
                    string codeLine = codeLines[initLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of MPI_Comm_rank
                    grammar_MPI_Init mpiInit;

                    // MPI_Init must have 2 arguments
                    if (vecLength != 2)
                    {
                        returnMessage message;
                        message.errorLine = initLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Init";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Init data in structure
                        mpiInit.initLineNumber = initLineNumber;
                        mpiInit.argC = separatedWords[0];
                        mpiInit.argV = separatedWords[1];
                        mpiInitCheck = true;
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }
            }
            // If MPI_Comm_rank is found
            else if (codeWords[i][j] == "MPI_Comm_rank")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    int commRankLineNumber = i;
                    cout << "MPI_Comm_rank found at line " << commRankLineNumber << endl << endl;

                    // Extract the code line containing MPI_Comm_rank
                    string codeLine = codeLines[commRankLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of MPI_Comm_rank
                    grammar_MPI_Comm_rank mpiCommRank;

                    // MPI_Comm_rank must have 2 arguments
                    if (vecLength != 2)
                    {
                        returnMessage message;
                        message.errorLine = commRankLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Comm_rank";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Comm_rank data in structure
                        mpiCommRank.commrankLineNumber = commRankLineNumber;
                        mpiCommRank.commrankCommunicator = separatedWords[0];
                        mpiCommRank.commrankRank = separatedWords[1];

                        // Get the mpi rank variable
                        mpiRank = Get_Rank_Variable(mpiCommRank.commrankRank);
                        cout << "MPI Rank variable is: " << mpiRank << endl << endl;
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // If an MPI_Send if found
            else if (codeWords[i][j] == "MPI_Send")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    string rankLine, senderRank = "NULL";
                    int sendLineNumber = i;
                    int forLoopSize = 1;
                    cout << "MPI_Send found at line " << sendLineNumber << endl << endl;

                    // Identify the rank of sender
                    for (int k = sendLineNumber - 1; k > 0; k--)
                    {
                        for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                        {
                            // Check if this MPI_Send is inside a for loop
                            if (codeWords[k][l] == "for")
                            {
                                string forLine = codeLines[k];
                                // Find the size of the for loop to determine the number of sends
                                forLoopSize = findLoopSize(forLine);
                            }
                            // If rank variable is found
                            if (codeWords[k][l] == mpiRank)
                            {
                                // Identify the rank line 
                                rankLine = codeLines[k];
                                string matchRankLine;

                                // Find the rank of the Sender 
                                matchRankLine = findMPIRank(rankLine);

                                if (matchRankLine != "NULL")
                                {
                                    // Get the rank
                                    senderRank = matchRankLine.substr(matchRankLine.length() - 1);
                                    break;
                                }
                            }
                        }
                        if (senderRank != "NULL")
                            break;
                    }                   
                    // Extract the code line containing MPI_Send
                    string codeLine = codeLines[sendLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // If sends are in a for loop, we have to check each send in the loop independently
                    if (forLoopSize != 1)
                    {
                        for (int k = 1; k < forLoopSize; k++)
                        {
                            // Check the grammar of this MPI_Send
                            grammar_MPI_Send mpiSend;

                            // Determine Destination rank as string
                            string destination = to_string(k);

                            // MPI_Send must have 6 arguments
                            if (vecLength != 6)
                            {
                                returnMessage message;
                                message.errorLine = sendLineNumber;
                                message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Send";
                                message.Check = false;
                                message.Potential = false;
                                messagesData.push_back(message);
                            }
                            else
                            {
                                // Store MPI_Send data in structure
                                mpiSend.sendID = sendID++;
                                mpiSend.sendLineNumber = sendLineNumber;
                                mpiSend.sendBuffer = separatedWords[0];
                                mpiSend.sendCount = separatedWords[1];
                                mpiSend.sendDatatype = separatedWords[2];
                                mpiSend.sendDestination = destination;
                                mpiSend.sendTag = separatedWords[4];
                                mpiSend.sendCommunicator = separatedWords[5];
                                mpiSend.correspondingReceive = 0;

                                // Test whether there is a corresponsding receive for this MPI_Send
                                returnMessage message = Check_MPI_Send(mpiSend, codeWords, codeLines, numLines, mpiRank, senderRank);

                                // If a correct matching receive is found
                                if (message.Check == true)
                                    mpiSend.correspondingReceive = message.errorLine;

                                // Save this MPI_Send's data in mpiSendData vector
                                mpiSendData.push_back(mpiSend);
                                messagesData.push_back(message);
                            }
                        }
                    }
                    // If send is not in a for loop
                    else {
                        // Check the grammar of this MPI_Send
                        grammar_MPI_Send mpiSend;

                        // MPI_Send must have 6 arguments
                        if (vecLength != 6)
                        {
                            returnMessage message;
                            message.errorLine = sendLineNumber;
                            message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Send";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            // Store MPI_Send data in structure
                            mpiSend.sendID = sendID++;
                            mpiSend.sendLineNumber = sendLineNumber;
                            mpiSend.sendBuffer = separatedWords[0];
                            mpiSend.sendCount = separatedWords[1];
                            mpiSend.sendDatatype = separatedWords[2];
                            mpiSend.sendDestination = separatedWords[3];
                            mpiSend.sendTag = separatedWords[4];
                            mpiSend.sendCommunicator = separatedWords[5];
                            mpiSend.correspondingReceive = 0;

                            // Test whether there is a corresponsding receive for this MPI_Send
                            returnMessage message = Check_MPI_Send(mpiSend, codeWords, codeLines, numLines, mpiRank, senderRank);

                            // If a correct matching receive is found
                            if (message.Check == true)
                                mpiSend.correspondingReceive = message.errorLine;

                            // Save this MPI_Send's data in mpiSendData vector
                            mpiSendData.push_back(mpiSend);
                            messagesData.push_back(message);
                        }
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // If an MPI_Recv is found
            else if (codeWords[i][j] == "MPI_Recv")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    string rankLine, receiverRank = "NULL";
                    int recvLineNumber = i;
                    int forLoopSize = 1;
                    cout << "MPI_Recv found at line " << recvLineNumber << endl << endl;

                    // Identify the rank of receiver
                    for (int k = recvLineNumber - 1; k > 0; k--)
                    {
                        for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                        {
                            // Check if this MPI_Send is inside a for loop
                            if (codeWords[k][l] == "for")
                            {
                                string forLine = codeLines[k];
                                // Find the size of the for loop to determine the number of sends
                                forLoopSize = findLoopSize(forLine);
                            }
                            // If rank variable is found
                            if (codeWords[k][l] == mpiRank)
                            {
                                // Identify the rank line 
                                rankLine = codeLines[k];
                                string matchRankLine;

                                // Find the rank of the Sender 
                                matchRankLine = findMPIRank(rankLine);

                                if (matchRankLine != "NULL")
                                {
                                    // Get the rank
                                    receiverRank = matchRankLine.substr(matchRankLine.length() - 1);
                                    break;
                                }
                            }
                        }
                        if (receiverRank != "NULL")
                            break;
                    }

                    // Extract the code line containing MPI_Recv
                    string codeLine = codeLines[recvLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // If receive is inside a for loop
                    if (forLoopSize != 1)
                    {
                        for (int k = 1; k < forLoopSize; k++)
                        {
                            // Determine source rank as string
                            string source = to_string(k);

                            // Check the grammar of this MPI_Recv
                            grammar_MPI_Recv mpiRecv;

                            // MPI_Recv must have 7 arguments
                            if (vecLength != 7)
                            {
                                returnMessage message;
                                message.errorLine = recvLineNumber;
                                message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Recv";
                                message.Check = false;
                                message.Potential = false;
                                messagesData.push_back(message);
                            }
                            else
                            {
                                // Store MPI_Recv data in structure
                                mpiRecv.recvID = recvID++;
                                mpiRecv.recvLineNumber = recvLineNumber;
                                mpiRecv.recvBuffer = separatedWords[0];
                                mpiRecv.recvCount = separatedWords[1];
                                mpiRecv.recvDatatype = separatedWords[2];
                                mpiRecv.recvSource = source;
                                mpiRecv.recvTag = separatedWords[4];
                                mpiRecv.recvCommunicator = separatedWords[5];
                                mpiRecv.recvStatus = separatedWords[6];
                                mpiRecv.correspondingSend = 0;

                                // Test whether there is a corresponsding send for this MPI_Recv
                                returnMessage message = Check_MPI_Recv(mpiRecv, codeWords, codeLines, startLine, mpiRank, receiverRank);

                                // If there is a matching send
                                if (message.Check == true)
                                    mpiRecv.correspondingSend = message.errorLine;

                                // Save this MPI_Recv's data in mpiRecvData vector
                                mpiRecvData.push_back(mpiRecv);
                                messagesData.push_back(message);
                            }
                        }
                    }
                    // If receive is not inside a for loop
                    else 
                    {
                        // Check the grammar of this MPI_Recv
                        grammar_MPI_Recv mpiRecv;

                        // MPI_Recv must have 7 arguments
                        if (vecLength != 7)
                        {
                            returnMessage message;
                            message.errorLine = recvLineNumber;
                            message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Recv";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            // Store MPI_Recv data in structure
                            mpiRecv.recvID = recvID++;
                            mpiRecv.recvLineNumber = recvLineNumber;
                            mpiRecv.recvBuffer = separatedWords[0];
                            mpiRecv.recvCount = separatedWords[1];
                            mpiRecv.recvDatatype = separatedWords[2];
                            mpiRecv.recvSource = separatedWords[3];
                            mpiRecv.recvTag = separatedWords[4];
                            mpiRecv.recvCommunicator = separatedWords[5];
                            mpiRecv.recvStatus = separatedWords[6];
                            mpiRecv.correspondingSend = 0;

                            // Test whether there is a corresponsding send for this MPI_Recv
                            returnMessage message = Check_MPI_Recv(mpiRecv, codeWords, codeLines, startLine, mpiRank, receiverRank);

                            // If there is a matching send
                            if (message.Check == true)
                                mpiRecv.correspondingSend = message.errorLine;

                            // Save this MPI_Recv's data in mpiRecvData vector
                            mpiRecvData.push_back(mpiRecv);
                            messagesData.push_back(message);
                        }
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // If an MPI_Isend if found
            else if (codeWords[i][j] == "MPI_Isend")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    string rankLine, senderRank = "NULL";
                    int IsendLineNumber = i;
                    cout << "MPI_Isend found at line " << IsendLineNumber << endl << endl;

                    // Identify the rank of sender
                    for (int k = IsendLineNumber - 1; k > 0; k--)
                    {
                        for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                        {
                            // If rank variable is found
                            if (codeWords[k][l] == mpiRank)
                            {
                                // Identify the rank line 
                                rankLine = codeLines[k];
                                string matchRankLine;

                                // Find the rank of the Sender 
                                matchRankLine = findMPIRank(rankLine);

                                if (matchRankLine != "NULL")
                                {
                                    // Get the rank
                                    senderRank = matchRankLine.substr(matchRankLine.length() - 1);
                                    break;
                                }
                            }
                        }
                        if (senderRank != "NULL")
                            break;
                    }
                    // Extract the code line containing MPI_Isend
                    string codeLine = codeLines[IsendLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this MPI_Send
                    grammar_MPI_Isend mpiIsend;

                    // MPI_Isend must have 6 arguments
                    if (vecLength != 7)
                    {
                        returnMessage message;
                        message.errorLine = IsendLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Isend";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Isend data in structure
                        mpiIsend.IsendID = IsendID++;
                        mpiIsend.IsendLineNumber = IsendLineNumber;
                        mpiIsend.IsendBuffer = separatedWords[0];
                        mpiIsend.IsendCount = separatedWords[1];
                        mpiIsend.IsendDatatype = separatedWords[2];
                        mpiIsend.IsendDestination = separatedWords[3];
                        mpiIsend.IsendTag = separatedWords[4];
                        mpiIsend.IsendCommunicator = separatedWords[5];
                        mpiIsend.IsendRequest = separatedWords[6];
                        mpiIsend.correspondingIReceive = 0;

                        // Test whether there is a corresponsding receive for this MPI_Isend
                        returnMessage message = Check_MPI_Isend(mpiIsend, codeWords, codeLines, numLines, mpiRank, senderRank);
                        messagesData.push_back(message);

                        // If a correct matching receive is found
                        if (message.Check == true)
                        {
                            mpiIsend.correspondingIReceive = message.errorLine;

                            // Test for a potential race condition
                            // If there is not MPI_Wait after MPI_Isend, it could lead to race condition
                            bool raceCond = true;
                            for (int k = mpiIsend.IsendLineNumber; k < mpiIsend.correspondingIReceive; k++)
                            {
                                for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                                {
                                    if (codeWords[k][l] == "MPI_Wait")
                                    {
                                        raceCond = false;
                                        break;
                                    }
                                }
                                if (raceCond == false)
                                    break;
                            }
                            if (raceCond == true)
                            {
                                returnMessage message1;
                                message1.errorLine = mpiIsend.IsendLineNumber;
                                message1.Error = "\t(MPI) POTENTIAL RACE CONDITION: Not waiting for the Isend to be received ";
                                message1.Check = false;
                                message1.Potential = false;
                                messagesData.push_back(message1);
                            }
                            else
                            {
                                returnMessage message1;
                                message1.errorLine = mpiIsend.IsendLineNumber;
                                message1.Error = "\t(MPI) No race condition ";
                                message1.Check = true;
                                message1.Potential = false;
                                messagesData.push_back(message1);
                            }
                        }
                        // Save this MPI_Isend's data in mpiIsendData vector
                        mpiIsendData.push_back(mpiIsend);   
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // If an MPI_Irecv is found
            else if (codeWords[i][j] == "MPI_Irecv")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    string rankLine, receiverRank = "NULL";
                    int IrecvLineNumber = i;
                    cout << "MPI_Irecv found at line " << IrecvLineNumber << endl << endl;

                    // Identify the rank of receiver
                    for (int k = IrecvLineNumber - 1; k > 0; k--)
                    {
                        for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                        {
                            // If rank variable is found
                            if (codeWords[k][l] == mpiRank)
                            {
                                // Identify the rank line 
                                rankLine = codeLines[k];
                                string matchRankLine;

                                // Find the rank of the Sender 
                                matchRankLine = findMPIRank(rankLine);

                                if (matchRankLine != "NULL")
                                {
                                    // Get the rank
                                    receiverRank = matchRankLine.substr(matchRankLine.length() - 1);
                                    break;
                                }
                            }
                        }
                        if (receiverRank != "NULL")
                            break;
                    }

                    // Extract the code line containing MPI_Irecv
                    string codeLine = codeLines[IrecvLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this MPI_Irecv
                    grammar_MPI_Irecv mpiIrecv;

                    // MPI_Recv must have 7 arguments
                    if (vecLength != 7)
                    {
                        returnMessage message;
                        message.errorLine = IrecvLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Irecv";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Irecv data in structure
                        mpiIrecv.IrecvID = IrecvID++;
                        mpiIrecv.IrecvLineNumber = IrecvLineNumber;
                        mpiIrecv.IrecvBuffer = separatedWords[0];
                        mpiIrecv.IrecvCount = separatedWords[1];
                        mpiIrecv.IrecvDatatype = separatedWords[2];
                        mpiIrecv.IrecvSource = separatedWords[3];
                        mpiIrecv.IrecvTag = separatedWords[4];
                        mpiIrecv.IrecvCommunicator = separatedWords[5];
                        mpiIrecv.IrecvRequest = separatedWords[6];
                        mpiIrecv.correspondingISend = 0;

                        // Test whether there is a corresponsding Isend for this MPI_Irecv
                        returnMessage message = Check_MPI_Irecv(mpiIrecv, codeWords, codeLines, startLine, mpiRank, receiverRank);
                        
                        // If there is a matching send
                        if (message.Check == true)
                            mpiIrecv.correspondingISend = message.errorLine;

                        // Save this MPI_Irecv's data in mpiIrecvData vector
                        mpiIrecvData.push_back(mpiIrecv);
                        messagesData.push_back(message);
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // if MPI_Bcast is found
            else if (codeWords[i][j] == "MPI_Bcast")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    int bcastLineNumber = i;
                    cout << "MPI_Bcast found at line " << bcastLineNumber << endl << endl;

                    // Extract the code line containing MPI_Bcast
                    string codeLine = codeLines[bcastLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this MPI_Bcast
                    grammar_MPI_Bcast mpiBcast;

                    // MPI_Bcast must have 5 arguments
                    if (vecLength != 5)
                    {
                        returnMessage message;
                        message.errorLine = bcastLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Bcast";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Bcast data in structure
                        mpiBcast.bcastID = bcastID++;
                        mpiBcast.bcastLineNumber = bcastLineNumber;
                        mpiBcast.bcastBuffer = separatedWords[0];
                        mpiBcast.bcastCount = separatedWords[1];
                        mpiBcast.bcastDatatype = separatedWords[2];
                        mpiBcast.bcastRoot = separatedWords[3];
                        mpiBcast.bcastCommunicator = separatedWords[4];

                        // Save this MPI_Bcast's data in mpiBcastData vector
                        mpiBcastData.push_back(mpiBcast);
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // if MPI_Bcast is found
            else if (codeWords[i][j] == "MPI_Reduce")
            {
                if (mpiInitCheck == true && mpiHeaderCheck == true)
                {
                    int reduceLineNumber = i;
                    cout << "MPI_Reduce found at line " << reduceLineNumber << endl << endl;

                    // Extract the code line containing MPI_Reduce
                    string codeLine = codeLines[reduceLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this MPI_Reduce
                    grammar_MPI_Reduce mpiReduce;

                    // MPI_Bcast must have 7 arguments
                    if (vecLength != 7)
                    {
                        returnMessage message;
                        message.errorLine = reduceLineNumber;
                        message.Error = "\t(MPI) SYNTAX: Invalid number of arguments of MPI_Reduce";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store MPI_Reduce data in structure
                        mpiReduce.reduceID = reduceID++;
                        mpiReduce.reduceLineNumber = reduceLineNumber;
                        mpiReduce.reduceSendbuf = separatedWords[0];
                        mpiReduce.reduceRecvbuf = separatedWords[1];
                        mpiReduce.reduceCount = separatedWords[2];
                        mpiReduce.reduceDatatype = separatedWords[3];
                        mpiReduce.reduceOperator = separatedWords[4];
                        mpiReduce.reduceRoot = separatedWords[5];
                        mpiReduce.reduceCommunicator = separatedWords[6];

                        // Save this MPI_Reduce's data in mpiReduceData vector
                        mpiReduceData.push_back(mpiReduce);
                    }
                }
                else
                {
                    if (mpiHeaderCheck == false)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) <mpi.h> HEADER file NOT INCLUDED";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(MPI) Using MPI Commands WITHOUT PROPER MPI INITIALIZATION";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
            }
            // If CUDA block grid dims are found
            else if (codeWords[i][j] == "dim3")
            {
                if (cudaHeaderCheck == true)
                {
                    cout << "CUDA block/grid dims initializations found at line " << i << endl << endl;

                    // Extract the code line containing dim3
                    string codeLine = codeLines[i];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this dim3
                    grammar_blockGridDims DIMS;

                    // DIMS must have at least arguments
                    if (vecLength < 2)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(CUDA) SYNTAX: Improper block/grid initialization";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store DIMS data in structure
                        DIMS.dimsLineNumber = i;
                        DIMS.xDim = separatedWords[0];
                        DIMS.yDim = separatedWords[1];
                        if (vecLength > 2)
                            DIMS.zDim = separatedWords[2];
                        else
                            DIMS.zDim = "NULL";

                        // Save this DIMS's data in cudaDIMS vector
                        cudaDimsData.push_back(DIMS);
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(CUDA) <cuda.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
            // If CUDA kernel function keywords are found
            else if (codeWords[i][j] == "__global__" || codeWords[i][j] == "__device__")
            {
                if (cudaHeaderCheck == true)
                {
                    // Identify the name of the kernel function
                    int kernelLineNumber = i;
                    cout << "CUDA kernel function declaration found at line " << kernelLineNumber << endl << endl;
                    kernelFuncName = Identify_Kernel_Func(codeWords, codeLines, kernelLineNumber);

                    // Note the number of arguments of kernel function call
                    string kernelLine = codeLines[kernelLineNumber];
                    vector <string> kernelSubstr = separateAtCommas(kernelLine);
                    kernelArgs = (int)size(kernelSubstr);

                    // Find the CUDA thread ID variable
                    cudaTidVariable = Identify_Tid_Var(codeWords, codeLines, kernelLineNumber, numLines);

                    // Check for possible race condition
                    returnMessage message1 = detectRaceCondCuda(cudaTidVariable, codeLines, kernelLineNumber, mainLine);
                    messagesData.push_back(message1);

                    // Check for possible livelock condition
                    bool isDeadlockCUDA = false;
                    for (int l = kernelLineNumber + 1; l < kernelLineNumber + 20; l++)
                    {
                        // Search for atomicCAS statement
                        string atomicLine = codeLines[l];
                        // Remove whitespace characters from the line
                        atomicLine.erase(remove_if(atomicLine.begin(), atomicLine.end(), ::isspace), atomicLine.end());

                        // Check if the modified string contains "atomicCAS"
                        string searchString1 = "atomicCAS";
                        size_t found1 = atomicLine.find(searchString1);

                        // If found, we have a potential deadlock situation     
                        if (found1 != std::string::npos)
                        {
                            returnMessage message1;
                            message1.errorLine = l;
                            message1.Error = "\t(CUDA) DEADLOCK: Only 1 thread allowed access to critical section\n\t\t\t\twhile the others are stuck in an infinite loop";
                            message1.Check = false;
                            message1.Potential = false;
                            messagesData.push_back(message1);
                            isDeadlockCUDA = true;
                            break;
                        }
                    }
                    if (isDeadlockCUDA == false)
                    {
                        returnMessage message1;
                        message1.errorLine = kernelLineNumber;
                        message1.Error = "\t(OpenMP) No deadlock";
                        message1.Check = true;
                        message1.Potential = false;
                        messagesData.push_back(message1);
                    }

                    // Search where the kernel function is being called
                    for (int k = kernelLineNumber + 1; k < numLines; k++)
                    {
                        for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                        {
                            if (codeWords[k][l] == kernelFuncName)
                            {
                                int callLineNumber = k;
                                cout << "Kernel Call found at line " << callLineNumber << endl << endl;

                                // Extract the code line containing Kernel call
                                string codeLine = codeLines[callLineNumber];

                                // Check for a potential livelock scenario
                                bool cudaLivelockCheck = false;
                                int cudaLivelockLine = callLineNumber;
                                for (int line = callLineNumber; line > callLineNumber - 4; line--)
                                {
                                    // Search for a line that contains while (true) immediately before kernel call                                
                                    string whileTrueCuda = codeLines[line];

                                    // Remove whitespace characters from the line
                                    whileTrueCuda.erase(std::remove_if(whileTrueCuda.begin(), whileTrueCuda.end(), ::isspace), whileTrueCuda.end());

                                    // Check if the modified string contains "while (true)"
                                    string searchStringCuda = "while(true)";
                                    size_t foundCudaTrue = whileTrueCuda.find(searchStringCuda);

                                    // If found, we have a livelock situation
                                    if (foundCudaTrue != std::string::npos)
                                    {
                                        cudaLivelockCheck = true;
                                        cudaLivelockLine = line;
                                        break;
                                    }
                                }
                                if (cudaLivelockCheck == true)
                                {
                                    returnMessage message;
                                    message.errorLine = cudaLivelockLine;
                                    message.Error = "\t(CUDA) LIVELOCK: CUDA kernel is being launched inifinitely";
                                    message.Check = false;
                                    message.Potential = false;
                                    messagesData.push_back(message);
                                }
                                else
                                {
                                    returnMessage message;
                                    message.errorLine = cudaLivelockLine;
                                    message.Error = "\t(CUDA) No livelock";
                                    message.Check = true;
                                    message.Potential = false;
                                    messagesData.push_back(message);
                                }
                                // Separate the words within brackets at commas
                                vector <string> separatedWords = separateAtCommas(codeLine);

                                // Determine the length
                                int vecLength = size(separatedWords);

                                // If the number of arguments do not match
                                if (vecLength != kernelArgs)
                                {
                                    returnMessage message;
                                    message.errorLine = k;
                                    message.Error = "\t(CUDA) SYNTAX: Invalid number of arguments in Kernel Call";
                                    message.Check = false;
                                    message.Potential = false;
                                    messagesData.push_back(message);
                                }
                                else
                                {
                                    // Check if the kernel arguments contain an array or pointer variable
                                    for (int m = 0; m < kernelArgs; m++)
                                    {
                                        string argOfKernel = separatedWords[m];
                                        returnMessage message = isCorrect(argOfKernel, codeWords, codeLines, callLineNumber);
                                        messagesData.push_back(message);
                                    }
                                }
                                bool threadSync = false;
                                for (int m = callLineNumber + 1; m < callLineNumber + 4; m++)
                                {
                                    for (int n = 0; n < MAX_WORDS_PER_LINE; n++)
                                    {
                                        if (codeWords[m][n] == "cudaDeviceSynchronize" || codeWords[m][n] == "__syncthreads")
                                        {
                                            threadSync = true;
                                            break;
                                        }
                                    }
                                    if (threadSync == true)
                                        break;
                                }
                                if (threadSync == false)
                                {
                                    returnMessage message;
                                    message.errorLine = k;
                                    message.Error = "\t(CUDA) POTENTIAL RACE CONDITION: Threads are not being synchronized";
                                    message.Check = false;
                                    message.Potential = false;
                                    messagesData.push_back(message);
                                }
                                else
                                {
                                    returnMessage message;
                                    message.errorLine = k;
                                    message.Error = "\t(CUDA) No race condition";
                                    message.Check = true;
                                    message.Potential = false;
                                    messagesData.push_back(message);
                                }
                            }
                        }
                    }
                }         
            }
            // If cudaMalloc is found
            else if (codeWords[i][j] == "cudaMalloc")
            {
                if (cudaHeaderCheck == true)
                {
                    int mallocLineNumber = i;
                    cout << "cudaMalloc found at line " << mallocLineNumber << endl << endl;

                    // Extract the code line containing cudaMalloc
                    string codeLine = codeLines[mallocLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this cudaMalloc
                    grammar_cudaMalloc MallocCUDA;

                    // cudaMalloc must have 2 arguments
                    if (vecLength != 2)
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(CUDA) SYNTAX: Invalid number of arguments in cudaMalloc";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store cudaMalloc data in structure
                        MallocCUDA.mallocLineNumber = mallocLineNumber;
                        MallocCUDA.devPtr = separatedWords[0];
                        MallocCUDA.size = separatedWords[1];

                        // Save this cudaMalloc's data in cudaMallocData vector
                        cudaMallocData.push_back(MallocCUDA);
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(CUDA) <cuda.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
            // if cudaMemcpy is found
            else if (codeWords[i][j] == "cudaMemcpy")
            {
                if (cudaHeaderCheck == true)
                {
                    int memCpyLineNumber = i;
                    cout << "cudaMemcpy found at line " << memCpyLineNumber << endl << endl;

                    // Extract the code line containing cudaMemcpy
                    string codeLine = codeLines[memCpyLineNumber];

                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this cudaMemcpy
                    grammar_cudaMemcpy memCpyCUDA;

                    // cudaMemcpy must have 4 arguments
                    if (vecLength != 4)
                    {
                        returnMessage message;
                        message.errorLine = memCpyLineNumber;
                        message.Error = "\t(CUDA) SYNTAX: Invalid number of arguments in cudaMemcpy";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store cudaMemcpy data in structure
                        memCpyCUDA.memCpyLineNumber = memCpyLineNumber;
                        memCpyCUDA.destination = separatedWords[0];
                        memCpyCUDA.source = separatedWords[1];
                        memCpyCUDA.count = separatedWords[2];
                        memCpyCUDA.kind = separatedWords[3];

                        // Save this cudaMemcpy's data in cudaMemcpy vector
                        cudaMemcpyData.push_back(memCpyCUDA);
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(CUDA) <cuda.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
            // If omp_lock_init is found
            else if (codeWords[i][j] == "omp_init_lock")
            {
                if (ompHeaderCheck == true)
                {
                    int lockInitLineNumber = i;
                    cout << "OMP Lock initialization found at line " << lockInitLineNumber << endl << endl;

                    // Extract the code line containing cudaMemcpy
                    string codeLine = codeLines[lockInitLineNumber];
                    
                    // Separate the words within brackets at commas
                    vector <string> separatedWords = separateAtCommas(codeLine);

                    // Determine the length
                    int vecLength = size(separatedWords);

                    // Check the grammar of this cudaMemcpy
                    grammar_omp_init_lock ompInitLock;

                    // cudaMemcpy must have 4 arguments
                    if (vecLength != 1)
                    {
                        returnMessage message;
                        message.errorLine = lockInitLineNumber;
                        message.Error = "\t(CUDA) Invalid number of arguments in omp_init_lock";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                    else
                    {
                        // Store omp_init_lock data in structure
                        ompInitLock.lockLineNumber = lockInitLineNumber;
                        ompInitLock.lockVariable = separatedWords[0];

                        // Identify the lock variable name
                        ompLockVariable = ompInitLock.lockVariable;

                        // Save this omp_init_lock data
                        ompLockData.push_back(ompInitLock);
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(OpenMP) <omp.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
            // if pragma construct is found
            else if (codeWords[i][j] == "pragma" || codeWords[i][j] == "#pragma")
            {
                if (ompHeaderCheck == true)
                {
                    // Extract the code line containing cudaMemcpy
                    int pragmaLineNumber = i;
                    string codeLine = codeLines[pragmaLineNumber];
                    
                    // Check if the pragma construct contains a 'omp for' or 'omp parallel'
                    size_t foundOmpFor = codeLine.find("omp for");
                    size_t foundOmpParallel = codeLine.find("omp parallel");
                    size_t foundOmpParallelFor = codeLine.find("omp parallel for");
                    if (foundOmpFor != string::npos || foundOmpParallelFor != string::npos)
                    {
                        cout << "'omp (parallel) for' found at line " << pragmaLineNumber << endl << endl;
                        
                        // Check if the next line starts with 'for' loop
                        bool startsWithFor = checkOmpFor(codeLines, pragmaLineNumber);
                        if (startsWithFor == true)
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) Correct 'omp for' call";
                            message.Check = true;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) 'omp for' call must be followed by a 'for' loop";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }                       
                    }
                    else if (foundOmpParallel != string::npos)
                    {
                        cout << "'omp parallel' found at line " << pragmaLineNumber << endl << endl;

                        // Check if the next line starts with '{' 
                        bool startsWithBracket = checkOmpParallel(codeLines, pragmaLineNumber);
                        if (startsWithBracket == true)
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) Correct 'omp parallel' call";
                            message.Check = true;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) 'omp parallel' must be followed by code enclosed in {}";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        // Check if the critical section of the code is protected with locks
                        bool lockFound = false;
                        for (int k = pragmaLineNumber; k < numLines; k++)
                        {
                            for (int l = 0; l < MAX_WORDS_PER_LINE; l++)
                            {
                                if (codeWords[k][l] == "omp_set_lock")
                                {
                                    lockFound = true;
                                    break;
                                }
                            }
                            if (lockFound == true)
                                break;
                        }
                        if (lockFound == true)
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) Critical section protected";
                            message.Check = true;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            returnMessage message;
                            message.errorLine = pragmaLineNumber;
                            message.Error = "\t(OpenMP) RACE CONDITION: Critical section is not protected";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        // Check for a possible livelock scenario
                        bool isLiveLock = false;
                        for (int l = pragmaLineNumber + 1; l < pragmaLineNumber + 20; l++)
                        {
                            // Search for a while (flag) statement
                            string whileFlagLine = codeLines[l];
                            // Remove whitespace characters from the line
                            whileFlagLine.erase(remove_if(whileFlagLine.begin(), whileFlagLine.end(), ::isspace), whileFlagLine.end());

                            // Check if the modified string contains "while (flag)"
                            string searchString1 = "while(flag)";
                            size_t found1 = whileFlagLine.find(searchString1);

                            // If found, we have a potential livelock situation     
                            if (found1 != std::string::npos)
                            {                              
                                returnMessage message1;
                                message1.errorLine = l;
                                message1.Error = "\t(OpenMP) LIVELOCK: Threads stuck in an infinite loop trying to acquire lock";
                                message1.Check = false;
                                message1.Potential = false;
                                messagesData.push_back(message1);   
                                isLiveLock = true;
                                break;
                            }
                        }
                        if (isLiveLock == false)
                        {
                            returnMessage message1;
                            message1.errorLine = pragmaLineNumber;
                            message1.Error = "\t(OpenMP) No livelock";
                            message1.Check = true;
                            message1.Potential = false;
                            messagesData.push_back(message1);
                        }
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(OpenMP) <omp.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
            // If omp lock is being set
            else if (codeWords[i][j] == "omp_set_lock")
            {
                if (ompHeaderCheck == true)
                {
                    int setLockLineNumber = i;
                    cout << "OMP Lock set at line " << setLockLineNumber << endl << endl;

                    if (ompLockVariable != "NULL")
                    {
                        bool foundUnset = false;
                        // Find unsetting of lock
                        for (int a = setLockLineNumber + 1; a < numLines; a++)
                        {
                            for (int b = 0; b < MAX_WORDS_PER_LINE; b++)
                            {
                                if (codeWords[a][b] == "omp_unset_lock")
                                {
                                    foundUnset = true;
                                    break;
                                }
                            }
                            if (foundUnset == true)
                                break;
                        }
                        if (foundUnset == true)
                        {
                            returnMessage message;
                            message.errorLine = i;
                            message.Error = "\t(OpenMP) Correct usage of lock and unlock";
                            message.Check = true;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                        else
                        {
                            returnMessage message;
                            message.errorLine = i;
                            message.Error = "\t(OpenMP) DEADLOCK: OMP lock is set but never unset";
                            message.Check = false;
                            message.Potential = false;
                            messagesData.push_back(message);
                        }
                    }
                    else
                    {
                        returnMessage message;
                        message.errorLine = i;
                        message.Error = "\t(OpenMP) Setting LOCK without intializing it";
                        message.Check = false;
                        message.Potential = false;
                        messagesData.push_back(message);
                    }
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(OpenMP) <omp.h> HEADER file NOT INCLUDED";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }             
            }
        }
        // Search for a while (true) statement
        string whileTrueLine = codeLines[i];
        // Remove whitespace characters from the line
        whileTrueLine.erase(std::remove_if(whileTrueLine.begin(), whileTrueLine.end(), ::isspace), whileTrueLine.end());

        // Check if the modified string contains "while (true)"
        string searchString1 = "while(true)";
        size_t found1 = whileTrueLine.find(searchString1);

        // If found, search for a break
        bool foundBreak = false;
        bool isCudaWhile = false;
        if (found1 != std::string::npos)
        {
            for (int c = i + 1; c < i + 20; c++)
            {
                for (int d = 0; d < MAX_WORDS_PER_LINE; d++)
                {
                    if (codeWords[c][d] == kernelFuncName)
                    {
                        isCudaWhile = true;
                        break;
                    }
                    else if (codeWords[c][d] == "break")
                    {
                        foundBreak = true;
                        break;
                    }
                }
                if (isCudaWhile == true || foundBreak == true)
                    break;
            }
            if (isCudaWhile == false)
            {
                if (foundBreak == false)
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(MPI) LIVELOCK: MPI processes are stuck in an infinite loop exchanging messages";
                    message.Check = false;
                    message.Potential = false;
                    messagesData.push_back(message);
                }
                else
                {
                    returnMessage message;
                    message.errorLine = i;
                    message.Error = "\t(MPI) No livelock";
                    message.Check = true;
                    message.Potential = false;
                    messagesData.push_back(message);
                }
            }
        }
    }
    // Print out the error messages on screen
    printErrorMessages(messagesData);

    // Write to Output file
    CreateOutputFiles(messagesData, mpiSendData, mpiRecvData,
        mpiIsendData, mpiIrecvData, mpiBcastData, 
        mpiReduceData, cudaMallocData, cudaMemcpyData);

    // Launch the dynamic part for evaluation of potential errors
    ExecuteDynamic();

    return 0;
}


