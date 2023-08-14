#include "ListErrors.h"

// Function for writing the errors to an output .txt file
void CreateOutputFiles(vector <returnMessage> messagesData, vector <grammar_MPI_Send> mpiSendData,
    vector <grammar_MPI_Recv> mpiRecvData, vector <grammar_MPI_Isend> mpiIsendData,
    vector <grammar_MPI_Irecv> mpiIrecvData, vector <grammar_MPI_Bcast> mpiBcastData,
    vector <grammar_MPI_Reduce> mpiReduceData, vector <grammar_cudaMalloc> cudaMallocData,
    vector <grammar_cudaMemcpy> cudaMemcpyData)
{
    // Create and open the actual erros file, potential errors file and log file
    ofstream OutFileActual("ActualErrors.txt");
    ofstream OutFilePotential("PotentialErrors.txt");
    ofstream OutFileLog("LogFile.txt");

    // First, write all possible errors to the file
    int vecLength = size(messagesData);
    int flagActual = 0, flagPotential = 0;
    OutFileActual << "########################################################################" << endl;
    OutFileActual << "#################*******  LIST OF ACTUAL ERRORS ******##################" << endl;
    OutFileActual << "########################################################################" << endl << endl;
    OutFilePotential << "########################################################################" << endl;
    OutFilePotential << "################*******  LIST OF POTENTIAL ERRORS ******################" << endl;
    OutFilePotential << "########################################################################" << endl << endl;

    
    for (int i = 0; i < vecLength; i++)
    {  
        if (messagesData[i].Check == false && messagesData[i].Potential == false)
        {
            flagActual = 1;
            OutFileActual << "ERROR (Line " << messagesData[i].errorLine << ")" << endl;
            OutFileActual << messagesData[i].Error << endl << endl;
        }
        else if (messagesData[i].Check == false && messagesData[i].Potential == true)
        {
            flagPotential = 1;
            OutFilePotential << "Potential ERROR (Line " << messagesData[i].errorLine << ")" << endl;
            OutFilePotential << messagesData[i].Error << endl << endl;
        }     
    }
    if (flagActual == 0)
        OutFileActual << "NO ACTUAL ERRORS FOUND" << endl;
    if (flagPotential == 0)
        OutFilePotential << "NO POTENTIAL ERRORS FOUND" << endl;

    // Write MPI_Send and Recv Info to log file
    OutFileLog << endl << endl;
    OutFileLog << "########################################################################" << endl;
    OutFileLog << "#############******  MPI BLOCKING COMMUNICATION INFO ******#############" << endl;
    OutFileLog << "########################################################################" << endl << endl;
    OutFileLog << "Total number of MPI_Send:  " << size(mpiSendData) << endl;
    for (int i = 0; i < (int)size(mpiSendData); i++)
    {
        OutFileLog << "MPI_Send number " << mpiSendData[i].sendID << " found at line:  " << mpiSendData[i].sendLineNumber << endl;
        if (mpiSendData[i].correspondingReceive != 0)
            OutFileLog << "\tCorresponding Receive found at line:  " << mpiSendData[i].correspondingReceive << endl;
        else
            OutFileLog << "\tNo Corresponding Receive found" << endl;
    }
    OutFileLog << endl << "Total number of MPI_Recv:  " << size(mpiRecvData) << endl;
    for (int i = 0; i < (int)size(mpiRecvData); i++)
    {
        OutFileLog << "MPI_Recv number " << mpiRecvData[i].recvID << " found at line:  " << mpiRecvData[i].recvLineNumber << endl;
        if (mpiRecvData[i].correspondingSend != 0)
            OutFileLog << "\tCorresponding Send found at line:  " << mpiRecvData[i].correspondingSend << endl;
        else
            OutFileLog << "\tNo Corresponding Send found" << endl;
    }

    // Write MPI_Isend and Irecv Info to file
    OutFileLog << endl << endl;
    OutFileLog << "########################################################################" << endl;
    OutFileLog << "############*****  MPI NON BLOCKING COMMUNICATION INFO *****############" << endl;
    OutFileLog << "########################################################################" << endl << endl;
    OutFileLog << "Total number of MPI_Isend:  " << size(mpiIsendData) << endl;
    for (int i = 0; i < (int)size(mpiIsendData); i++)
    {
        OutFileLog << "MPI_Isend number " << mpiIsendData[i].IsendID << " found at line:  " << mpiIsendData[i].IsendLineNumber << endl;
        if (mpiIsendData[i].correspondingIReceive != 0)
            OutFileLog << "\tCorresponding Irecv found at line:  " << mpiIsendData[i].correspondingIReceive << endl;
        else
            OutFileLog << "\tNo Corresponding Irecv found" << endl;
    }
    OutFileLog << endl << "Total number of MPI_Irecv:  " << size(mpiIrecvData) << endl;
    for (int i = 0; i < (int)size(mpiIrecvData); i++)
    {
        OutFileLog << "MPI_Irecv number " << mpiIrecvData[i].IrecvID << " found at line:  " << mpiIrecvData[i].IrecvLineNumber << endl;
        if (mpiIrecvData[i].correspondingISend != 0)
            OutFileLog << "\tCorresponding Isend found at line:  " << mpiIrecvData[i].correspondingISend << endl;
        else
            OutFileLog << "\tNo Corresponding Isend found" << endl;
    }
    
    // Write MPI_Bcast and MPI_Reduce info to file
    OutFileLog << endl << endl;
    OutFileLog << "########################################################################" << endl;
    OutFileLog << "############******  MPI COLLECTIVE COMMUNICATION INFO ******############" << endl;
    OutFileLog << "########################################################################" << endl << endl;
    OutFileLog << "Total number of MPI_Bcast:  " << size(mpiBcastData) << endl;
    for (int i = 0; i < (int)size(mpiBcastData); i++)
        OutFileLog << "MPI_Bcast number " << mpiBcastData[i].bcastID << " found at line:  " << mpiBcastData[i].bcastLineNumber << endl;
    OutFileLog << endl << "Total number of MPI_Reduce:  " << size(mpiReduceData) << endl;
    for (int i = 0; i < (int)size(mpiReduceData); i++)   
        OutFileLog << "MPI_Reduce number " << mpiReduceData[i].reduceID << " found at line:  " << mpiReduceData[i].reduceLineNumber << endl;
   

    // Close the output file
    OutFileActual.close();
    OutFilePotential.close();
    OutFileLog.close();
}
