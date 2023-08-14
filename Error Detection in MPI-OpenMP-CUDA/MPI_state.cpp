#include "MPI_state.h"

/* Function definitions */
/* ------------------------------------------------------- */
// Function for obtaining rank variable for MPI_Comm_rank
string Get_Rank_Variable(string mpiCommrankRank)
{
	string mpiRank;
	// Check if there is an '&' symbol. If yes, remove it
	if (mpiCommrankRank.substr(0, 1) == "&")
		mpiRank = mpiCommrankRank.substr(1);
	else
		mpiRank = mpiCommrankRank;

	// Return mpi rank
	return mpiRank;
}

// Function for checking correctness of MPI_Send
returnMessage Check_MPI_Send(grammar_MPI_Send mpiSend, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank, string senderRank)
{
	returnMessage message;
	int searchLine = 0,  recvLineNumber = 0;
	string rankLine;

	// Required string for correct rank match
	string requiredStr = mpiRank + '=' + '=' + mpiSend.sendDestination;

	// Check if the right rank is receiving the send message
	for (int i = mpiSend.sendLineNumber; i < numLines; i++)
	{
		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
		{
			// If rank variable is found
			if (codeWords[i][j] == mpiRank)
			{
				// Identify the rank line 
				rankLine = codeLines[i];
				string matchRankLine;

				// Extract the matching rank line if any
				matchRankLine = findMPIRank(rankLine);

				// Check if it matches the required string
				if (matchRankLine == requiredStr)
				{
					searchLine = i;
					break;
				}
			}
		}
		if (searchLine != 0)
			break;
	}

	// If a required matching receiving rank is found, search for matching MPI_Recv
	if (searchLine != 0)
	{
		// Search the tokens for MPI_Recv in the lines after MPI_Send
		for (int i = searchLine; i < numLines; i++)
		{
			for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
			{
				// if MPI_Recv is found
				if (codeWords[i][j] == "MPI_Recv")
				{
					recvLineNumber = i;
					break;
				}
			}
			if (recvLineNumber != 0)
				break;
		}
		// Check for match if an MPI_Recv is found
		if (recvLineNumber != 0)
		{
			// Extract the code line containing MPI_Recv
			string codeLine = codeLines[recvLineNumber];

			// Separate the words within brackets at commas
			vector <string> separatedWords = separateAtCommas(codeLine);

			// Determine the length
			int vecLength = size(separatedWords);

			// MPI_Recv has 7 argurments
			if (vecLength != 7)
			{
				message.errorLine = recvLineNumber;
				message.Error = "\t(MPI) SYNTAX: Incorrect number of arguments in MPI_Recv";
				message.Check = false;
				message.Potential = false;
				return message;
			}
			else
			{
				// Check if the send and recv counts match
				if (separatedWords[1] != mpiSend.sendCount)
				{
					message.errorLine = recvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Send and Receive COUNTS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the send and recv datatypes match
				else if (separatedWords[2] != mpiSend.sendDatatype)
				{
					message.errorLine = recvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Send and Receive DATATYPES do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				else if (separatedWords[3] != senderRank && separatedWords[3] != "i")
				{
					if (separatedWords[3] == "MPI_ANY_SOURCE")
					{
						message.errorLine = mpiSend.sendLineNumber;
						message.Error = "\t(MPI) POTENTIAL ERROR in RECV (Being Investigated in Recv)";
						message.Check = true;
						message.Potential = true;
						return message;
					}
					else
					{
						message.errorLine = mpiSend.sendLineNumber;
						message.Error = "\t(MPI) DEADLOCK: NO CORRESPONDING RECV found";
						message.Check = false;
						message.Potential = false;
						return message;
					}
				}
				// Check if the send and recv tags match
				else if (separatedWords[4] != mpiSend.sendTag)
				{
					if (separatedWords[4] == "MPI_ANY_TAG")
					{
						message.errorLine = mpiSend.sendLineNumber;
						message.Error = "\t(MPI) POTENTIAL ERROR in RECV (Being Investigated in Recv)";
						message.Check = true;
						message.Potential = true;
						return message;
					}
					else
					{
						message.errorLine = recvLineNumber;
						message.Error = "\t(MPI) MISMATCH: MPI Send and Recv TAGS do not match";
						message.Check = false;
						message.Potential = false;
						return message;
					}
				}
				// Check if the send and recv communicators match
				else if (separatedWords[5] != mpiSend.sendCommunicator)
				{
					message.errorLine = recvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Send and Recv COMMUNICATORS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Else we have found a matching MPI_Recv
				else
				{
					message.errorLine = recvLineNumber;
					message.Error = "\t(MPI) Matching MPI_recv FOUND at line ";
					message.Check = true;
					message.Potential = false;
					return message;
				}
			}
		}
		// If matching MPI_Recv is not found
		else
		{
			message.errorLine = mpiSend.sendLineNumber;
			message.Error = "\t(MPI) DEADLOCK: Matching MPI_Recv not found for MPI_Send ";
			message.Check = false;
			message.Potential = false;
			return message;
		}
	}
	// If a matching receiving rank is not found
	else
	{
		message.errorLine = mpiSend.sendLineNumber;
		message.Error = "\t(MPI) DEADLOCK: The required DESTINATION RANK IS NOT RECEIVING the message";
		message.Check = false;
		message.Potential = false;
		return message;
	}
}

// Function for checking correctness of MPI_Recv
returnMessage Check_MPI_Recv(grammar_MPI_Recv mpiRecv, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int startLine, string mpiRank, string receiverRank)
{
	returnMessage message;
	int searchLine = 0, sendLineNumber = 0;
	string rankLine, requiredStr;

	// Check for wild card source and tag
	if (mpiRecv.recvSource == "MPI_ANY_SOURCE")
	{
		message.errorLine = mpiRecv.recvLineNumber;
		message.Error = "\t(MPI) WILDCARD SOURCE: Needs further investigation";
		message.Check = false;
		message.Potential = true;
		return message;
	}
	else if (mpiRecv.recvTag == "MPI_ANY_TAG")
	{
		message.errorLine = mpiRecv.recvLineNumber;
		message.Error = "\t(MPI) WILDCARD TAG: Needs further investigation";
		message.Check = false;
		message.Potential = true;
		return message;		
	}
	else
	{
		// Required string for correct rank match
		requiredStr = mpiRank + '=' + '=' + mpiRecv.recvSource;
		
		// Check if the right rank is sending the received message
		for (int i = mpiRecv.recvLineNumber - 1; i >= startLine; i--)
		{
			for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
			{
				// If rank variable is found
				if (codeWords[i][j] == mpiRank)
				{
					// Identify the rank line 
					rankLine = codeLines[i];
					string matchRankLine;

					// Extract the matching rank line if any
					matchRankLine = findMPIRank(rankLine);

					// Check if it matches the required string
					if (matchRankLine == requiredStr)
					{
						searchLine = i;
						break;
					}
				}
			}
			if (searchLine != 0)
				break;
		}
		if (searchLine != 0)
		{
			// Search the tokens for MPI_Send in the lines before MPI_Recv
			for (int i = searchLine/*mpiRecv.recvLineNumber - 1*/; i < mpiRecv.recvLineNumber/*>= startLine*/; i++/*--*/)
			{
				for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
				{
					// if MPI_Recv is found
					if (codeWords[i][j] == "MPI_Send")
					{
						sendLineNumber = i;
						break;
					}
				}
				if (sendLineNumber != 0)
					break;
			}
			// Check for match if an MPI_Send is found
			if (sendLineNumber != 0)
			{
				// Extract the code line containing MPI_Send
				string codeLine = codeLines[sendLineNumber];

				// Separate the words within brackets at commas
				vector <string> separatedWords = separateAtCommas(codeLine);

				// Determine the length
				int vecLength = size(separatedWords);

				// MPI_Send has 7 argurments
				if (vecLength != 6)
				{
					message.errorLine = sendLineNumber;
					message.Error = "\t(MPI) SYNTAX: Incorrect number of arguments in MPI_Send";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				else
				{
					// Check if the send and recv counts match
					if (separatedWords[1] != mpiRecv.recvCount)
					{
						message.errorLine = sendLineNumber;
						message.Error = "\t(MPI) MISMATCH: MPI Send and Receive COUNTS do not match";
						message.Check = false;
						message.Potential = false;
						return message;
					}
					// Check if the send and recv datatypes match
					else if (separatedWords[2] != mpiRecv.recvDatatype)
					{
						message.errorLine = sendLineNumber;
						message.Error = "\t(MPI) MISMATCH: MPI Send and Receive DATATYPES do not match";
						message.Check = false;
						message.Potential = false;
						return message;
					}
					// Check if the send and recv ranks match
					else if (separatedWords[3] != receiverRank && separatedWords[3] != "i")
					{
						message.errorLine = mpiRecv.recvLineNumber;
						message.Error = "\t(MPI) DEADLOCK: NO CORRESPONDING SEND found";
						message.Check = false;
						message.Potential = false;
						return message;
					}
					// Check if the send and recv tags match
					else if (separatedWords[4] != mpiRecv.recvTag)
					{						
						message.errorLine = sendLineNumber;
						message.Error = "\t(MPI) MISMATCH: MPI Send and Recv TAGS do not match";
						message.Check = false;
						message.Potential = false;
						return message;						
					}
					// Check if the send and recv communicators match
					else if (separatedWords[5] != mpiRecv.recvCommunicator)
					{
						message.errorLine = sendLineNumber;
						message.Error = "\t(MPI) MISMATCH: MPI Send and Recv COMMUNICATORS do not match";
						message.Check = false;
						message.Potential = false;
						return message;
					}
					// Else we have found a matching MPI_Send
					else
					{
						message.errorLine = sendLineNumber;
						message.Error = "\t(MPI) Matching MPI_Send FOUND at line ";
						message.Check = true;
						message.Potential = false;
						return message;
					}
				}
			}
			// If matching MPI_Send is not found
			else
			{
				message.errorLine = mpiRecv.recvLineNumber;
				message.Error = "\t(MPI) DEADLOCK: Matching MPI_Send not found for MPI_Recv ";
				message.Check = false;
				message.Potential = false;
				return message;
			}
		}
		else
		{
			message.errorLine = mpiRecv.recvLineNumber;
			message.Error = "\t(MPI) DEADLOCK: The required SOURCE RANK IS NOT SENDING the message ";
			message.Check = false;
			message.Potential = false;
			return message;
		}
	}	
}
// Function for checking correctness of MPI_Isend
returnMessage Check_MPI_Isend(grammar_MPI_Isend mpiIsend, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank, string senderRank)
{
	returnMessage message;
	int searchLine = 0, IrecvLineNumber = 0;
	string rankLine;

	// Required string for correct rank match
	string requiredStr = mpiRank + '=' + '=' + mpiIsend.IsendDestination;

	// Check if the right rank is receiving the send message
	for (int i = 0; i < numLines; i++)
	{
		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
		{
			// If rank variable is found
			if (codeWords[i][j] == mpiRank)
			{
				// Identify the rank line 
				rankLine = codeLines[i];
				string matchRankLine;

				// Extract the matching rank line if any
				matchRankLine = findMPIRank(rankLine);

				// Check if it matches the required string
				if (matchRankLine == requiredStr)
				{
					searchLine = i;
					break;
				}
			}
		}
		if (searchLine != 0)
			break;
	}

	// If a required matching receiving rank is found, search for matching MPI_Irecv
	if (searchLine != 0)
	{
		// Search the tokens for MPI_Irecv in the lines after MPI_Send
		for (int i = mpiIsend.IsendLineNumber + 1; i < numLines; i++)
		{
			for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
			{
				// if MPI_Irecv is found
				if (codeWords[i][j] == "MPI_Irecv")
				{
					IrecvLineNumber = i;
					break;
				}
			}
		}
		// Check for match if an MPI_Irecv is found
		if (IrecvLineNumber != 0)
		{
			// Extract the code line containing MPI_Irecv
			string codeLine = codeLines[IrecvLineNumber];

			// Separate the words within brackets at commas
			vector <string> separatedWords = separateAtCommas(codeLine);

			// Determine the length
			int vecLength = size(separatedWords);

			// MPI_Irecv has 7 argurments
			if (vecLength != 7)
			{
				message.errorLine = IrecvLineNumber;
				message.Error = "\t(MPI) SYNTAX: Incorrect number of arguments in MPI_Irecv";
				message.Check = false;
				message.Potential = false;
				return message;
			}
			else
			{
				// Check if the Isend and Irecv counts match
				if (separatedWords[1] != mpiIsend.IsendCount)
				{
					message.errorLine = IrecvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv COUNTS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv datatypes match
				else if (separatedWords[2] != mpiIsend.IsendDatatype)
				{
					message.errorLine = IrecvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv DATATYPES do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				else if (separatedWords[3] != senderRank)
				{
					message.errorLine = mpiIsend.IsendLineNumber;
					message.Error = "\t(MPI) DEADLOCK: NO CORRESPONDING Irecv found";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv tags match
				else if (separatedWords[4] != mpiIsend.IsendTag)
				{
					message.errorLine = IrecvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv TAGS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv communicators match
				else if (separatedWords[5] != mpiIsend.IsendCommunicator)
				{
					message.errorLine = IrecvLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv COMMUNICATORS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Else we have found a matching MPI_Irecv
				else
				{
					message.errorLine = IrecvLineNumber;
					message.Error = "\t(MPI) Matching MPI_Irecv FOUND at line ";
					message.Check = true;
					message.Potential = false;
					return message;
				}
			}
		}
		// If matching MPI_Iecv is not found
		else
		{
			message.errorLine = mpiIsend.IsendLineNumber;
			message.Error = "\t(MPI) DEADLOCK: Matching MPI_Irecv not found for MPI_Isend ";
			message.Check = false;
			message.Potential = false;
			return message;
		}
	}
	// If a matching receiving rank is not found
	else
	{
		message.errorLine = mpiIsend.IsendLineNumber;
		message.Error = "\t(MPI) DEADLOCK: The required DESTINATION RANK IS NOT RECEIVING the message";
		message.Check = false;
		message.Potential = false;
		return message;
	}
}
// Function for checking correctness of MPI_Irecv
returnMessage Check_MPI_Irecv(grammar_MPI_Irecv mpiIrecv, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int startLine, string mpiRank, string receiverRank)
{
	returnMessage message;
	int searchLine = 0, IsendLineNumber = 0;
	string rankLine;

	// Required string for correct rank match
	string requiredStr = mpiRank + '=' + '=' + mpiIrecv.IrecvSource;

	// Check if the right rank is sending the received message
	for (int i = mpiIrecv.IrecvLineNumber - 1; i >= startLine; i--)
	{
		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
		{
			// If rank variable is found
			if (codeWords[i][j] == mpiRank)
			{
				// Identify the rank line 
				rankLine = codeLines[i];
				string matchRankLine;

				// Extract the matching rank line if any
				matchRankLine = findMPIRank(rankLine);

				// Check if it matches the required string
				if (matchRankLine == requiredStr)
				{
					searchLine = i;
					break;
				}
			}
		}
		if (searchLine != 0)
			break;
	}

	if (searchLine != 0)
	{
		// Search the tokens for MPI_Isend in the lines before MPI_Irecv
		for (int i = mpiIrecv.IrecvLineNumber - 1; i >= startLine; i--)
		{
			for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
			{
				// if MPI_Isend is found
				if (codeWords[i][j] == "MPI_Isend")
				{
					IsendLineNumber = i;
					break;
				}
			}
		}
		// Check for match if an MPI_Isend is found
		if (IsendLineNumber != 0)
		{
			// Extract the code line containing MPI_Isend
			string codeLine = codeLines[IsendLineNumber];

			// Separate the words within brackets at commas
			vector <string> separatedWords = separateAtCommas(codeLine);

			// Determine the length
			int vecLength = size(separatedWords);

			// MPI_Isend has 7 argurments
			if (vecLength != 7)
			{
				message.errorLine = IsendLineNumber;
				message.Error = "\t(MPI) SYNTAX: Incorrect number of arguments in MPI_Isend";
				message.Check = false;
				message.Potential = false;
				return message;
			}
			else
			{
				// Check if the Isend and Irecv counts match
				if (separatedWords[1] != mpiIrecv.IrecvCount)
				{
					message.errorLine = IsendLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv COUNTS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv datatypes match
				else if (separatedWords[2] != mpiIrecv.IrecvDatatype)
				{
					message.errorLine = IsendLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv DATATYPES do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv ranks match
				else if (separatedWords[3] != receiverRank)
				{
					message.errorLine = mpiIrecv.IrecvLineNumber;
					message.Error = "\t(MPI) DEADLOCK: NO CORRESPONDING Isend found";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv tags match
				else if (separatedWords[4] != mpiIrecv.IrecvTag)
				{
					message.errorLine = IsendLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv TAGS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Check if the Isend and Irecv communicators match
				else if (separatedWords[5] != mpiIrecv.IrecvCommunicator)
				{
					message.errorLine = IsendLineNumber;
					message.Error = "\t(MPI) MISMATCH: MPI Isend and Irecv COMMUNICATORS do not match";
					message.Check = false;
					message.Potential = false;
					return message;
				}
				// Else we have found a matching MPI_Isend
				else
				{
					message.errorLine = IsendLineNumber;
					message.Error = "\t(MPI) Matching MPI_Isend FOUND at line ";
					message.Check = true;
					message.Potential = false;
					return message;
				}
			}
		}
		// If matching MPI_Isend is not found
		else
		{
			message.errorLine = mpiIrecv.IrecvLineNumber;
			message.Error = "\t(MPI) DEADLOCK: Matching MPI_Isend not found for MPI_Irecv ";
			message.Check = false;
			message.Potential = false;
			return message;
		}
	}
	else
	{
		message.errorLine = mpiIrecv.IrecvLineNumber;
		message.Error = "\t(MPI) DEADLOCK: The required SOURCE RANK IS NOT SENDING the message ";
		message.Check = false;
		message.Potential = false;
		return message;
	}
}


// Function for checking correct calling of MPI_Bcast
//returnMessage Check_MPI_Bcast(grammar_MPI_Bcast mpiBcast, string codeWords[][MAX_WORDS_PER_LINE], string codeLines[], int numLines, string mpiRank)
//{
//	SpaceOrTab bcastSpacesTabs, rankLineSpacesTabs;
//	int bcastLineNumber = mpiBcast.bcastLineNumber;
//	string bcastLine = codeLines[bcastLineNumber];
//	string rankLine;
//
//	// Required difference in indent
//	int difference = 0, reqDifference = 0;
//
//	// Count and check whether leading characters are spaces or tabs in bcastLine
//	bcastSpacesTabs = countSpacesTabs(bcastLine);
//
//	for (int i = bcastLineNumber - 1; i > 0; i--)
//	{
//		for (int j = 0; j < MAX_WORDS_PER_LINE; j++)
//		{
//			// If rank variable is found
//			if (codeWords[i][j] == mpiRank)
//			{
//				// Identify the rank line 
//				rankLine = codeLines[i];
//
//				// Find '=='
//				size_t found = rankLine.find("==");
//
//				// If line contains '=='
//				if (found != std::string::npos)
//				{
//					// Count and check whether leading characters are spaces or tabs in rankLine
//					rankLineSpacesTabs = countSpacesTabs(rankLine);
//
//					// Calculate the difference
//					difference = (bcastSpacesTabs.count - rankLineSpacesTabs.count);
//				}
//			}
//		}
//	}
//}
