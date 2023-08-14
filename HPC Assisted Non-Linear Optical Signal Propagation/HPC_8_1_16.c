//Author: Rana Ahmad Bilal Khalid
//Python code used for help by Prof. Stylianos Sygletos
//Parallelized Implementation of the SSF method for solving NLSE
//==============================================================

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<fftw3.h>
#include<time.h>
#include<complex.h>
#include<omp.h>

//Define required constant values
//-------------------------------
#define PI 3.141592654

//Constants for real and imaginary axis of 'fftw_complex' arrays
#define REAL 0
#define IMAG 1

//Minimum and maximum constant functions
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

//Define a structure for returning 'mQAM_Tx function' outputs
struct ModulatedQAM_Tx
{
	double _Complex* DataMod;
	double _Complex* TxSignal;
};

//Define a structure for returning 'qam_modulate function' outputs
struct Modulated_Data
{
	double _Complex* DataMod;
	double _Complex* Alphabet;
};

//Define a structure for returning 'Gaussian_Filter function' outputs
struct Gaussian
{
	int* TimeIndex;
	double* h_Gaussian;
};

//Function definitions (Detailed descriptions after 'MAIN')
//===============================================================================
int GenRandBin();
int* bin2dec(double Symbols, double k, int* Matrix);
int* dec2bin(double Symbols, double k, int* vector);
double* rrcosine(double rolloff, double span, double NSpS);
struct Gaussian GaussianFilter(double beta, double span, double NSpS);
struct Modulated_Data qam_modulate(int* DataSymbolsIn, double M, double Symbols);
void fft(fftw_complex* IN, fftw_complex* OUT, int Size);
void ifft(fftw_complex* IN, fftw_complex* OUT, int Size);
void parallel_fft(fftw_complex* IN, fftw_complex* OUT, int Size, int thread_count);
void parallel_ifft(fftw_complex* IN, fftw_complex* OUT, int Size, int thread_count);
void fftshift(fftw_complex* IN, fftw_complex* OUT, int Size);
void ifftshift(fftw_complex* IN, fftw_complex* OUT, int Size);
fftw_complex* NLS_SMF_SinglePol(fftw_complex* Ein, double Length, double Gamma, double Disp, double a_loss, double* w, double MaxPhaseChange, double MaxStepWidth, int Ein_Length, int thread_count);
fftw_complex* AmpCG(fftw_complex* Eout, double Gain, double NF, double SampleRate, double NSamples);
fftw_complex* Disp_Eq(fftw_complex* Ein, double Disp, double Length, double* w, double NSamples, int thread_count);
double _Complex* convolve(double* h, double _Complex* x, int lenH, int lenX, int* lenY);
double _Complex* upfirdn(double* h, double _Complex* x, int up, int down, int lenH, int lenX, int* lenY);
struct ModulatedQAM_Tx mQAM_Tx(double M, double symbols, double k, double NSpS, double* FilterResp, double Span, int* lenTxSignal, int* DataInMatrix);
int* qam_demodulate(double _Complex* Rx_Sig, double M, int lenRxSignal);
double _Complex* Phase_Eq(double _Complex* RxSignal, int lenRxSignal);
double Calc_BER(int* DataInMatrix, int* DataOutMatrix, double NumBits);
//=================================================================================


int main(void)
{
	clock_t start, finish;	

	double M = 16;												//Modulation Level
	double k = log2(M);											//Bits per symbol

	double Symbols = pow(2, 16);								//Total Symbols
	double NumBits = Symbols * k;								//Total Number of Bits
	double NSpS = 16;											//Number of Samples per Symbol
	int N_spans = 1;											//Number of span divisions
	double BaudRate = 32 * pow(10, 9);							//Baud Rate
	double SampleRate = NSpS * BaudRate;						//Sampling frequency

	double SimulWind = Symbols / BaudRate;						//Simulation Window
	double Ws = 2 * PI * SampleRate;							
	double NSamples = NSpS * Symbols;							//Total number of Samples

	//Ask user input for number of threads
	int thread_count = 8;
	//printf("Enter the number of threads that will execute fft/ifft blocks:  ");
	//scanf("%d", &thread_count);
	//printf("\n\n");

	//Initialize FFT use with multiple threads
	int fftw_init_threads(void);

	//Creating Time axis
	double* Time = NULL;									
	Time = (double*)malloc(NSamples * sizeof(double));			
	for (int i = 0; i < NSamples; i++)
	{															
		Time[i] = i * ((SimulWind - (1 / SampleRate)) / (NSamples - 1));   
	}

	//Creating the frequency axis
	double* w = NULL;
	w = (double*)malloc(NSamples * sizeof(double));				
	for (int i = 0; i < NSamples; i++)
	{																	
		w[i] = Ws * (((-NSamples / 2) + (double)(i)) / NSamples);		
	}

	//Create Raised Cosine Filter
	double Span = 64;											//Filter span in symbols
	double RollOff = 0.1;										//Rolloff factor for raised cosine transmitter
	double* rrcFilter = rrcosine(RollOff, Span, NSpS);			//Create a root-raised cosine filter impulse response

	//Fiber parameters of the nonlinear fiber
	double Pch_dBm_min = -10;
	//double Pch_dBm_max = 10;
	double a_loss = 0.2 * pow(10, -3);							//Fiber loss coefficient
	double Disp = 17 * pow(10, -6);								//Dispersion constant
	double Length_km = 100;										//Transmission length in km
	double Length = Length_km * 1000;							//Transmission length in metres
	double Gamma = 1.4 * pow(10, -3);
	double MaxPhaseChange = 0.005;
	double MaxStepWidth = 1000;
	double Gain_dB = a_loss * Length;							//Gain (dB)
	double Gain = pow(10, (Gain_dB / 10));						//Gain
	double NF_dB = 4.5;											//Noise figure (dB)
	double NF = pow(10, (NF_dB / 10));							//Noise figure
	//int batch_total = 15;										//15 batches of length = 100 km

	
	for (int PchdBm_index = (int)Pch_dBm_min; PchdBm_index <= (int)Pch_dBm_min; PchdBm_index++)
	{
		double Pch_dBm = (double)PchdBm_index;
		double Pch = pow(10, Pch_dBm / 10) * pow(10, -3);
		
		int lenTxSignal, i;		//Ein Length
		//Generate and Modulate the random binary data
		int* DataInMatrix = NULL;
		DataInMatrix = (int*)malloc(Symbols * k * sizeof(int));

		//Get random binary numbers in a matrix of dimension [symbols x k]
		for (int i = 0; i < Symbols; i++)
		{
			for (int j = 0; j < k; j++)
			{
				DataInMatrix[(int)(i * k) + j] = GenRandBin();
			}
		}
		struct ModulatedQAM_Tx MD_Tx = mQAM_Tx(M, Symbols, k, NSpS, rrcFilter, Span, &lenTxSignal, DataInMatrix);

		//Determine the mean of 'TxSignal'
		double TxSignal_mean = 0;
		printf("\n%d\n\n", lenTxSignal);

		for (int i = 0; i < lenTxSignal; i++)
		{
			double temp = pow(cabs(MD_Tx.TxSignal[i]), 2);
			TxSignal_mean += temp;
		}
		TxSignal_mean /= (double)lenTxSignal;
		double RootTxSignal_mean = sqrt(TxSignal_mean);

		//Pulse shaping and saving result in FFTW_complex array 'Ein'
		fftw_complex* Ein = NULL;
		Ein = (fftw_complex*)fftw_malloc(lenTxSignal * sizeof(fftw_complex));

		for (int i = 0; i < lenTxSignal; i++)
		{
			//Pulse Shaping
			double realtemp_TxSignal = creal(MD_Tx.TxSignal[i]);
			double imagtemp_TxSignal = cimag(MD_Tx.TxSignal[i]);
			Ein[i][REAL] = (realtemp_TxSignal / RootTxSignal_mean) * sqrt(Pch);
			Ein[i][IMAG] = (imagtemp_TxSignal / RootTxSignal_mean) * sqrt(Pch);
		}
		start = clock();
		for (int span_index = 1; span_index <= N_spans; span_index++)
		{
			//Calaculate output signal 'Eout' after passing through a single mode nonlinear fiber of 'length'
			fftw_complex* Eout = NLS_SMF_SinglePol(Ein, Length, Gamma, Disp, a_loss, w, MaxPhaseChange, MaxStepWidth, lenTxSignal, thread_count);

			//Signal Amplification using a constant gain amplifier
			Ein = AmpCG(Eout, Gain, NF, SampleRate, NSamples);

			printf("Batch Index:  0     PchdBm Index:  %d     Span Index:  %d\n", PchdBm_index, span_index);
			fftw_free(Eout);
		}
		finish = clock();
		double elapsed = ((double)(finish - start) / CLOCKS_PER_SEC);
		printf("\n\nDone! :)   Time taken with %d threads =   %.6f sec\n\n", thread_count, elapsed/thread_count);

		//Dispersion equalization and storing result in 'E_disp_eq'
		fftw_complex* E_disp_eq = Disp_Eq(Ein, Disp, (Length * N_spans), w, NSamples, thread_count);

		//Output signal normalization and storing result in 'RxSigRin_x'
		double _Complex* RxSigRin_x = NULL;
		RxSigRin_x = (double _Complex*)malloc(lenTxSignal * sizeof(double _Complex));

#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < lenTxSignal; i++)
		{
			//Singal Normalization
			RxSigRin_x[i] = CMPLX ( (E_disp_eq[i][REAL] / lenTxSignal) * (RootTxSignal_mean / sqrt(Pch)), (E_disp_eq[i][IMAG] / lenTxSignal) * (RootTxSignal_mean / sqrt(Pch)) );
		}
		
		//Downsampling the received signal
		//Storing length of downfiltered signal in 'lenRxSignal_Filtered'
		int lenRxSignal_Filtered;	
		double _Complex* RxFiltSignal = upfirdn(rrcFilter, RxSigRin_x, 1, (int)NSpS, (int)(Span * NSpS + 1), lenTxSignal, &lenRxSignal_Filtered);

		//Store length of relevant part of received signal in lenRxSignal
		int lenRxSignal = lenRxSignal_Filtered - (2 * (Span / 2));

		//Extract the relevant received symbols and store in RxSignal
		double _Complex* RxSignal = NULL;
		RxSignal = (double _Complex*)malloc(lenRxSignal * sizeof(double _Complex));

#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < lenRxSignal; i++)
		{
			RxSignal[i] = RxFiltSignal[i + (int)(Span / 2)];
			//fprintf(fptr, "%.8f     %.8f\n", creal(RxSignal[i]), cimag(RxSignal[i]));
		}

		/* Phase Equalization */
		double _Complex* RxCorrected = Phase_Eq(RxSignal, lenRxSignal);

		//Demodulate the signal
		int* DataSymbolsOut = qam_demodulate(RxCorrected, M, lenRxSignal);

		//Convert the symbols into binary form
		int* DataOutMatrix = dec2bin(Symbols, k, DataSymbolsOut);

		//Calculate Bit Error Rate
		double BER = Calc_BER(DataInMatrix, DataOutMatrix, NumBits);
		printf("\n\nBER =   %f\n\n", BER);

		//free dynamically allocated arrays
		fftw_free(Ein);
		fftw_free(E_disp_eq);
		free(RxSigRin_x);
		free(RxFiltSignal);
		free(RxSignal);
	}
		
	//Free dynamically allocated arrays
	free(Time);
	free(w);
	free(rrcFilter);

	return 0;
}

//==============================================================
//FUNCTION DESCRIPTIONS BELOW ==================================
//==============================================================


//Generates random binary number i.e. either 0 or 1
//Outputs:  Binary number (int)
//-------------------------------------------------
int GenRandBin()
{
	int num;
	//Generate random number between -1 and 1
	double rand_num = (((double)rand() / RAND_MAX) * 2) - 1;
	//Make the numbers either 0 or 1 (binary)
	if (rand_num < 0)
		num = 0;
	else
		num = 1;

	return num;
}

//Converts a binary matrix into an array of non-negative decimal integers
//Inputs:   Symbols (double), Bits/Symbol (double), 
//          Input data Matrix of size (symbols x k) (Type: int)
//Outputs:  Output Symbols vector of size (1 x symbols) (Type: int)
//-----------------------------------------------------------------------
int* bin2dec(double Symbols, double k, int* Matrix)
{
	//Create the outpur vector and allocate memory
	int* OutVector = NULL;
	OutVector = (int*)malloc(Symbols * sizeof(int));

	//Convert each row of input matrix into decimal value
	for (int i = 0; i < Symbols; i++)
	{
		int DecVal = 0;
		for (int j = 0; j < k; j++)
		{
			int temp = Matrix[(int)(i * k) + j] * pow(2, j);
			DecVal = DecVal + temp;
		}
		OutVector[i] = DecVal;
	}
	//Return the output vector
	return OutVector;
}

//Converts an array of non-negative decimal integers into a binary matrix
//Each row in matrix is a binary form of corresponding element in array
//Inputs:   Symbols (double), Bits/Symbol (double),
//          Received data symbols vector of size (1 x symbols) (type: int)
//Outputs:  Output data vector of size (symbols x k) (type: int)
//-----------------------------------------------------------------------
int* dec2bin(double Symbols, double k, int* vector)
{
	//Create the output matrix and allocate memory
	int* OutMatrix = NULL;
	OutMatrix = (int*)malloc(Symbols * k * sizeof(int));
	
	//Convert each symbol into a row of it's equivalent binary value
	for (int i = 0; i < Symbols; i++)
	{
		int temp = vector[i];
		for (int j = 0; j < k; j++)
		{
			int rem = temp % 2;
			OutMatrix[(int)(i * k) + j] = rem;
			temp = temp / 2;
		}
	}
	//return the output matrix
	return OutMatrix;
}

//Impulse response of a square root raised cosine filter
//Inputs:  Rolloff factor, Span, Samples/Symbol
//Output:  Array of type (double) & length (span * NSpS)
//------------------------------------------------------
double* rrcosine(double rolloff, double span, double NSpS)
{
	double N = span * NSpS;
	double Ts = NSpS;
	double* h_rrc = NULL;
	h_rrc = (double*)malloc((N + 1) * sizeof(double));

	for (int i = 0; i <= N; i++)
	{
		//Model the impulse response of a raised cosine filter
		int t = i - (N / 2);
		if (t == 0)
		{
			h_rrc[i] = (1 / sqrt(Ts)) * (1 - rolloff + (4 * rolloff / PI));
		}
		else if ((rolloff != 0) && (t == Ts / (4 * rolloff) || t == -Ts / (4 * rolloff)))
		{
			h_rrc[i] = (1 / sqrt(Ts)) * (rolloff / (sqrt(2))) * (((1 + 2 / PI) * (sin(PI / (4 * rolloff)))) + ((1 - 2 / PI) * (cos(PI / (4 * rolloff)))));
		}
		else
		{
			h_rrc[i] = (1 / sqrt(Ts)) * (sin(PI * t * (1 - rolloff) / Ts) + 4 * rolloff * (t / Ts) * cos(PI * t * (1 + rolloff) / Ts)) / (PI * t * (1 - (4 * rolloff * t / Ts) * (4 * rolloff * t / Ts)) / Ts);
		}
	}
	return h_rrc;
}

//Impulse response of a Gaussian Filter
//------------------------------------------------------------------
struct Gaussian GaussianFilter(double beta, double span, double NSpS)
{
	double Alpha = NSpS * beta;
	double N = span * NSpS;
	//double Ts = NSpS;

	struct Gaussian GA;
	GA.TimeIndex = (int*)malloc((N + 1) * sizeof(int));
	GA.h_Gaussian = (double*)malloc((N + 1) * sizeof(double));

	for (int i = 0; i <= N; i++)
	{
		GA.TimeIndex[i] = (int)(i - N / 2);
		GA.h_Gaussian[i] = (sqrt(PI) / Alpha) * exp(-(pow(PI * GA.TimeIndex[i] / Alpha, 2)));
	}
	return GA;
}

//Modulates the DataSymbolsIn according to their complex coordinates
//-----------------------------------------------------------------------------
struct Modulated_Data qam_modulate(int* DataSymbolsIn, double M, double Symbols)
{
	double order = sqrt(M);
	double* quadrat_iq = NULL;
	quadrat_iq = (double*)malloc(M * sizeof(double));

	for (int i = 0; i < order; i++)
	{
		quadrat_iq[i] = (1 - order) + (2 * i);
	}

	struct Modulated_Data MD;
	MD.Alphabet = (double _Complex*)malloc(order * order * sizeof(double _Complex));
	MD.DataMod = (double _Complex*)malloc(Symbols * sizeof(double _Complex));

	for (int i = 0; i < order; i++)
	{
		for (int j = 0; j < order; j++)
		{
			MD.Alphabet[(int)(i * order) + j] = CMPLX( quadrat_iq[i], quadrat_iq[j] );
		}
	}
	for (int i = 0; i < Symbols; i++)
	{
		int temp = DataSymbolsIn[i];
		MD.DataMod[i] = MD.Alphabet[temp];
	}
	return MD;
}

//Function for calculating 1-D FFT
//----------------------------------------------------
void fft(fftw_complex* IN, fftw_complex* OUT, int Size)
{
	fftw_plan plan;

	plan = fftw_plan_dft_1d(Size, IN, OUT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	fftw_destroy_plan(plan);
	fftw_cleanup();
}

//Function for calculating 1-D IFFT
//-----------------------------------------------------
void ifft(fftw_complex* IN, fftw_complex* OUT, int Size)
{
	fftw_plan plan;

	plan = fftw_plan_dft_1d(Size, IN, OUT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	//Scale the output
	/*for (int i = 0; i < Size; i++)
	{
		OUT[i][REAL] /= Size;
		OUT[i][IMAG] /= Size;
	}*/

	fftw_destroy_plan(plan);
	fftw_cleanup();
}

//Function for calculating 1-D parallel FFT
//---------------------------------------------------------
void parallel_fft(fftw_complex* IN, fftw_complex* OUT, int Size, int thread_count)
{
	fftw_plan_with_nthreads(thread_count);
	fftw_plan plan;

	plan = fftw_plan_dft_1d(Size, IN, OUT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	fftw_destroy_plan(plan);
	void fftw_cleanup_threads(void);
}

//Function for calculating 1-D parallel IFFT
//---------------------------------------------------------
void parallel_ifft(fftw_complex* IN, fftw_complex* OUT, int Size, int thread_count)
{
	fftw_plan_with_nthreads(thread_count);
	fftw_plan plan;

	plan = fftw_plan_dft_1d(Size, IN, OUT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	//Scale the output
	/*for (int i = 0; i < Size; i++)
	{
		OUT[i][REAL] /= Size;
		OUT[i][IMAG] /= Size;
	}*/

	fftw_destroy_plan(plan);
	void fftw_cleanup_threads(void);
}

//Function for applying fftshift on an array
//---------------------------------------------------------
void fftshift(fftw_complex* IN, fftw_complex* OUT, int Size)
{
	if (Size % 2 == 0) // Even Case
	{
		fftw_complex* temp = NULL;
		temp = (fftw_complex*)fftw_malloc((Size / 2) * sizeof(fftw_complex));

		for (int i = 0; i < Size / 2; i++)
		{
			temp[i][REAL] = IN[i][REAL];
			temp[i][IMAG] = IN[i][IMAG];
			OUT[i][REAL] = IN[(Size / 2) + i][REAL];
			OUT[i][IMAG] = IN[(Size / 2) + i][IMAG];
			OUT[(Size / 2) + i][REAL] = temp[i][REAL];
			OUT[(Size / 2) + i][IMAG] = temp[i][IMAG];
		}
		fftw_free(temp);
	}
	else	// Odd case
	{
		fftw_complex* temp = NULL;
		temp = (fftw_complex*)fftw_malloc((Size / 2) * sizeof(fftw_complex));

		for (int i = (Size / 2) + 1; i < Size; i++)
		{
			temp[i - ((Size / 2) + 1)][REAL] = IN[i][REAL];
			temp[i - ((Size / 2) + 1)][IMAG] = IN[i][IMAG];
		}
		for (int i = 0; i <= (Size / 2); i++)
		{
			OUT[Size - 1 - i][REAL] = IN[(Size / 2) - i][REAL];
			OUT[Size - 1 - i][IMAG] = IN[(Size / 2) - i][IMAG];
		}
		for (int i = 0; i < (Size / 2); i++)
		{
			OUT[i][REAL] = temp[i][REAL];
			OUT[i][IMAG] = temp[i][IMAG];
		}
		fftw_free(temp);
	}
}

//Function for applying ifftshift on an array
//----------------------------------------------------------
void ifftshift(fftw_complex* IN, fftw_complex* OUT, int Size)
{
	if (Size % 2 == 0) // Even Case
	{
		fftw_complex* temp = NULL;
		temp = (fftw_complex*)fftw_malloc((Size / 2) * sizeof(fftw_complex));

		for (int i = 0; i < Size / 2; i++)
		{
			temp[i][REAL] = IN[i][REAL];
			temp[i][IMAG] = IN[i][IMAG];
			OUT[i][REAL] = IN[(Size / 2) + i][REAL];
			OUT[i][IMAG] = IN[(Size / 2) + i][IMAG];
			OUT[(Size / 2) + i][REAL] = temp[i][REAL];
			OUT[(Size / 2) + i][IMAG] = temp[i][IMAG];
		}
		fftw_free(temp);
	}
	else	// Odd case
	{
		fftw_complex* temp = NULL;
		temp = (fftw_complex*)fftw_malloc((Size / 2) * sizeof(fftw_complex));

		for (int i = 0; i < (Size / 2); i++)
		{
			temp[i][REAL] = IN[i][REAL];
			temp[i][IMAG] = IN[i][IMAG];
		}
		for (int i = 0; i <= (Size / 2); i++)
		{
			OUT[i][REAL] = IN[(Size / 2) + i][REAL];
			OUT[i][IMAG] = IN[(Size / 2) + i][IMAG];
		}
		for (int i = (Size / 2) + 1; i < Size; i++)
		{
			OUT[i][REAL] = temp[i - ((Size / 2) + 1)][REAL];
			OUT[i][IMAG] = temp[i - ((Size / 2) + 1)][IMAG];
		}
		fftw_free(temp);
	}
}

//Function implements the Non-Linear Schrodinger Equation for the case of Single Polarization, Single Mode Fibers
//---------------------------------------------------------------------------------------------------------------
fftw_complex* NLS_SMF_SinglePol(fftw_complex* Ein, double Length, double Gamma, double Disp, double a_loss, double* w, double MaxPhaseChange, double MaxStepWidth, int Ein_Length, int thread_count)
{
	double z0 = 0;
	double c = 3 * pow(10, 8);
	double lambda0 = 1550 * pow(10, -9);
	double alpha = (a_loss) * (1 / log10(exp(1))) / 10;
	double beta2 = -(pow(lambda0, 2)) * Disp / (2 * PI * c);

	double dph_max = MaxPhaseChange;
	double _Complex* temp = NULL;
	double num, max = 0;
	temp = (double _Complex*)malloc(Ein_Length * sizeof(double _Complex));
	int i;

#	pragma omp parallel for num_threads(thread_count)
	for (i = 0; i < Ein_Length; i++)
	{
		temp[i] = CMPLX( Ein[i][REAL], Ein[i][IMAG] );
		num = pow(cabs(temp[i]), 2);
#		pragma omp critical
		{
			if (num >= max)
			{
				max = num;
			}
		}
	}
	free(temp);
	double dz_phase = dph_max / (Gamma * max);
	double dz;
	if (MaxStepWidth < dz_phase)
	{
		dz = MaxStepWidth;
	}
	else
	{
		dz = dz_phase;
	}

	double _Complex* TF_DispFiber = NULL;
	fftw_complex* PF = NULL;
	fftw_complex* E_out_disp = NULL;
	fftw_complex* E_out_nl = NULL;

	TF_DispFiber = (double _Complex*)malloc(Ein_Length * sizeof(double _Complex));
	PF = (fftw_complex*)fftw_malloc(Ein_Length * sizeof(fftw_complex));
	E_out_disp = (fftw_complex*)fftw_malloc(Ein_Length * sizeof(fftw_complex));
	E_out_nl = (fftw_complex*)fftw_malloc(Ein_Length * sizeof(fftw_complex));

	/* dz is the step-size which represents 'h' */
	while (z0 + dz < Length)
	{
		//First step (Propagate dz/2 distance with DISPERSION only)
		parallel_fft(Ein, PF, Ein_Length, thread_count);
		fftshift(PF, PF, Ein_Length);

#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < Ein_Length; i++)
		{
			double temp1 = (-beta2) * (pow(w[i], 2) / 2) * (dz / 2);
			TF_DispFiber[i] = CMPLX( 0, temp1 );
			TF_DispFiber[i] = cexp(TF_DispFiber[i]);

			double real_temp = creal(TF_DispFiber[i]);
			double imag_temp = cimag(TF_DispFiber[i]);
			real_temp = real_temp * exp((-(alpha / 2) * (dz / 2)));
			imag_temp = imag_temp * exp((-(alpha / 2) * (dz / 2)));

			TF_DispFiber[i] = CMPLX( real_temp, imag_temp );

			double PF_temp_real = PF[i][REAL];
			PF[i][REAL] = (PF[i][REAL] * real_temp) - (PF[i][IMAG] * imag_temp);
			PF[i][IMAG] = (PF_temp_real * imag_temp) + (PF[i][IMAG] * real_temp);
		}
		ifftshift(PF, PF, Ein_Length);
		parallel_ifft(PF, E_out_disp, Ein_Length, thread_count);

		//2nd Step (Non-linear section)
#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < Ein_Length; i++)
		{
			double _Complex temp2 = CMPLX( (E_out_disp[i][REAL] / Ein_Length), (E_out_disp[i][IMAG] / Ein_Length) );
			double abs_temp = cabs(temp2);
			abs_temp = Gamma * dz * pow(abs_temp, 2);
			temp2 = CMPLX( 0, (-1) * abs_temp );
			temp2 = cexp(temp2);

			E_out_nl[i][REAL] = ((E_out_disp[i][REAL] / Ein_Length) * creal(temp2)) - ((E_out_disp[i][IMAG] / Ein_Length) * cimag(temp2));
			E_out_nl[i][IMAG] = ((E_out_disp[i][REAL] / Ein_Length) * cimag(temp2)) + ((E_out_disp[i][IMAG] / Ein_Length) * creal(temp2));
		}

		//Final step (Propagate the remaining dz/2 distance considering dispersion only)
		parallel_fft(E_out_nl, PF, Ein_Length, thread_count);
		fftshift(PF, PF, Ein_Length);

#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < Ein_Length; i++)
		{
			double PF_temp_real = PF[i][REAL];
			PF[i][REAL] = (PF[i][REAL] * creal(TF_DispFiber[i])) - (PF[i][IMAG] * cimag(TF_DispFiber[i]));
			PF[i][IMAG] = (PF_temp_real * cimag(TF_DispFiber[i])) + (PF[i][IMAG] * creal(TF_DispFiber[i]));
		}
		ifftshift(PF, PF, Ein_Length);
		parallel_ifft(PF, E_out_disp, Ein_Length, thread_count);

		//Calculate parameters for the next step
		z0 = z0 + dz;
		Ein = E_out_disp;
		double _Complex* temp1 = NULL;
		double num1, max1 = 0;
		temp1 = (double _Complex*)malloc(Ein_Length * sizeof(double _Complex));

#		pragma omp parallel for num_threads(thread_count)
		for (i = 0; i < Ein_Length; i++)
		{
			Ein[i][REAL] /= Ein_Length;
			Ein[i][IMAG] /= Ein_Length;
			temp1[i] = CMPLX( Ein[i][REAL], Ein[i][IMAG] );
			num1 = pow(cabs(temp1[i]), 2);
#			pragma omp critical
			{
				if (num1 >= max1)
				{
					max1 = num1;
				}
			}
		}
		free(temp1);
		dz_phase = dph_max / (Gamma * max1);
		if (MaxStepWidth < dz_phase)
		{
			dz = MaxStepWidth;
		}
		else
		{
			dz = dz_phase;
		}
	}
	//Final Step
	dz = Length - z0;

	//First step (dispersive of Fourier)
	parallel_fft(Ein, PF, Ein_Length, thread_count);
	fftshift(PF, PF, Ein_Length);

#	pragma omp parallel for num_threads(thread_count)
	for (i = 0; i < Ein_Length; i++)
	{
		double temp1 = (-beta2) * (pow(w[i], 2) / 2) * (dz / 2);
		TF_DispFiber[i] = CMPLX( 0, temp1 );
		TF_DispFiber[i] = cexp(TF_DispFiber[i]);

		double real_temp = creal(TF_DispFiber[i]);
		double imag_temp = cimag(TF_DispFiber[i]);
		real_temp = real_temp * exp((-(alpha / 2) * (dz / 2)));
		imag_temp = imag_temp * exp((-(alpha / 2) * (dz / 2)));

		TF_DispFiber[i] = CMPLX( real_temp, imag_temp );

		double PF_temp_real = PF[i][REAL];
		PF[i][REAL] = (PF[i][REAL] * real_temp) - (PF[i][IMAG] * imag_temp);
		PF[i][IMAG] = (PF_temp_real * imag_temp) + (PF[i][IMAG] * real_temp);
	}
	ifftshift(PF, PF, Ein_Length);
	parallel_ifft(PF, E_out_disp, Ein_Length, thread_count);

	//Second Step (nonlinear section)
#	pragma omp parallel for num_threads(thread_count)
	for (i = 0; i < Ein_Length; i++)
	{
		double _Complex temp2 = CMPLX( (E_out_disp[i][REAL] / Ein_Length), (E_out_disp[i][IMAG] / Ein_Length) );
		double abs_temp = cabs(temp2);
		abs_temp = Gamma * dz * pow(abs_temp, 2);
		temp2 = CMPLX( 0, (-1) * abs_temp );
		temp2 = cexp(temp2);

		E_out_nl[i][REAL] = ((E_out_disp[i][REAL] / Ein_Length) * creal(temp2)) - ((E_out_disp[i][IMAG] / Ein_Length) * cimag(temp2));
		E_out_nl[i][IMAG] = ((E_out_disp[i][REAL] / Ein_Length) * cimag(temp2)) + ((E_out_disp[i][IMAG] / Ein_Length) * creal(temp2));
	}

	//Final Step (dispersive of Fourier)
	parallel_fft(E_out_nl, PF, Ein_Length, thread_count);
	fftshift(PF, PF, Ein_Length);

#	pragma omp parallel for num_threads(thread_count)
	for (i = 0; i < Ein_Length; i++)
	{
		double PF_temp_real = PF[i][REAL];
		PF[i][REAL] = (PF[i][REAL] * creal(TF_DispFiber[i])) - (PF[i][IMAG] * cimag(TF_DispFiber[i]));
		PF[i][IMAG] = (PF_temp_real * cimag(TF_DispFiber[i])) + (PF[i][IMAG] * creal(TF_DispFiber[i]));
	}
	ifftshift(PF, PF, Ein_Length);
	parallel_ifft(PF, E_out_disp, Ein_Length, thread_count);

	return E_out_disp;
}

//Function for a Constant Gain Optical Amplifier
//----------------------------------------------
fftw_complex* AmpCG(fftw_complex* Eout, double Gain, double NF, double SampleRate, double NSamples)
{
	double h = 6.63 * pow(10, -34);
	double fc = 193.1 * pow(10, 12);
	double No = h * fc * (Gain - 1) * (NF / 2);
	double randnum1, randnum2;
	double factor = (pow(No * SampleRate / 2, 0.5));
	fftw_complex* NewEin;
	NewEin = (fftw_complex*)fftw_malloc(NSamples * sizeof(fftw_complex));

	for (int i = 0; i < NSamples; i++)
	{
		randnum1 = (((double)rand() / RAND_MAX) * 4) - 2;
		randnum2 = (((double)rand() / RAND_MAX) * 4) - 2;
		randnum1 = randnum1 * factor;
		randnum2 = randnum2 * factor;

		//Adding noise to the real part
		NewEin[i][REAL] = (pow(Gain, 0.5) * (Eout[i][REAL] / NSamples)) + randnum1;
		//Adding noise to the imaginary part
		NewEin[i][IMAG] = (pow(Gain, 0.5) * (Eout[i][IMAG] / NSamples)) + randnum2;
	}
	return NewEin;
}

//Function for equalization of Linear Dispersive Effects
//------------------------------------------------------
fftw_complex* Disp_Eq(fftw_complex* Ein, double Disp, double Length, double* w, double NSamples, int thread_count)
{
	double c = 3 * pow(10, 8);
	double lambda0 = 1550 * pow(10, -9);
	double beta2 = -(pow(lambda0, 2)) * Disp / (2 * PI * c);
	int i;

	double _Complex* TF_DispFiber1 = NULL;
	fftw_complex* PF1 = NULL;
	TF_DispFiber1 = (double _Complex*)malloc(NSamples * sizeof(double _Complex));
	PF1 = (fftw_complex*)fftw_malloc(NSamples * sizeof(fftw_complex));

	//fft(Ein, PF1, NSamples);
	parallel_fft(Ein, PF1, NSamples, thread_count);
	fftshift(PF1, PF1, NSamples);

#	pragma omp parallel for num_threads(thread_count)
	for (i = 0; i < (int)NSamples; i++)
	{
		double temporary = (beta2) * (pow(w[i], 2) / 2) * (Length);
		TF_DispFiber1[i] = CMPLX( 0 , temporary );
		TF_DispFiber1[i] = cexp(TF_DispFiber1[i]);

		double PF1_temp_real = PF1[i][REAL];
		PF1[i][REAL] = (PF1[i][REAL] * creal(TF_DispFiber1[i])) - (PF1[i][IMAG] * cimag(TF_DispFiber1[i]));
		PF1[i][IMAG] = (PF1_temp_real * cimag(TF_DispFiber1[i])) + (PF1[i][IMAG] * creal(TF_DispFiber1[i]));
	}
	ifftshift(PF1, PF1, NSamples);
	parallel_ifft(PF1, PF1, NSamples, thread_count);
	//ifft(PF1, PF1, NSamples);

	return PF1;
}

//Function for convolving two 1-Dimensional arrays
//------------------------------------------------
double _Complex* convolve(double* h, double _Complex* x, int lenH, int lenX, int* lenY)
{
	//Determine the convolution length
	int nconv = lenH + lenX - 1;
	//Store in the address of output Length
	*lenY = nconv;

	int i, j, h_start, x_start, x_end;
	double _Complex* y = NULL;
	y = (double _Complex*)malloc(nconv * sizeof(double _Complex));

	//Convolution process
	for (i = 0; i < nconv; i++)
	{
		//Determine the start/end index
		x_start = MAX(0, i - lenH + 1);
		x_end = MIN(i + 1, lenX);
		h_start = MIN(i, (lenH - 1));

		y[i] = CMPLX( 0, 0 );
		double real_ytemp = creal(y[i]);
		double imag_ytemp = cimag(y[i]);
		double real_xtemp;
		double imag_xtemp;

		for (j = x_start; j < x_end; j++)
		{
			real_xtemp = creal(x[j]);
			imag_xtemp = cimag(x[j]);
			real_ytemp += h[h_start] * real_xtemp;
			imag_ytemp += h[h_start] * imag_xtemp;
			h_start--;
		}
		y[i] = CMPLX( real_ytemp, imag_ytemp );
	}
	return y;
}

//Function for upsampling, FIR filtering & downsampling
//Defaul values of 'up' and 'down' should be given as 1
//-----------------------------------------------------
double _Complex* upfirdn(double* h, double _Complex* x, int up, int down, int lenH, int lenX, int* lenY)
{
	//(UpSampling) Creating zero padded input signal 
	//Padding determined by upsampling rate i.e NSpS in this case
	int len_new_x = lenX * up;
	double _Complex* new_x = NULL;
	new_x = (double _Complex*)malloc(len_new_x * sizeof(double _Complex));
	for (int i = 0; i < (len_new_x); i++)
	{
		if (i % up == 0)
		{
			new_x[i] = x[i / up];
		}
		else
		{
			//Zero Padding
			new_x[i] = CMPLX( 0, 0 );
		}
	}
	//FIR Filtering (Convolution of input signal and filter impulse response)
	//Length of the output of 'convolve' function
	int temp_lenY;
	double _Complex* y = convolve(h, new_x, lenH, len_new_x, &temp_lenY);

	//(DownSampling) Decimation by a factor 'down'
	double _Complex* y_final = NULL;
	y_final = (double _Complex*)malloc((temp_lenY / down) * sizeof(double _Complex));
	double real_yfinal, imag_yfinal;

	//Final length of output of 'upfirdn' function
	*lenY = (temp_lenY / down);

	for (int i = 0; i < (temp_lenY / down); i++)
	{
		//Only copying values which are multiples of 'down'
		real_yfinal = creal(y[i * down]);
		imag_yfinal = cimag(y[i * down]);
		y_final[i] = CMPLX( real_yfinal, imag_yfinal );
	}
	return y_final;
}

//Function for creating the QAM modulated transmission signal
//-----------------------------------------------------------
struct ModulatedQAM_Tx mQAM_Tx(double M, double symbols, double k, double NSpS, double* FilterResp, double Span, int* lenTxSignal, int* DataInMatrix)
{
	//Get decimal value corresponding to each symbol
	int* DataSymbolsIn;
	DataSymbolsIn = bin2dec(symbols, k, DataInMatrix);

	//Modulate the input data symbols
	struct Modulated_Data MD = qam_modulate(DataSymbolsIn, M, symbols);

	//Upsample/Filter/Downsaple the modulated data
	//and store length of filtered signal in 'lenSig_Filtered'
	int lenSig_Filtered;
	double _Complex* TxSignal_Filtered = upfirdn(FilterResp, MD.DataMod, NSpS, 1, (int)(Span * NSpS + 1), (int)symbols, &lenSig_Filtered);

	//Store length of final TxSignal in 'lenTxSignal'
	*lenTxSignal = (lenSig_Filtered - (2 * (Span * (NSpS / 2))));

	//Create object for storing 'mQAM_Tx' final data
	struct ModulatedQAM_Tx MD_Tx;
	MD_Tx.TxSignal = (double _Complex*)malloc((*lenTxSignal) * sizeof(double _Complex));
	MD_Tx.DataMod = (double _Complex*)malloc(symbols * sizeof(double _Complex));

	//'MD_Tx.DataMod' is simply MD.DataMod (Copying)
	for (int i = 0; i < symbols; i++)
	{
		MD_Tx.DataMod[i] = MD.DataMod[i];
	}

	//Copy the relevant filtered signal into 'TxSignal'
	for (int i = 0; i < *lenTxSignal; i++)
	{
		MD_Tx.TxSignal[i] = TxSignal_Filtered[i + (int)(Span * (NSpS / 2))];
	}

	//Return the ModulatedQAM_Tx object
	return MD_Tx;
}

//Function for demodulating the input QAM symbols
int* qam_demodulate(double _Complex* Rx_Sig, double M, int lenRxSignal)
{
	double order = sqrt(M);
	//Create output data symbol integer values vector
	int* DataSymbolsOut = NULL;
	DataSymbolsOut = (int*)malloc(lenRxSignal * sizeof(int));
	for (int i = 0; i < lenRxSignal; i++)
	{
		double s_real = round((creal(Rx_Sig[i]) + order - 1) / 2);
		double s_imag = round((cimag(Rx_Sig[i]) + order - 1) / 2);

		//Clip the real value between 0 and (order - 1)
		if (s_real < 0)
			s_real = 0;
		if (s_real > (order - 1))
			s_real = order - 1;

		//Clip the imaginary value between 0 and (order - 1)
		if (s_imag < 0)
			s_imag = 0;
		if (s_imag > (order - 1))
			s_imag = order - 1;

		DataSymbolsOut[i] = (int)(order * s_real + s_imag);
	}
	return DataSymbolsOut;
}

//Function for calculating the BER
//Inputs: Received Bits, Reference Data Bits, Total number of bits
//Output: Bit Error Rate
//-------------------------------------------------------------------
double Calc_BER(int* DataInMatrix, int* DataOutMatrix, double NumBits)
{
	int NumErr = 0;	/* Initialize number of bits in error to zero */
	int* XOR_Array = NULL;	/* For storing result of XOR operation */
	XOR_Array = (int*)malloc(NumBits * sizeof(int));

	/* XOR operation between received bits and reference data bits */
	for (int i = 0; i < (int)(NumBits); i++)
	{
		XOR_Array[i] = DataOutMatrix[i] ^ DataInMatrix[i];
		/* XOR returns zero if both operands are same */
		NumErr += XOR_Array[i];
	}
	/* BER = number of bits in error / total number of bits */
	double BER = NumErr / NumBits;
	/* Return BER */
	return BER;
}


//Function for phase equalization
//Inputs: Received Signal (RxSignal), Length of RxSignal
//Output: Phase Equalized Signal (RxCorrected)
//-------------------------------------------------------
double _Complex* Phase_Eq(double _Complex* RxSignal, int lenRxSignal)
{
	/* Intitalize array for storing phase corrected signal */
	double _Complex* RxCorrected = NULL;
	RxCorrected = (double _Complex*)malloc(lenRxSignal * sizeof(double _Complex));

	double minimum_error = 100;		/* Initialize to any random high value */
	int correct_phase = 10;			/* Initialize to any angle value */

	int CheckLength;
	if (lenRxSignal > 10000)
	{
		CheckLength = 10000;
	}
	else
	{
		CheckLength = lenRxSignal;
	}

	/* Vary the phase angle from 0 to 360 degrees */
	for (int angle = 0; angle <= 360; angle++)
	{
		/* Calculate e^j(phi) */
		double _Complex mult_factor = CMPLX(0, (2 * PI * angle / 360));
		mult_factor = cexp(mult_factor);
		double sum = 0;		/* For storing sum of LMS error */

		/* Implementation of equation 3.1.8 */
		for (int i = 0; i < CheckLength; i++)
		{
			double real_temp = (creal(RxSignal[i]) * creal(mult_factor)) - (cimag(RxSignal[i]) * cimag(mult_factor));
			double imag_temp = (creal(RxSignal[i]) * cimag(mult_factor)) + (cimag(RxSignal[i]) * creal(mult_factor));
			RxCorrected[i] = CMPLX(real_temp - creal(RxSignal[i]), imag_temp - cimag(RxSignal[i]));

			/* Caclulate LMS Error */
			double error = pow(cabs(RxCorrected[i]), 2);
			sum += error;
		}
		double mean_error = sum / lenRxSignal;
		if (mean_error < minimum_error)
		{
			/* Adjust the phase angle to the value that gives minimum error */
			minimum_error = mean_error;
			correct_phase = angle;
		}
	}
	/* Adjust the phase of the received signal using the correct phase value */
	double _Complex mult_factor = CMPLX(0, (2 * PI * correct_phase / 360));
	mult_factor = cexp(mult_factor);
	for (int i = 0; i < lenRxSignal; i++)
	{
		double real_temp = (creal(RxSignal[i]) * creal(mult_factor)) - (cimag(RxSignal[i]) * cimag(mult_factor));
		double imag_temp = (creal(RxSignal[i]) * cimag(mult_factor)) + (cimag(RxSignal[i]) * creal(mult_factor));
		RxCorrected[i] = CMPLX(real_temp, imag_temp);
	}
	/* Return the phase equalized signal */
	return RxCorrected;
}