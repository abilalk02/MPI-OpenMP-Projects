
all: runFile


runFile: runFile.cpp
	g++ -g -Wall -c helper.cpp -lstdc++ -lm
	g++ -g -Wall -c lexical.cpp -lstdc++ -lm
	g++ -g -Wall -c MPI_state.cpp -lstdc++ -lm
	g++ -g -Wall -c CUDA_state.cpp -lstdc++ -lm
	g++ -g -Wall -c OMP_state.cpp -lstdc++ -lm
	g++ -g -Wall -c ListErrors.cpp -lstdc++ -lm
	g++ -g -Wall -c dynamic.cpp -lstdc++ -lm
	g++ -g -Wall -c runFile.cpp -lstdc++ -lm
	g++ helper.o lexical.o MPI_state.o CUDA_state.o OMP_state.o ListErrors.o dynamic.o runFile.o -o runFile

clean:
	$(RM) runFile runFile.o helper.o lexical.o MPI_state.o CUDA_state.o OMP_state.o ListErrors.o dynamic.o


run: run_runFile

run_runFile:
	./runFile
	

