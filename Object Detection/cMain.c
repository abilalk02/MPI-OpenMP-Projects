#include "headers.h"
#include <omp.h>
#include <mpi.h>
#include <cstddef>

#define ROOT 0

#define WORK_TAG 555
#define FINISH_TAG 666
#define RESULT_TAG 777

void printMatrix(Matrix *matrix);


int main(int argc, char **argv)
{
    // Variables
    int my_rank;
    int num_of_processes;
    int tag = WORK_TAG;
    int num_pictures, num_objects;
    int sent_picture_index = 0;
    double threshold;
    int local_finished = 0;
    int global_finished = 0;

    // input/output files
    FILE *input_file;
    FILE *output_file;

    // Initialization of MPI, getting the rank and the amount of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);

    // Set the number of threads for openmp
    int thread_count = 2;

    // The MPI_Matrix custom datatype to transfer a Matrix struct on MPI
    MPI_Datatype MPI_Matrix;
    int block_lengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3] = {offsetof(Matrix, picture_id), offsetof(Matrix, data), offsetof(Matrix, size)};
    MPI_Type_create_struct(3, block_lengths, offsets, types, &MPI_Matrix);
    MPI_Type_commit(&MPI_Matrix);

    Matrix* pictures = NULL;
    Matrix* object_matrixes = NULL;

    // master process
    if (my_rank == 0) {
        // Tracking how many active processes are there
        int active_processes = 0;

        // Open input file for reading
        input_file = fopen("input.txt", "r");
        if (!input_file)
        {
            fprintf(stderr, "Error: could not open input file\n");
            return (1);
        }

        // Open output file for writing
        output_file = fopen("output.txt", "w+");
        if (!output_file)
        {
            fprintf(stderr, "Error: could not open output file\n");
            return (1);
        }

        // Read the threshold value from the input file
        fscanf(input_file, "%lf", &threshold);

        // Read the number of pictures from the input file
        fscanf(input_file, "%d", &num_pictures);
        printf("number of pics %d\n", num_pictures);

        // Allocating memory for the picture matrixes.
        pictures = (Matrix*)calloc(num_pictures, sizeof(Matrix));

        // Reading the picture related data from the input file
        for (int i = 0; i < num_pictures; i++)
        {
            fscanf(input_file, "%d", &pictures[i].picture_id);
            fscanf(input_file, "%d", &pictures[i].size);
            printf("Size %d, id %d \n", pictures[i].size, pictures[i].picture_id);

            int s = pictures[i].size * pictures[i].size;
            pictures[i].data = (int*)malloc(s * sizeof(int));

            for (int j = 0; j < s; j++)
            {
                fscanf(input_file, "%d ", &pictures[i].data[j]);
            }
        }

        // Read the number of objects from the input file
        fscanf(input_file, "%d", &num_objects);

        printf("number of obj %d\n", num_objects);

        // Allocating memory for the object matrixes
        Matrix* objects = (Matrix*)calloc(num_objects, sizeof(Matrix));

        // Reading the object related data from the input file
        for (int i = 0; i < num_objects; i++) {
            fscanf(input_file, "%d", &objects[i].picture_id); //redundant but crucial to keep a single struct for everything
            fscanf(input_file, "%d", &objects[i].size);
            printf("Size %d, id %d \n", objects[i].size, objects[i].picture_id);
            int s = objects[i].size * objects[i].size;
            objects[i].data = (int*)malloc(s * sizeof(int));

            for (int j = 0; j < s; j++)
            {
                fscanf(input_file, "%d ", &objects[i].data[j]);
            }
        }

        // Limit the number of processes to the number of pictures, because there is no point in more processes working
        if (num_of_processes > num_pictures) {
            num_of_processes = num_pictures;
        }

        // Sending the objects to the MPI processes
        for (int i = 1; i < num_of_processes; i++) {
            // Sending the amount of objects that the MPI process should expect and the threshold
            MPI_Send(&num_objects, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&threshold, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

            // Sending the objects by utilizing the MPI_Pack / MPI_Unpack functions
            // I have tried various ways of transfering Structs in MPI but a custom datatype and MPI_Pack are the only things that worked for me
            // The buffer and the position assist in transfering the contiguous memory
            for (int j = 0; j < num_objects; j++) {
                char buffer[200000];
                int position = 0;
                MPI_Pack(&objects[j], 1, MPI_Matrix, buffer, 200000, &position, MPI_COMM_WORLD);
                MPI_Pack(objects[j].data, objects[j].size * objects[j].size, MPI_INT, buffer, 200000, &position, MPI_COMM_WORLD);
                MPI_Send(buffer, position, MPI_PACKED, i, 0, MPI_COMM_WORLD);
                //free(objects[j].matrix.data);
            }
        }
    }
    else
    {
        // MPI Processes 
        // Receive the amount of object matrixes that will soon come and the threshold against which the process should check
        MPI_Recv(&num_objects, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        object_matrixes = (Matrix*)calloc(num_objects, sizeof(Matrix));

        MPI_Recv(&threshold, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Get the available objects from the master process
        for (int i = 0; i < num_objects; i++) {
            char buffer[200000];
            int position = 0;
            MPI_Recv(buffer, 200000, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Unpack(buffer, 200000, &position, &object_matrixes[i], 1, MPI_Matrix, MPI_COMM_WORLD);
            object_matrixes[i].data = (int*)malloc(sizeof(int) * object_matrixes[i].size * object_matrixes[i].size);
            MPI_Unpack(buffer, 200000, &position, object_matrixes[i].data, object_matrixes[i].size * object_matrixes[i].size, MPI_INT, MPI_COMM_WORLD);
            printf("OBJECT %d MATRIX SIZE: %d\n", (i + 1), object_matrixes[i].size);
            //printMatrix(&object_matrixes[i]);

        }
    }

    // Start the timer
    double start_time = MPI_Wtime();

    if (my_rank == 0)
    {
        // As long as there are pictures to process
        while (sent_picture_index < num_pictures) {

            // This flag is set to 1 when all pictures have been processed
            int terminate_flag = 0;
            // Send terminate flag info to other MPI processes
            for (int i = 1; i < num_of_processes; i++)
                MPI_Send(&terminate_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            // Temporary buffer that holds availability signal info of other processes
            int buffer[2] = { 0 , 0 };

            // A small array that stores rank and availability signal of free worker
            int worker_info[2] = {0, 0};
            int availability = worker_info[0];
           
            // Wait until a process is available for work
            while (availability != 1) {
                MPI_Recv(worker_info, 2, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                availability = worker_info[0];
            }

            // Work flag is 0 for process that is assigned work
            // It is 1 for the processes that are not assigned work in current round
            int work_flag = 0;
            MPI_Send(&work_flag, 1, MPI_INT, worker_info[1], 0, MPI_COMM_WORLD);

            int picture_position = 0;
            char picture_buffer[1500000];
            printf("sending picture_matrix\n");
            printf("picture_matrix size before sending: %d\n", pictures[sent_picture_index].size);
            pictures[sent_picture_index].picture_id = sent_picture_index + 1;
            MPI_Pack(&(pictures[sent_picture_index]), 1, MPI_Matrix, picture_buffer, 1500000, &picture_position, MPI_COMM_WORLD);
            MPI_Pack(pictures[sent_picture_index].data, pictures[sent_picture_index].size* pictures[sent_picture_index].size, MPI_INT, picture_buffer, 1500000, &picture_position, MPI_COMM_WORLD);
            MPI_Send(picture_buffer, picture_position, MPI_PACKED, worker_info[1], tag, MPI_COMM_WORLD);
            //free(pictures[sent_picture_index].matrix.data);
            sent_picture_index++;            
            printf("sent picture_matrix\n");          

            // Receive signals sent by processes that haven't been assigned work in this round
            // Set their work flag to 1 and send it to them as well
            for (int rank = 1; rank < num_of_processes; rank++) 
            {
                work_flag = 1;
                if (rank != worker_info[1]) {
                    MPI_Recv(buffer, 2, MPI_INT, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(&work_flag, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                }
            }
        }
        
        // Terminate flag is set to 1 after all pictures have been processed and this
        // info is sent to all other processes
        int terminate_flag = 1;
        for (int i = 1; i < num_of_processes; i++)
            MPI_Send(&terminate_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    else
    {
        // As long as terminate_flag != 1, processes will keep on asking for more data
        // from process 0
        int terminate_flag = 0;
        while (terminate_flag != 1)
        {           
            MPI_Recv(&terminate_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // if there are pictures to process
            if (terminate_flag == 0)
            {
                // Send availability signal and rank info to process 0
                int worker_info[2];
                worker_info[0] = 1;
                worker_info[1] = my_rank;
                MPI_Send(worker_info, 2, MPI_INT, 0, 1, MPI_COMM_WORLD);

                // Receive flag info about whether work has been assigned or not
                int work_flag;
                MPI_Recv(&work_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Process the picture If work has been assigned
                if (work_flag != 1)
                {
                    MPI_Status status;
                    // Get a single picture matrix from the master process
                    char picture_buffer[1500000];
                    int position = 0;
                    Matrix* picture_matrix = (Matrix*)malloc(sizeof(Matrix));

                    MPI_Recv(picture_buffer, 1500000, MPI_PACKED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Unpack(picture_buffer, 1500000, &position, picture_matrix, 1, MPI_Matrix, MPI_COMM_WORLD);
                    picture_matrix->data = (int*)malloc(sizeof(int) * picture_matrix->size * picture_matrix->size);
                    MPI_Unpack(picture_buffer, 1500000, &position, picture_matrix->data, picture_matrix->size* picture_matrix->size, MPI_INT, MPI_COMM_WORLD);
                    printf("PICTURE MATRIX SIZE: %d\n", picture_matrix->size);
                    sent_picture_index++;

                    // The result array with a default value of -1 to understand if 3 objects were found
                    int result[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
                    // A shared parameter that the openMP uses 
                    int counter = 0;

                    // It will search the picture with findSubMatrix and save the information to the result array
                    for (int i = 0; i < num_objects; i++) {
                        int temp_result[2] = { -1, -1 };
                        // OpenMP is utilized in findSubMatrix function
                        findSubMatrix(picture_matrix, &object_matrixes[i], threshold, temp_result);
                        if (temp_result[0] != -1 && temp_result[1] != -1) {
                            result[counter * 3] = i + 1;
                            result[counter * 3 + 1] = temp_result[0];
                            result[counter * 3 + 2] = temp_result[1];
                            counter++;
                        }
                    }
                    for (int i = 0; i < 3; i++) {
                        printf("ID %d, (%d, %d)\n", result[i * 3], result[i * 3 + 1], result[i * 3 + 2]);
                    }
                    // Adding the picture id as the last element of the array
                    result[9] = picture_matrix->picture_id;
                    // Sending the array to process 0
                    MPI_Send(&result, 10, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
                }
            }
        }
    }
    // Stop the timer
    double end_time = MPI_Wtime();

    // Write results to output file
    if (my_rank == 0)
    {
        // Receive Data
        for (int i = 0; i < num_pictures; i++) 
        {          
            // Array of results from an MPI process, consists of object ids, object coordinates and a picture id
            int result_from_process[10];
            MPI_Recv(result_from_process, 10, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < 10; j++) {
                printf("result_from_process: %d\n", result_from_process[j]);
            }
            // Print the result to the output file
            printMatchingResults(result_from_process, output_file);
        }
        fclose(output_file);
        // Print the amount of time the program took
        printf("Execution time = %1.2f seconds\n", end_time - start_time);
    }

    // Closing the MPI
    MPI_Finalize();
    return 0;
    }

    void printMatrix(Matrix * matrix)
    {

        int row, columns;
        for (row = 0; row < matrix->size; row++)
        {
            for (columns = 0; columns < matrix->size; columns++)
            {
                printf("%d     ", matrix->data[row * matrix->size + columns]);
            }
            printf("\n");
        }
    }