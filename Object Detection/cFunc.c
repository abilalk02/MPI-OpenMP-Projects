#include "headers.h"
#include <omp.h>

// Function to read a matrix from a file
void readMatrix(FILE* fp, Matrix* matrix) {
    for (int i = 0; i < matrix->size; i++) {
        for (int j = 0; j < matrix->size; j++) {
            fscanf(fp, "%d", &matrix->data[i* matrix->size+j]);
        }
    }
}

// Prints the results to an output file, self-explanatory
void printMatchingResults(int* results, FILE* output) {
    // The results int array holds 10 integers, 3 object ids, 6 coordinates, 1 picture id
    // I'm checking if the last coordinate still has its default value, if it has - it means only 0-2 objects were found
    if (results[8] != -1) {
        fprintf(output, "Picture %d: found Objects: ", results[9]);
        for (int i = 0; i < 3; i++) {
            fprintf(output, "%d Position%d(%d,%d) ; ", results[i*3], i+1, results[i*3+1], results[i*3+2]);
        }
        fprintf(output, "\n");
    } else {
        fprintf(output, "Picture %d: No three different Objects were found\n", results[9]);
    }
}

// Finding the approximation for the given objectMatrix, for every pictureMatrix it checks the relative difference
// and if it is lower than the given threshold - it replaces the matchValue as the best current option
// Afterwards it sets the coordinates of the objectMatrix in the result array
void findSubMatrix(Matrix* pictureMatrix, Matrix* objectMatrix, double threshold, int* result) {
    int pictureSize = pictureMatrix->size;
    int objectSize = objectMatrix->size;
    double matching_value = 10000;

    // Use parallel for loop
#   pragma omp parallel for num_threads(2)
    for (int i = 0; i <= (pictureSize - objectSize); i++) {
        for (int j = 0; j <= (pictureSize - objectSize); j++) {
            double matchValue = findRelativeDifference(pictureMatrix, objectMatrix, i, j);
            if (matchValue < threshold && matchValue < matching_value) {
                matching_value = matchValue;
                result[0] = i;
                result[1] = j;
                printf("found match! %f at %d and %d\n", matching_value, i, j);
            }
        }
    }
}

// Finds the relative difference of a given area (determined by row and col parameters)
// Uses the absDiff function which returns the absolute value of the difference
double findRelativeDifference(Matrix* pictureMatrix, Matrix* objectMatrix, int row, int col) {
    int i, j;
    double total_diff = 0.0;
    int pictureSize = pictureMatrix->size;
    int objectSize = objectMatrix->size;

    for (i = 0; i < objectSize; i++) {
        for (j = 0; j < objectSize; j++) {
            int k = row + i;
            int m = col + j;
            total_diff += absDiff(pictureMatrix->data[k * pictureSize + m], objectMatrix->data[i * objectSize + j]);
        }
    }
    return (total_diff / (objectSize * objectSize));
}

// Returns the absolute value of the difference, as described in the task doc
double absDiff(int p, int o) {
    double r =((p - o)/(double)p);
    if (r < 0) r = r * -1;
    return r;
}
