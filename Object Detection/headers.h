#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Matrix struct
// picture_id - A parameter to hold the id of the picture, in case the matrix belongs to a picture
// data - the matrix itself, size*size int array holding the numbers
// size - the length of the matrix, i.e rows size or columns size
typedef struct matrix {
    int picture_id;
    int* data;
    int size;
} Matrix;

void readMatrix(FILE* fp, Matrix* matrix);
double absDiff(int p, int o);
double findRelativeDifference(Matrix* pictureMatrix, Matrix* objectMatrix, int row, int col);
void printMatchingResults(int* results, FILE* output);
void findSubMatrix(Matrix* pictureMatrix, Matrix* objectMatrix, double threshold, int* result);