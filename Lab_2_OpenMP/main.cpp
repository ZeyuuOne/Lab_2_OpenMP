#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int m, n;
float* A;
float* L, * U;
int i, j, k;
double start, end;
FILE* inFile, * outFile;
int numThreads = 12;

int main(int argc, char* argv[])
{
    inFile = fopen("LU.in", "r");
    fscanf(inFile, "%d %d", &m, &n);
    A = (float*)malloc(sizeof(float) * m * m);
    for (i = 0; i < m; i++) for (j = 0; j < m; j++) fscanf(inFile, "%f", A + i * m + j);
    fclose(inFile);
    L = (float*)malloc(sizeof(float) * m * m);
    U = (float*)malloc(sizeof(float) * m * m);

    start = omp_get_wtime();

#pragma omp parallel num_threads(numThreads) private(i, j, k)
    {
        for (k = 0; k < m; k++) {
#pragma omp for
            for (i = k + 1; i < m; i++) A[i * m + k] = A[i * m + k] / A[k * m + k];
#pragma omp for
            for (i = k + 1; i < m; i++) for (j = k + 1; j < m; j++) {
                A[i * m + j] = A[i * m + j] - A[i * m + k] * A[k * m + j];
            }
        }
#pragma omp for
        for (i = 0; i < m; i++) {
            L[i * m + i] = 1;
            for (j = 0; j < m; j++) {
                if (i > j) L[i * m + j] = A[i * m + j];
                else U[i * m + j] = A[i * m + j];
            }
        }
    }

    end = omp_get_wtime();

    outFile = fopen("LU.out", "w");
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++) fprintf(outFile, "%f\t", L[i * m + j]);
        fprintf(outFile ,"\n");
    }
    fprintf(outFile, "\n");
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++) fprintf(outFile, "%f\t", U[i * m + j]);
        fprintf(outFile, "\n");
    }
    fclose(outFile);

    printf("M: %d, Thread Num: %d, Time: %fs\n", m, numThreads, end - start);
    return 0;
}