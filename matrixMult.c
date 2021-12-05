#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

// return a random double between 0 and 1
double randDouble(double min, double max) {
  double range = max - min;
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

// initialize values
void initHost(double *hA[N], double *hB[N]) {
  printf("function\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      hA[i][j] = 0.0;
      hB[i][j] = 0.0;
      // hA[i][j] = randDouble(-1.0, 1.0);
      // hB[i][j] = randDouble(-1.0, 1.0);
    }
  }
  printf("initialized host inputs\n");
}

// multiply matrices on CPU
void matrixMultiply(double *A[N], double *B[N], double *C[N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// use matrix mult function to benchmark CPU performance
void cpuBench(double **A, double **B) {
  double **testC = (double **)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    testC[i] = (double *)malloc(N * sizeof(double));
  }

  printf("multiplying on CPU...\n");
  clock_t start = clock();
  matrixMultiply(A, B, testC);
  clock_t end = clock();
  double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;
  printf("finished multiplying. Time taken: %0.3f seconds\n", cpuTime);

  printf("verification...\n");
  start = clock();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        if (testC[i][j] != A[i][k] * B[k][j]) {
          printf("A[%d][%d] %f * B[%d][%d] %f = C[%d][%d] %f", i, k, A[i][k], k,
                 j, B[k][j], i, j, testC[i][j]);
        }
      }
    }
  }
  end = clock();
  double verTime = (double)(end - start) / CLOCKS_PER_SEC;
  printf("CPU values correct. Time taken: %0.3f seconds\n", verTime);
  free(testC);
}

void main(int argc, char *argv[]) {

  // setup randomization & error return
  time_t t;
  srand((unsigned)time(&t));
  cl_int ret;

  printf("mallocing host inputs\n");
  double **hA = (double **)malloc(N * sizeof(double));
  double **hB = (double **)malloc(N * sizeof(double));
  double **hC = (double **)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    hA[i] = (double *)malloc(N * sizeof(double));
    hB[i] = (double *)malloc(N * sizeof(double));
    hC[i] = (double *)malloc(N * sizeof(double));
  }
  printf("success\n");

  initHost(hA, hB);
  cpuBench(hA, hB);

  free(hA);
  free(hB);
  free(hC);
}