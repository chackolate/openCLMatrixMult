#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

// return a random double between min and max
double randDouble(double min, double max) {
  double range = max - min;
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

// initialize values
void initHost(double *hA, double *hB) {
  for (int i = 0; i < N * N; i++) {
    hA[i] = randDouble(-2, 2);
    hB[i] = randDouble(-2, 2);
  }
  printf("initialized host inputs\n");
}

// multiply matrices on CPU
void matrixMultiply(double *A, double *B, double *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float accumulator = 0.0f;
      for (int k = 0; k < N; k++) {
        accumulator += A[k * N + i] * B[j * N + k];
      }
      C[j * N + i] = accumulator;
    }
  }
}

// use matrix mult function to benchmark CPU performance
void cpuBench(double *A, double *B) {
  double *testC = (double *)malloc(N * N * sizeof(double));

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
      float accumulator = 0.0f;
      for (int k = 0; k < N; k++) {
        accumulator += A[k * N + i] * B[j * N + k];
      }
      if (testC[j * N + i] != accumulator) {
        printf("test [%d][%d] failed\n", i, j);
        break;
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

  double *hA = (double *)malloc(N * N * sizeof(double));
  double *hB = (double *)malloc(N * N * sizeof(double));
  double *hC = (double *)malloc(N * N * sizeof(double));

  printf("host arrays malloced\n");

  initHost(hA, hB);
  cpuBench(hA, hB);

  free(hA);
  free(hB);
  free(hC);
}