#define CL_TARGET_OPENCL_VERSION 200

#include "clHelper.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// multiply matrices on CPU
void matrixMultiply(const double *A, const double *B, double *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double accumulator = 0.0;
      for (int k = 0; k < N; k++) {
        accumulator += A[k * N + i] * B[j * N + k];
      }
      C[j * N + i] = accumulator;
    }
  }
}

double rms(double *A, double *B, int n) {
  int square = 0;
  double mean = 0.0, root = 0.0;

  for (int i = 0; i < n; i++) {
    square += pow(A[i], 2);
  }
  mean = (square / (float)(n));
  root = sqrt(mean);
  return root;
}

int checkEq(const double *A, const double *B) {
  for (int i = 0; i < N * N; i++) {
    if (fabs(A[i] - B[i]) > 0.00001) {
      return 0;
    }
  }
  return 1;
}

// use matrix mult function to benchmark CPU vs GPU performance & results
void cpuBench(const double *A, const double *B, const double *C) {
  double *testC = (double *)malloc(N * N * sizeof(double));

  printf("multiplying on CPU...\n");
  clock_t start = clock();
  matrixMultiply(A, B, testC);
  clock_t end = clock();
  double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;
  printf("finished multiplying. Time taken: %3f seconds\n", cpuTime);

  // FILE *outfile;
  // int i;

  printf("verification...\n");
  if (checkEq(C, testC)) {
    printf("All values agree");
  } else {
    printf("discrepancy found");
  }
  free(testC);
}

void main(int argc, char *argv[]) {

  // setup randomization & error return
  time_t t;
  srand((unsigned)time(&t));
  size_t bytes = N * N * sizeof(double *);

  // host matrices
  double *hA = (double *)malloc(bytes);
  double *hB = (double *)malloc(bytes);
  double *hC = (double *)malloc(bytes);

  runKernel(hA, hB, hC, "matrix.cl", "mult");
  runKernel(hA, hB, hC, "matrix.cl", "mult2");

  cpuBench(hA, hB, hC);

  free(hA);
  free(hB);
  free(hC);
}