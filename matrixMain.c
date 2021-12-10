#define CL_TARGET_OPENCL_VERSION 200

#include "clHelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// multiply matrices on CPU
void matrixMultiply(float *A, float *B, float *C) {
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

// use matrix mult function to benchmark CPU vs GPU performance & results
void cpuBench(float *A, float *B, float *C) {
  float *testC = (float *)malloc(N * N * sizeof(float));

  double error;

  printf("multiplying on CPU...\n");
  clock_t start = clock();
  matrixMultiply(A, B, testC);
  clock_t end = clock();
  float cpuTime = (float)(end - start) / CLOCKS_PER_SEC;
  printf("finished multiplying. Time taken: %3f seconds\n", cpuTime);

  printf("verification...\n");
  for (int i = 0; i < N * N; i++) {
    error += abs((C[i] - testC[i]) / testC[i]) * 100; // percent error
  }
  error = error / (N * N);
  printf("%03f%% error\n", error);
  free(testC);
}

void main(int argc, char *argv[]) {

  // setup randomization & error return
  time_t t;
  srand((unsigned)time(&t));
  size_t bytes = N * N * sizeof(float *);

  float *hA = (float *)malloc(bytes);
  float *hB = (float *)malloc(bytes);
  float *hC = (float *)malloc(bytes);

  runKernel(hA, hB, hC, "matrix.cl", "mult");

  cpuBench(hA, hB, hC);

  free(hA);
  free(hB);
  free(hC);
}