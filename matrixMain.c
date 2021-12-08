#define CL_TARGET_OPENCL_VERSION 200

#include "clHelper.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
  size_t bytes = N * N * sizeof(double *);
  double nanoseconds = 0.0f;

  double *hA = (double *)malloc(bytes);
  double *hB = (double *)malloc(bytes);
  double *hC = (double *)malloc(bytes);

  printf("host arrays malloced\n");

  // initialize values
  initHost(hA, hB);

  // load kernel file
  size_t kernelSize;
  char *kernelSource = kernelFromFile(&kernelSize, "matrix.cl");

  // get platform & device
  cl_platform_id platformID = NULL;
  cl_device_id deviceID = NULL;
  getPlatformDevice(&platformID, &deviceID);

  // create context
  cl_context context;
  createContext(&context, &platformID, &deviceID);

  // create command queue
  cl_command_queue commandQueue;
  createQueue(&commandQueue, &context, &deviceID, 0, 1);

  // create buffers
  cl_mem dA, dB, dC;
  cl_double d_A;
  createBuffer(&dA, bytes, CL_MEM_READ_ONLY, &context);
  createBuffer(&dB, bytes, CL_MEM_READ_ONLY, &context);
  createBuffer(&dC, bytes, CL_MEM_READ_WRITE, &context);
  writeBuffer(dA, hA, &commandQueue);
  writeBuffer(dB, hB, &commandQueue);
  //--------------------------------------
  // writeBuffer(dC, hA, &commandQueue);

  // create program from kernel source
  cl_program program;
  createProgramFromSource(&program, &context, kernelSource, &kernelSize);

  // build program
  buildProgram(&program, &deviceID);

  // create kernel
  cl_kernel kernel;
  createKernel(&kernel, &program, "mult");

  // set arguments
  setArgs(&kernel, dA, dB, dC);

  // exec kernel
  cl_event done = NULL;
  execKernel(commandQueue, kernel, &done);

  readBuffer(dC, hC, &commandQueue);
  ret = clFinish(commandQueue);
  checkErr(ret, "finished queue");

  // profiling
  cl_ulong timeStart;
  cl_ulong timeEnd;
  clGetEventProfilingInfo(done, CL_PROFILING_COMMAND_START, sizeof(timeStart),
                          &timeStart, NULL);
  clGetEventProfilingInfo(done, CL_PROFILING_COMMAND_END, sizeof(timeEnd),
                          &timeEnd, NULL);
  nanoseconds = timeEnd - timeStart;

  gpuBench(hA, hB, hC, nanoseconds);

  ret = clFlush(commandQueue);

  ret = clReleaseEvent(done);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(dC);
  ret = clReleaseMemObject(dB);
  ret = clReleaseMemObject(dA);
  ret = clReleaseCommandQueue(commandQueue);
  ret = clReleaseContext(context);
  free(kernelSource);
  free(hA);
  free(hB);
  free(hC);
}