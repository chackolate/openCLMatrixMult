#define CL_TARGET_OPENCL_VERSION 200

#include <math.h>
#include <opencl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

const char *kernelSource =
    "\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n"
    "__kernel void vecAdd(  __global double *a,                       \n"
    "                       __global double *b,                       \n"
    "                       __global double *c)                       \n"
    "{                                                               \n"
    "    //Get our global thread ID                                  \n"
    "    int id = get_global_id(0);                                  \n"
    "                                                                \n"
    "    //Make sure we do not go out of bounds                      \n"
    "    if (id < N)                                                 \n"
    "        c[id] = a[id] + b[id];                                  \n"
    "}                                                               \n"
    "\n";

void checkError(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error during operation: '%s': %d\n", operation, err);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  // host inputs
  float *hA;
  float *hB;
  // host output
  float *hC;

  // device inputs
  cl_mem dA;
  cl_mem dB;
  // device output
  cl_mem dC;

  cl_platform_id platformID;
  cl_device_id deviceID;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  size_t bytes = N * sizeof(float);

  // allocate host memory
  hA = (float *)malloc(bytes);
  hB = (float *)malloc(bytes);
  hC = (float *)malloc(bytes);

  // fill in host inputs
  int i;
  for (i = 0; i < N; i++) {
    hA[i] = sinf(i) * sinf(i);
    hB[i] = cosf(i) * cosf(i);
  }

  size_t globalSize, localSize;
  cl_int err;

  // work items per work group
  localSize = 64;

  // total work items
  globalSize = ceil(N / (float)localSize) * localSize;

  // Platform ID
  err = clGetPlatformIDs(1, &platformID, NULL);
  checkError(err, "platform ID");

  // Device ID
  err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
  checkError(err, "device ID");

  // context
  context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
  checkError(err, "creating context");

  // queue in context
  queue = clCreateCommandQueueWithProperties(context, deviceID, 0, &err);
  checkError(err, "creating queue");

  // allocate device memory
  dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer A");
  dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer B");
  dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer C");

  // program in context
  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                      NULL, &err);
  checkError(err, "creating program");

  // Build executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkError(err, "building program");

  // kernel in program
  kernel = clCreateKernel(program, "vecAdd", &err);
  checkError(err, "creating kernel");

  // write data from host to device
  err = clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, bytes, hA, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, bytes, hB, 0, NULL, NULL);

  // set arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);

  // exec kernel over range
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                               0, NULL, NULL);

  // wait for finish
  clFinish(queue);

  // read device to host
  clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, hC, 0, NULL, NULL);

  // Sum entire vector and divide result by n (should be 1)
  float sum = 0;
  for (i = 0; i < N; i++) {
    sum += hC[i];
  }
  printf("Final result: %f\n", sum / N);

  // clean up
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(hA);
  free(hB);
  free(hC);
  return 0;
}