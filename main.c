#define CL_TARGET_OPENCL_VERSION 200

#include <math.h>
#include <opencl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

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
  printf("host floats\n");

  // device inputs
  cl_mem dA;
  cl_mem dB;
  // device output
  cl_mem dC;
  printf("device memory\n");

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
  printf("malloc host memory\n");

  // fill in host inputs
  int i;
  for (i = 0; i < N; i++) {
    hA[i] = sinf(i) * sinf(i);
    hB[i] = cosf(i) * cosf(i);
  }
  printf("fill in host inputs\n");

  size_t globalSize, localSize;
  cl_int err;

  // work items per work group
  localSize = 64;
  // total work items
  globalSize = ceil(N / (float)localSize) * localSize;

  // Platform ID
  err = clGetPlatformIDs(1, &platformID, NULL);
  checkError(err, "platform ID");
  printf("got platform id\n");
  // Device ID
  err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
  checkError(err, "device ID");
  printf("got device id\n");
  // context
  context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
  checkError(err, "creating context");
  printf("created context\n");

  // queue in context
  // cl_queue_properties *profiling = (void *)(long)CL_QUEUE_PROFILING_ENABLE;
  clCreateCommandQueueWithProperties(context, deviceID, NULL, &err);
  checkError(err, "creating queue");
  printf("created queue\n");

  // allocate device memory
  dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer A");
  dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer B");
  dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  checkError(err, "creating buffer C");
  printf("created device buffers\n");

  // read file
  FILE *filePointer;
  char *sourceStr;
  size_t sourceSize, programSize;

  filePointer = fopen("vectorAddition.cl", "rb");
  if (!filePointer) {
    printf("failed to load kernel from source");
    return 1;
  }
  fseek(filePointer, 0, SEEK_END);
  programSize = ftell(filePointer);
  rewind(filePointer);
  sourceStr = (char *)malloc(programSize + 1);
  sourceStr[programSize] = '\0';
  fread(sourceStr, sizeof(char), programSize, filePointer);
  fclose(filePointer);
  // program in context
  program = clCreateProgramWithSource(context, 1, (const char **)&sourceStr,
                                      &programSize, &err);
  checkError(err, "creating program");
  printf("created program from source\n");

  // Build executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkError(err, "building program");
  printf("built program\n");

  // kernel in program
  kernel = clCreateKernel(program, "vectorAdd", &err);
  checkError(err, "creating kernel");
  printf("created kernel in program\n");

  // set arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dA);
  checkError(err, "setting arg0");
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dB);
  checkError(err, "setting arg1");
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dC);
  checkError(err, "setting arg2");
  printf("set kernel arguments\n");

  // write data from host to device
  err = clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, bytes, hA, 0, NULL, NULL);
  checkError(err, "writing to device");
  err = clEnqueueWriteBuffer(queue, dB, 1, 0, bytes, hB, 0, NULL, NULL);
  checkError(err, "writing to device");

  printf("wrote from host to device\n");

  // exec kernel over range
  cl_event event;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                               0, NULL, NULL);
  printf("kernel queued\n");

  // wait for finish
  clFinish(queue);
  printf("finished\n");

  // read device to host
  clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, hC, 0, NULL, NULL);

  // Sum entire vector and divide result by n (should be 1)
  float sum = 0;
  for (i = 0; i < N; i++) {
    sum += hC[i];
    // printf("%f + %f = %f\n", hA[i], hB[i], hC[i]);
  }
  printf("Final result: %f\n", sum / N);

  // clean up
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseEvent(event);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(hA);
  free(hB);
  free(hC);
  return 0;
}