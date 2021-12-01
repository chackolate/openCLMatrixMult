#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

// void checkError(cl_int err, const char *operation) {
//   if (err != CL_SUCCESS) {
//     fprintf(stderr, "Error during operation: '%s': %d\n", operation, err);
//     exit(1);
//   }
// }

const char *getErrorString(cl_int error) {
  switch (error) {
  // run-time and JIT compiler errors
  case 0:
    return "CL_SUCCESS";
  case -1:
    return "CL_DEVICE_NOT_FOUND";
  case -2:
    return "CL_DEVICE_NOT_AVAILABLE";
  case -3:
    return "CL_COMPILER_NOT_AVAILABLE";
  case -4:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5:
    return "CL_OUT_OF_RESOURCES";
  case -6:
    return "CL_OUT_OF_HOST_MEMORY";
  case -7:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8:
    return "CL_MEM_COPY_OVERLAP";
  case -9:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case -10:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11:
    return "CL_BUILD_PROGRAM_FAILURE";
  case -12:
    return "CL_MAP_FAILURE";
  case -13:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case -16:
    return "CL_LINKER_NOT_AVAILABLE";
  case -17:
    return "CL_LINK_PROGRAM_FAILURE";
  case -18:
    return "CL_DEVICE_PARTITION_FAILED";
  case -19:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  // compile-time errors
  case -30:
    return "CL_INVALID_VALUE";
  case -31:
    return "CL_INVALID_DEVICE_TYPE";
  case -32:
    return "CL_INVALID_PLATFORM";
  case -33:
    return "CL_INVALID_DEVICE";
  case -34:
    return "CL_INVALID_CONTEXT";
  case -35:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case -36:
    return "CL_INVALID_COMMAND_QUEUE";
  case -37:
    return "CL_INVALID_HOST_PTR";
  case -38:
    return "CL_INVALID_MEM_OBJECT";
  case -39:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40:
    return "CL_INVALID_IMAGE_SIZE";
  case -41:
    return "CL_INVALID_SAMPLER";
  case -42:
    return "CL_INVALID_BINARY";
  case -43:
    return "CL_INVALID_BUILD_OPTIONS";
  case -44:
    return "CL_INVALID_PROGRAM";
  case -45:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46:
    return "CL_INVALID_KERNEL_NAME";
  case -47:
    return "CL_INVALID_KERNEL_DEFINITION";
  case -48:
    return "CL_INVALID_KERNEL";
  case -49:
    return "CL_INVALID_ARG_INDEX";
  case -50:
    return "CL_INVALID_ARG_VALUE";
  case -51:
    return "CL_INVALID_ARG_SIZE";
  case -52:
    return "CL_INVALID_KERNEL_ARGS";
  case -53:
    return "CL_INVALID_WORK_DIMENSION";
  case -54:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case -55:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case -56:
    return "CL_INVALID_GLOBAL_OFFSET";
  case -57:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case -58:
    return "CL_INVALID_EVENT";
  case -59:
    return "CL_INVALID_OPERATION";
  case -60:
    return "CL_INVALID_GL_OBJECT";
  case -61:
    return "CL_INVALID_BUFFER_SIZE";
  case -62:
    return "CL_INVALID_MIP_LEVEL";
  case -63:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64:
    return "CL_INVALID_PROPERTY";
  case -65:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66:
    return "CL_INVALID_COMPILER_OPTIONS";
  case -67:
    return "CL_INVALID_LINKER_OPTIONS";
  case -68:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";

  // extension errors
  case -1000:
    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001:
    return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002:
    return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003:
    return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004:
    return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005:
    return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default:
    return "Unknown OpenCL error";
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
  localSize = 16;
  // total work items
  globalSize = ceil(N / (float)localSize) * localSize;

  // Platform ID
  err = clGetPlatformIDs(1, &platformID, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("got platform id\n");
  // Device ID
  err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("got device id\n");
  // context
  context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("created context\n");

  // queue in context
  clCreateCommandQueueWithProperties(context, deviceID, 0, &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("created queue\n");

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
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("created program from source\n");

  // Build executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    // Allocate memory for the log
    char *memLog = (char *)malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, log_size,
                          memLog, NULL);

    // Print the log
    printf("%s\n", memLog);
  }
  printf("built program\n");

  // kernel in program
  kernel = clCreateKernel(program, "vectorAdd", &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("created kernel in program\n");

  // allocate device memory
  dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("created device buffers%d\n", err);

  // write data from host to device
  err = clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, bytes, hA, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  err = clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, bytes, hB, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("wrote from host to device\n");

  // set arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("set kernel arguments\n");

  // exec kernel over range
  // cl_event event;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                               0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }
  printf("kernel queued\n");

  // wait for finish
  clFinish(queue);
  printf("finished\n");

  // read device to host
  err = clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, hC, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("%s\n", getErrorString(err));
  }

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
  // clReleaseEvent(event);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(hA);
  free(hB);
  free(hC);
  // free(err);
  return 0;
}