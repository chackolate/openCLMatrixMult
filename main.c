#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
// #include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512 // length of vector
#define MAX_SOURCE_SIZE (0x100000)

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
  // host arrays
  float *hA = (float *)malloc(sizeof(float) * N);
  float *hB = (float *)malloc(sizeof(float) * N);
  float *hC = (float *)malloc(sizeof(float) * N);

  // initialize values
  int i = 0;
  for (i = 0; i < N; ++i) {
    hA[i] = i + 1;
    hB[i] = (i + 1) + 2;
  }

  // load kernel from file
  FILE *kernelFile;
  char *kernelSource;
  size_t kernelSize;

  kernelFile = fopen("vecAddKernel.cl", "rb");
  if (!kernelFile) {
    fprintf(stderr, "kernel file not found.\n");
    exit(-1);
  }
  kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
  kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
  fclose(kernelFile);

  // get platform & device
  cl_platform_id platformID = NULL;
  cl_device_id deviceID = NULL;
  cl_uint retNumDevices;
  cl_uint retNumPlatforms;
  cl_int ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
  ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID,
                       &retNumDevices);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // create context
  cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // create command queue
  cl_command_queue commandQueue =
      clCreateCommandQueueWithProperties(context, deviceID, 0, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // memory buffers
  cl_mem dA =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }
  cl_mem dB =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }
  cl_mem dC =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // copy host vectors to device
  ret = clEnqueueWriteBuffer(commandQueue, dA, CL_TRUE, 0, N * sizeof(float),
                             hA, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }
  ret = clEnqueueWriteBuffer(commandQueue, dB, CL_TRUE, 0, N * sizeof(float),
                             hB, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // create program from kernel source
  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                (const size_t *)&kernelSize, &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // build program
  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // create kernel
  cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // set kernel arguments
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dA);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dB);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dC);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // execute kernel
  size_t globalItemSize = N;     // 1024
  size_t localItemSize = N / 16; // 64
  ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize,
                               &localItemSize, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // read device vectors to host
  ret = clEnqueueReadBuffer(commandQueue, dC, CL_TRUE, 0, N * sizeof(float), hC,
                            0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    getErrorString(ret);
  }

  // verify answer
  for (i = 0; i < N; i++) {
    if (hC[i] != (hA[i] + hB[i])) {
      printf("A%f + B%f = C%f", hA[i], hB[i], hC[i]);
      break;
    }
  }
  if (i == N) {
    printf("appears to be working fine");
  }

  ret = clFlush(commandQueue);
  ret = clFinish(commandQueue);
  ret = clReleaseCommandQueue(commandQueue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(dA);
  ret = clReleaseMemObject(dB);
  ret = clReleaseMemObject(dC);
  ret = clReleaseContext(context);
  free(hA);
  free(hB);
  free(hC);

  return 0;
}