#include "clHelper.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

// simple function to get output error string from an error int
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

// real function to check error message and print success or fail
void checkErr(cl_int error, char *success) {
  if (error != CL_SUCCESS) {
    printf("%s\n", getErrorString(error));
  } else {
    printf("%s\n", success);
  }
}

// random double between min and max
double randDouble(double min, double max) {
  double range = max - min;
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

// initialize host inputs
void initHost(double *hA, double *hB) {
  for (int i = 0; i < N * N; i++) {
    hA[i] = randDouble(-2, 2);
    hB[i] = randDouble(-2, 2);
  }
}

// load kernel from file
void kernelFromFile(size_t *kernelSize, char *kernelSource, char *filename) {
  FILE *kernelFile = fopen(filename, "rb");
  if (!kernelFile) {
    fprintf(stderr, "kernel file not found.\n");
    exit(-1);
  }
  *kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
  fclose(kernelFile);
  printf("loaded kernel from file\n");
}

// get platform & device
void getPlatformDevice(cl_platform_id *platformID, cl_device_id *deviceID) {
  cl_uint retNumDevices;
  cl_uint retNumPlatforms;
  cl_int err = clGetPlatformIDs(1, platformID, &retNumPlatforms);
  checkErr(err, "got platform");
  err = clGetDeviceIDs(*platformID, CL_DEVICE_TYPE_GPU, 1, deviceID,
                       &retNumDevices);
  checkErr(err, "got device");
}

// create context
void createContext(cl_context *context, cl_device_id *deviceID) {
  cl_int err;
  *context = clCreateContext(NULL, 1, deviceID, NULL, NULL, &err);
  checkErr(err, "created context");
}

// create command queue
void createQueue(cl_command_queue *commandQueue, cl_context *context,
                 cl_device_id *deviceID, int outOfOrder, int profiling) {
  cl_int err;
  cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, 0, 0};
  if (outOfOrder) {
    props[1] |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (profiling) {
    props[1] |= CL_QUEUE_PROFILING_ENABLE;
  }
  *commandQueue =
      clCreateCommandQueueWithProperties(*context, *deviceID, props, &err);
  checkErr(err, "created command queue");
}

void createBuffer(cl_mem *deviceBuffer, int direction, cl_context *context) {
  cl_int err;
  *deviceBuffer = clCreateBuffer(
      *context, direction, N * sizeof(double), NULL,
      &err); // for flags: direction will be 1(output) or 2(input).
  checkErr(err, "created buffer");
}

// copy host vectors to device
void cpyHostToDevice(cl_mem dest, double *source,
                     cl_command_queue *commandQueue) {
  cl_int err;
  err = clEnqueueWriteBuffer(*commandQueue, dest, CL_TRUE, 0,
                             N * sizeof(double), source, 0, NULL, NULL);
  checkErr(err, "copied host to device");
}

// create program from kernel source
void createProgramFromSource(cl_program *program, cl_context *context,
                             const char *kernelSource, size_t *kernelSize) {
  cl_int err;
  *program =
      clCreateProgramWithSource(*context, 1, &kernelSource, kernelSize, &err);
  checkErr(err, "created program from source");
}

// build program
void buildProgram(cl_program *program, cl_device_id *deviceID) {
  cl_int err;
  err = clBuildProgram(*program, 1, deviceID, NULL, NULL, NULL);
  checkErr(err, "built program");
}

// create kernel
void createKernel(cl_kernel *kernel, cl_program *program, char *funcName) {
  cl_int err;
  *kernel = clCreateKernel(*program, funcName, &err);
  checkErr(err, "created kernel");
}

void setArgs(cl_kernel *kernel, cl_mem dA, cl_mem dB, cl_mem dC) {
  cl_int err;
  int n = N;
  int *nP = &n;
  err = clSetKernelArg(*kernel, 0, sizeof(n), (void *)&nP);
  checkErr(err, "set arg 0");
  err = clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void *)&dA);
  checkErr(err, "set arg 1");
  err = clSetKernelArg(*kernel, 2, sizeof(cl_mem), (void *)&dB);
  checkErr(err, "set arg 2");
  err = clSetKernelArg(*kernel, 3, sizeof(cl_mem), (void *)&dC);
  checkErr(err, "set arg 3");
}

void execKernel(cl_device_id deviceID, cl_command_queue *commandQueue,
                cl_kernel *kernel, cl_event *event) {
  size_t localWorkSize;
  cl_int err;
  err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                        (void *)&localWorkSize, NULL);
  printf("using device max work group size of %d\n", localWorkSize);
  checkErr(err, "work size set");
  size_t globalWorkSize = ceil((N * N) / (float)localWorkSize) *
                          localWorkSize; // N*N global items to calculate
  err = clEnqueueNDRangeKernel(*commandQueue, *kernel, 1, NULL, &globalWorkSize,
                               &localWorkSize, 0, NULL, event);
  checkErr(err, "kernel executed");
  err = clWaitForEvents(1, event);
  checkErr(err, "finished execution");
}

// read device vectors back to host
void readDeviceToHost(cl_mem source, double *dest,
                      cl_command_queue *commandQueue) {
  cl_int err;
  err = clEnqueueReadBuffer(*commandQueue, source, CL_TRUE, 0,
                            N * sizeof(double), (void *)dest, 0, NULL, NULL);
  checkErr(err, "read device to host");
}

void gpuBench(double *A, double *B, double *C, double nanoseconds) {
  printf("verification...\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float accumulator = 0.0;
      for (int k = 0; k < N; k++) {
        accumulator += A[k * N + i] * B[j * N + k];
        if (C[j * N + i] != accumulator) {
          // printf("%f * %f != %f\n", A[k * N + i], B[j * N + k], accumulator);
          printf("fail\n");
          break;
        }
      }
    }
  }
  printf("GPU values correct. Time taken: %0.3f milliseconds\n",
         nanoseconds / 1000000.0);
}