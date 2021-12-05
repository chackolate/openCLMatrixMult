#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 27) // length of vector 4096
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

void checkErr(cl_int error, char *success) {
  if (error != CL_SUCCESS) {
    printf("%s", getErrorString(error));
  } else {
    printf("%s\n", success);
  }
}

// return a random double between 0 and 1
double randDouble(double min, double max) {
  double range = max - min;
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

// initialize values
void initHost(double *hA, double *hB) {
  for (int i = 0; i < N; i++) {
    hA[i] = randDouble(-1.0, 1.0);
    hB[i] = randDouble(-1.0, 1.0);
  }
  printf("initialized host inputs\n");
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
                 cl_device_id *deviceID) {
  cl_int err;
  *commandQueue =
      clCreateCommandQueueWithProperties(*context, *deviceID, 0, &err);
  checkErr(err, "created command queue");
}

// create memory buffers
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

// set kernel arguments
void setArgs(cl_kernel *kernel, cl_mem dA, cl_mem dB, cl_mem dC) {
  cl_int err;
  err = clSetKernelArg(*kernel, 0, sizeof(cl_mem), (void *)&dA);
  checkErr(err, "set arg 1");
  err = clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void *)&dB);
  checkErr(err, "set arg 2");
  err = clSetKernelArg(*kernel, 2, sizeof(cl_mem), (void *)&dC);
  checkErr(err, "set arg 3");
}

// execute kernel
void execKernel(size_t globalItemSize, size_t localItemSize,
                cl_command_queue *commandQueue, cl_kernel *kernel) {
  cl_int err;
  err = clEnqueueNDRangeKernel(*commandQueue, *kernel, 1, NULL, &globalItemSize,
                               &localItemSize, 0, NULL, NULL);
  checkErr(err, "kernel executed");
}

// read device vectors back to host
void readDeviceToHost(cl_mem source, double *dest,
                      cl_command_queue *commandQueue) {
  cl_int err;
  err = clEnqueueReadBuffer(*commandQueue, source, CL_TRUE, 0,
                            N * sizeof(double), (void *)dest, 0, NULL, NULL);
}

int main(int argc, char *argv[]) {
  time_t t;
  srand((unsigned)time(&t));
  cl_int ret;
  // host arrays
  int bytes = N * sizeof(double);
  double *hA = (double *)malloc(bytes);
  double *hB = (double *)malloc(bytes);
  double *hC = (double *)malloc(bytes);

  // initialize values
  initHost(hA, hB);

  // load kernel from file
  char *kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
  size_t kernelSize;
  char filename[] = "vectorAddition.cl";
  kernelFromFile(&kernelSize, kernelSource, filename);

  // get platform & device
  cl_platform_id platformID = NULL;
  cl_device_id deviceID = NULL;
  getPlatformDevice(&platformID, &deviceID);

  // create context
  cl_context context;
  createContext(&context, &deviceID);

  // create command queue
  cl_command_queue commandQueue;
  createQueue(&commandQueue, &context, &deviceID);

  // memory buffers
  cl_mem dA, dB, dC;
  createBuffer(&dA, CL_MEM_READ_ONLY, &context);
  createBuffer(&dB, CL_MEM_READ_ONLY, &context);
  createBuffer(&dC, CL_MEM_WRITE_ONLY, &context);

  // copy host vectors to device
  cpyHostToDevice(dA, hA, &commandQueue);
  cpyHostToDevice(dB, hB, &commandQueue);

  // create program from kernel source
  cl_program program;
  createProgramFromSource(&program, &context, kernelSource, &kernelSize);

  // build program
  buildProgram(&program, &deviceID);

  // create kernel
  cl_kernel kernel;
  createKernel(&kernel, &program, "vectorAdd");

  // set kernel arguments
  setArgs(&kernel, dA, dB, dC);

  // execute kernel
  size_t globalItemSize = N;  // N global items
  size_t localItemSize = 256; // within each global are 256 local items (max)
  execKernel(globalItemSize, localItemSize, &commandQueue, &kernel);

  // read device vectors to host
  readDeviceToHost(dC, hC, &commandQueue);

  // verify answer
  int i;
  for (i = 0; i < N; i++) {
    if (hC[i] != (hA[i] + hB[i])) {
      printf("A%f + B%f = C%f\n", hA[i], hB[i], hC[i]);
      break;
    }
  }
  if (i == N) {
    printf("All values correct. Random results: \n");
    for (int j = 0; j < 5; j++) {
      int element = rand() % (N + 1);
      printf("%d: %f + %f = %f\n", element, hA[element], hB[element],
             hC[element]);
    }
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