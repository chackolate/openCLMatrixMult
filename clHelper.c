#include "clHelper.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

//-----------------host helper stuff-----------------
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
    exit(0);
  } else {
    if (VERBOSE) {
      printf("%s\n", success);
    } else {
      printf(".");
    }
  }
}

// random double between min and max
double randdouble(double min, double max) {
  return (((double)rand() * (max - min)) / ((double)RAND_MAX + min));
}

// initialize host inputs
void initHost(double *hA, double *hB) {
  for (int i = 0; i < N * N; i++) {
    hA[i] = (randdouble(-10.0, 10.0));
    hB[i] = randdouble(-10.0, 10.0);
  }
}

//-----------------configure environment-----------------
// get platform & device
void getPlatformDevice(cl_platform_id *platformID, cl_device_id *deviceID) {
  cl_uint retNumDevices;
  cl_uint retNumPlatforms;
  cl_int err = clGetPlatformIDs(1, platformID, &retNumPlatforms);
  checkErr(err, "got platform");
  err =
      clGetDeviceIDs(*platformID, CL_DEVICE_TYPE_GPU, 0, NULL, &retNumDevices);
  err = clGetDeviceIDs(*platformID, CL_DEVICE_TYPE_GPU, retNumDevices, deviceID,
                       NULL);
  checkErr(err, "got device");
}

// create context
void createContext(cl_context *context, cl_platform_id *platformID,
                   cl_device_id *deviceID) {
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)*platformID, 0};
  cl_int err;
  *context = clCreateContext(props, 1, deviceID, NULL, NULL, &err);
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
//------------------------------------------------------------
// load kernel from file
char *kernelFromFile(size_t *kernelSize, char *filename) {
  FILE *kernelFile = fopen(filename, "rb");
  if (!kernelFile) {
    fprintf(stderr, "kernel file not found.\n");
    exit(-1);
  }
  // get size
  fseek(kernelFile, 0, SEEK_END);
  long size = ftell(kernelFile);
  rewind(kernelFile);

  // read string
  char *source = (char *)malloc((size + 1) * sizeof(char));
  fread(source, 1, size * sizeof(char), kernelFile);
  fclose(kernelFile);
  printf("loaded file %s\n", filename);

  *kernelSize = size;
  return source;
}

// compile program from source
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
  // Check for compilation errors
  // size_t logSize;
  // err = clGetProgramBuildInfo(*program, *deviceID, CL_PROGRAM_BUILD_LOG, 0,
  //                             NULL, &logSize);
  // checkErr(err, "build passed");
  // char *messages = (char *)malloc((1 + logSize) * sizeof(char));
  // err = clGetProgramBuildInfo(*program, *deviceID, CL_PROGRAM_BUILD_LOG,
  //                             logSize, messages, NULL);
  // checkErr(err, "build passed");
  // messages[logSize] = '\0';
  // if (logSize > 10) {
  //   printf("## Compiler message: %s\n", messages);
  // }
  // free(messages);
}

void createBuffer(cl_mem *deviceBuffer, size_t size, int direction,
                  cl_context *context) {
  cl_int err;
  *deviceBuffer = clCreateBuffer(
      *context, direction, size, NULL,
      &err); // for flags: direction will be 1(output) or 2(input).
  checkErr(err, "created buffer");
}

// copy host to device
void writeBuffer(cl_mem dest, double *source, cl_command_queue *commandQueue) {
  cl_int err;
  err = clEnqueueWriteBuffer(*commandQueue, dest, CL_TRUE, 0,
                             N * N * sizeof(*source), source, 0, NULL, NULL);
  checkErr(err, "copied host to device");
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
  err = clSetKernelArg(*kernel, 0, sizeof(int), (void *)&n);
  checkErr(err, "set arg 0");
  err = clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void *)&dA);
  checkErr(err, "set arg 1");
  err = clSetKernelArg(*kernel, 2, sizeof(cl_mem), (void *)&dB);
  checkErr(err, "set arg 2");
  err = clSetKernelArg(*kernel, 3, sizeof(cl_mem), (void *)&dC);
  checkErr(err, "set arg 3");
}

void execKernel(cl_command_queue commandQueue, cl_kernel kernel,
                cl_event *event) {
  const int tile = 16; // solve the matrix in groups of 256
  const size_t local[2] = {tile, tile};
  const size_t global[2] = {N, N};
  cl_int err;
  // err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
  // sizeof(size_t),
  //                       (void *)&localWorkSize, NULL);
  // printf("using device max work group size of %d\n", localWorkSize);
  // size_t globalWorkSize = N * N; // N*N global items to calculate
  err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global, local, 0,
                               NULL, event);
  checkErr(err, "kernel executed");
  err = clWaitForEvents(1, event);
  checkErr(err, "finished execution");
}

// read device vectors back to host
void readBuffer(cl_mem source, double *dest, cl_command_queue *commandQueue) {
  cl_int err;
  err = clEnqueueReadBuffer(*commandQueue, source, CL_TRUE, 0,
                            N * N * sizeof(*dest), (void *)dest, 0, NULL, NULL);
  checkErr(err, "read device to host");
}

void timeProf(double *nanoseconds, cl_event done) {
  cl_ulong timeStart;
  cl_ulong timeEnd;
  clGetEventProfilingInfo(done, CL_PROFILING_COMMAND_START, sizeof(timeStart),
                          &timeStart, NULL);
  clGetEventProfilingInfo(done, CL_PROFILING_COMMAND_END, sizeof(timeEnd),
                          &timeEnd, NULL);
  *nanoseconds = timeEnd - timeStart;
}

void runKernel(double *hA, double *hB, double *hC, char *filename, char *func) {
  size_t bytes = N * N * sizeof(double *);
  double nanoseconds = 0.0f;
  cl_int ret;

  // initialize values
  initHost(hA, hB);

  // load kernel file
  size_t kernelSize;
  char *kernelSource = kernelFromFile(&kernelSize, filename);

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
  createBuffer(&dA, bytes, CL_MEM_READ_ONLY, &context);
  createBuffer(&dB, bytes, CL_MEM_READ_ONLY, &context);
  createBuffer(&dC, bytes, CL_MEM_READ_WRITE, &context);
  writeBuffer(dA, hA, &commandQueue);
  writeBuffer(dB, hB, &commandQueue);
  writeBuffer(dC, hA, &commandQueue); // clear any previous results in output

  // create program from kernel source
  cl_program program;
  createProgramFromSource(&program, &context, kernelSource, &kernelSize);

  // build program
  buildProgram(&program, &deviceID);

  // create kernel
  cl_kernel kernel;
  createKernel(&kernel, &program, func);

  // set arguments
  setArgs(&kernel, dA, dB, dC);

  // exec kernel
  cl_event done = NULL;
  execKernel(commandQueue, kernel, &done);

  readBuffer(dC, hC, &commandQueue);
  ret = clFinish(commandQueue);
  checkErr(ret, "finished queue");

  // profiling
  timeProf(&nanoseconds, done);

  ret = clReleaseEvent(done);
  checkErr(ret, "released event");
  ret = clReleaseKernel(kernel);
  checkErr(ret, "released kernel");
  ret = clReleaseProgram(program);
  checkErr(ret, "released program");
  ret = clReleaseMemObject(dC);
  ret |= clReleaseMemObject(dB);
  ret |= clReleaseMemObject(dA);
  checkErr(ret, "released mem buffers");
  ret = clReleaseCommandQueue(commandQueue);
  checkErr(ret, "released command queue");
  ret = clReleaseContext(context);
  checkErr(ret, "released context");
  free(kernelSource);
  checkErr(ret, "freed kernel source");
  printf("!\nkernel %s:%s run in %f milliseconds\n", filename, func,
         nanoseconds / 1000000.0);
}