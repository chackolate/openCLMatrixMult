// helper for openCL matrix multiplication

#ifndef CLHELPER_H_
#define CLHELPER_H_

#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

#define N 128
#define MAX_SOURCE_SIZE (0x100000)

const char *getErrorString(cl_int error);

void checkErr(cl_int error, char *success);

double randDouble(double min, double max);

void initHost(double *hA, double *hB);

void kernelFromFile(size_t *kernelSize, char *kernelSource, char *filename);

void getPlatformDevice(cl_platform_id *platformID, cl_device_id *deviceID);

void createContext(cl_context *context, cl_device_id *deviceID);

void createQueue(cl_command_queue *commandQueue, cl_context *context,
                 cl_device_id *deviceID, int outOfOrder, int profiling);

void createBuffer(cl_mem *deviceBuffer, int direction, cl_context *context);

void cpyHostToDevice(cl_mem dest, double *source,
                     cl_command_queue *commandQueue);

void createProgramFromSource(cl_program *program, cl_context *context,
                             const char *kernelSource, size_t *kernelSize);

void buildProgram(cl_program *program, cl_device_id *deviceID);

void createKernel(cl_kernel *kernel, cl_program *program, char *funcName);

void setArgs(cl_kernel *kernel, cl_mem dA, cl_mem dB, cl_mem dC);

void execKernel(cl_device_id deviceID, cl_command_queue *commandQueue,
                cl_kernel *kernel, cl_event *event);

void readDeviceToHost(cl_mem source, double *dest,
                      cl_command_queue *commandQueue);

void gpuBench(double *A, double *B, double *C, double nanoseconds);

#endif