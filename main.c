#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 200
#define MAX_SOURCE_SIZE (0x1000000)
#define N 1024 // vector length

int main(void) {
  // create inputs
  int i;
  float *h_A = (float *)malloc(sizeof(float) * N);
  float *h_B = (float *)malloc(sizeof(float) * N);
  float *h_C = (float *)malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) {
    h_A[i] = i;
    h_B[i] = N - i;
  }

  // load kernel .cl into array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;
  fp = fopen("vectorAddition.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // platform & device info
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(0, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, &device_id,
                       &ret_num_devices);
  printf("%d\n", ret);

  // OpenCL context
  cl_context context =
      clCreateContextFromType(NULL, 1, &device_id, NULL, NULL, &ret);

  // command queue
  cl_command_queue command_queue =
      clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
  // clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

  // memory buffers on device
  cl_mem d_A =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
  cl_mem d_B =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
  cl_mem d_C =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);

  // copy host lists to device buffers
  ret = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, N * sizeof(float),
                             h_A, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, N * sizeof(float),
                             h_B, 0, NULL, NULL);

  // create program from kernel
  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                (const size_t *)&source_size, &ret);

  // Build program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // create kernel
  cl_kernel kernel = clCreateKernel(program, "vectorAdd", &ret);

  // kernel arguments
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_A);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_B);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_C);

  // execute kernel on list
  size_t global_item_size = N; // 1024
  size_t local_item_size = 64; // process in groups of 64
  ret =
      clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size,
                             &local_item_size, 0, NULL, NULL);

  // Read device C to host C
  ret = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, N * sizeof(float),
                            h_C, 0, NULL, NULL);

  // display results
  for (int i = 0; i < N; i++) {
    // printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
  }

  // clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(d_A);
  ret = clReleaseMemObject(d_B);
  ret = clReleaseMemObject(d_C);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}