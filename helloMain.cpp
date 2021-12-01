// 1. Setup:
// get devices & platform
// create context (share between host & device)
// create command queues (submit work here)
// 2. Compilation:
// create program
// build program
// create kernel
// 3. Create memory objects
// 4. Enqueue writes copy data to GPU
// 5. Set kernel arguments
// 6. Enqueue kernel executions
// 7. Enqueue reads copy data back from GPU
// 8. Wait for commands to finish

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>

int main() {
  // Setup
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  auto platform = platforms.front();
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  auto device = devices.front();

  // 1. Create context
  cl::Context context(device);
  // 2. Load file
  std::ifstream helloWorldFile("helloWorld.cl");
  std::string src(std::istreambuf_iterator<char>(helloWorldFile),
                  (std::istreambuf_iterator<char>()));
  // 3. Create program
  cl_int err;
  cl::Program program(context, src, CL_TRUE, &err);
  // 4. Create kernel
  cl::Kernel kernel(program, "helloWorld", &err);
  // 5. Create memory objects
  char bufStr[16];
  cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                    sizeof(bufStr));
  kernel.setArg(0, memBuf);

  cl::CommandQueue queue(context, device);
  queue.enqueueNDRangeKernel(kernel, NULL, NULL, NULL, NULL, NULL);
  queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(bufStr), bufStr);

  std::cout << bufStr;
  std::cin.get();
}