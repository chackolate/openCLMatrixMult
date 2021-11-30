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
#include <cl.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  // Setup
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
}