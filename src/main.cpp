#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include <memory>
#include "CL/cl.h"
#include "fft_config.h"

#define N (1 << LOGN)

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL, queue2 = NULL, queue3 = NULL;
static cl_kernel fft_kernel = NULL, fetch_kernel = NULL, transpose_kernel = NULL;
static cl_program program = NULL;
static cl_int status = 0;

// use struct to store complex numbers 
typedef struct {
  double x;
  double y;
} double2;

typedef struct {
  float x;
  float y;
} float2;

bool init();
void cleanup();
static void test_fft(bool mangle, bool inverse);
static int coord(int iteration, int i);
static void fourier_transform_gold(bool inverse, int lognr_points, double2 * data);
static void fourier_stage(int lognr_points, double2 * data);
static int mangle_bits(int x);

// Host memory
float2 *h_inData, *h_outData;
double2 *h_verify, *h_verify_tmp;

// Device  buffers
#if USE_SVM_API == 0
cl_mem d_inData, d_outData, d_tmp;
#else
float2 *h_tmp;
#endif /* USE_SVM_API == 0 */

// Entry point.
int main(int argc, char **argv) {
  if(!init()) {
    return false;
  }

  h_inData = (float2 *)malloc(sizeof(float2) * N * N);
  h_outData = (float2 *)malloc(sizeof(float2) * N * N);

  h_verify = (double2 *),alloc(sizeof(double2) * N * N);
  h_verify_tmp = (double2 *)malloc(sizeof(double2) * N * N);
  
  if (!(h_inData && h_outData && h_verify && h_verify_tmp)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  test_fft(false, false); // test FFT transform with ordered memory layout
  test_fft(false, true); // test inverse FFT transform with ordered memory layout

  test_fft(true, false); // test FFT transform with alternative memory layout
  test_fft(true, true); // test inverse FFT transform with alternative memory layout

  // Free the resources allocated
  cleanup();

  return 0;
}

void test_fft(bool mangle, bool inverse) {
  printf("Launching %sFFT transform (%s data layout)\n", inverse ? "inverse " : "", mangle ? "alternative" : "ordered");


  // Initialize input and produce verification data
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int where = mangle ? mangle_bits(coord(i, j)) :  coord(i, j);
      h_verify[coord(i, j)].x = h_inData[where].x = (float)((double)rand() / (double)RAND_MAX);
      h_verify[coord(i, j)].y = h_inData[where].y = (float)((double)rand() / (double)RAND_MAX);
    }
  }

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");
  d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queue, d_inData, CL_TRUE, 0, sizeof(float2) * N * N, h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Convert bool to int (opencl specific)
  int inverse_int = inverse;

  int mangle_int = mangle;

  printf("Kernel initialization is complete.\n");

  // Get the iterationstamp to evaluate performance
  double time = getCurrentTimestamp();

  // Loop twice over the kernels
  for (int i = 0; i < 2; i++) {

    status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), i == 0 ? (void *)&d_inData : (void *)&d_tmp);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void*)&mangle_int);
    checkError(status, "Failed to set kernel arg 1");
    size_t lws_fetch[] = {N};
    size_t gws_fetch[] = {N * N / 8};
    status = clEnqueueNDRangeKernel(queue, fetch_kernel, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Launch the fft kernel - we launch a single work item hence enqueue a task
    status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 0");
    status = clEnqueueTask(queue2, fft_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Set the kernel arguments
    status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), i == 0 ? (void *)&d_tmp : (void *)&d_outData);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(transpose_kernel, 1, sizeof(cl_int), (void*)&mangle_int);
    checkError(status, "Failed to set kernel arg 1");
    size_t lws_transpose[] = {N};
    size_t gws_transpose[] = {N * N / 8};
    status = clEnqueueNDRangeKernel(queue3, transpose_kernel, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Wait for all command queues to complete pending events
    status = clFinish(queue);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
  }

  // Record execution time
  time = getCurrentTimestamp() - time;

  status = clEnqueueReadBuffer(queue, d_outData, CL_TRUE, 0, sizeof(float2) * N * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");

  printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
  double gpoints_per_sec = ((double) N * N / time) * 1E-9;
  double gflops = 2 * 5 * N * N * (log((float)N)/log((float)2))/(time * 1E9);
  printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);

 
}

// provides a linear offset in the input array
int coord(int iteration, int i) {
  return iteration * N + i;
}

// This modifies the linear matrix access offsets to provide an alternative
// memory layout to improve the efficiency of the memory accesses
int mangle_bits(int x) {
   const int NB = LOGN / 2;
   int a95 = x & (((1 << NB) - 1) << NB);
   int a1410 = x & (((1 << NB) - 1) << (2 * NB));
   int mask = ((1 << (2 * NB)) - 1) << NB;
   a95 = a95 << NB;
   a1410 = a1410 >> NB;
   return (x & ~mask) | a95 | a1410;
}


// Reference Fourier transform
void fourier_transform_gold(bool inverse, const int lognr_points, double2 *data) {
   const int nr_points = 1 << lognr_points;

   // The inverse requires swapping the real and imaginary component
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         std::swap(data[i].x, data[i].y);
      }
   }
   // Do a FT recursively
   fourier_stage(lognr_points, data);

   // The inverse requires swapping the real and imaginary component
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         std::swap(data[i].x, data[i].y);
      }
   }
}

void fourier_stage(int lognr_points, double2 *data) {
   int nr_points = 1 << lognr_points;

   if (nr_points == 1) return;

   double2 *half1 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
   double2 *half2 = (double2 *)alloca(sizeof(double2) * nr_points / 2);

   for (int i = 0; i < nr_points / 2; i++) {
      half1[i] = data[2 * i];
      half2[i] = data[2 * i + 1];
   }

   fourier_stage(lognr_points - 1, half1);
   fourier_stage(lognr_points - 1, half2);

   for (int i = 0; i < nr_points / 2; i++) {
      data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
   }
}

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Query available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // ok, take the first, hopefully the gpu
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create one command queue for each kernel.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  std::string binary_file = getBoardBinaryFile("fft2d", device);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fft_kernel = clCreateKernel(program, "fft2d", &status);
  checkError(status, "Failed to create kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create kernel");

  cl_device_svm_capabilities caps = 0;

  status = clGetDeviceInfo(
    device,
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    0
  );
  checkError(status, "Failed to get device info");

    // Free resources
    cleanup();
  
  return true;
}

// Free the resources, its a law!
void cleanup() {
  if(fft_kernel) 
    clReleaseKernel(fft_kernel);  
  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
  if (h_verify)
    alignedFree(h_verify);
  if (h_verify_tmp)
    alignedFree(h_verify_tmp);
  if(h_inData)
	alignedFree(h_inData);
  if (h_outData)
	alignedFree(h_outData);
  if (d_inData)
	clReleaseMemObject(d_inData);
  if (d_outData) 
	clReleaseMemObject(d_outData);
  if (d_tmp)
    clReleaseMemObject(d_tmp);
  if(context)
    clReleaseContext(context);
}



