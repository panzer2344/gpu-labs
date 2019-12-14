#include <CL/cl.h>
#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <cmath>
#include <limits>
#include <ctime>
#include "omp.h"
#include <chrono>

const unsigned int SIZE = int(1e+8);

using namespace std;
using namespace chrono;
std::string read_file(const std::string & path) {
	return std::string(
		(std::istreambuf_iterator<char>(
			*std::make_unique<std::ifstream>(path)
			)),
		std::istreambuf_iterator<char>()
	);
}

bool is_equalf(double x, double y) {
	return std::fabs(x - y) < std::numeric_limits<float>::epsilon();
}

bool is_equald(double x, double y) {
	return std::fabs(x - y) < std::numeric_limits<double>::epsilon();
}

void checkf(int n, float * res1, float* res2, float* res3, float* res4)
{
	int err = 0;
	for (int i = 0; i < n; i++)
	{
		if (!is_equalf(res1[i], res2[i]) || !is_equalf(res2[i], res3[i]) || !is_equalf(res2[i], res4[i]))
		{
			err++;
			printf("	seq : %f    opencl : %f    openmp : %f		opencl_cpu : %f \n", res1[i], res2[i], res3[i], res4[i]);
			printf("    falied math\n");
		}
	}
	std::cout << "number of errors = " << err << std::endl;
	return;
}

void checkd(int n, double * res1, double* res2, double *res3)
{
	int err = 0;
	for (int i = 0; i < n; i++)
	{
		if (!is_equalf(res1[i], res2[i]) || !is_equalf(res2[i], res3[i]))
		{
			err++;
			printf("	seq : %f    opencl : %f    openmp : %f \n", res1[i], res2[i], res3[i]);
			printf("	falied math\n");
		}
	}
	std::cout << "number of errors = " << err << std::endl;
	return;
}

void  saxpy_omp(int n, float a, float *x, float *y)
{
#pragma omp parallel 
	{
		cout << "hell" << endl;
	}

#pragma omp parallel for
	for (int i = 0; i < n; i++)
		y[i] = y[i] + a * x[i];
	return;
}

void  daxpy_omp(int n, double a, double *x, double *y)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		y[i] = y[i] + a * x[i];
	return;
}

void  saxpy(int n, float a, float *x, float *y)
{
	for( int i = 0; i < n; i++)
		y[i] = y[i] + a * x[i];
	return;
}

void  daxpy(int n, double a, double *x, double *y)
{
	for (int i = 0; i < n; i++)
		y[i] = y[i] + a * x[i];
	return;
}

void saxpy_setting(cl_context context, cl_device_id device, cl_kernel kernel, cl_command_queue queue, cl_context context_CPU, cl_device_id device_CPU, cl_kernel kernel_CPU, cl_command_queue queue_CPU)
{
	printf("saxpy setting\n");

	srand(time(0));
	cl_int error = 0;

	clock_t start = clock();
	clock_t finish = clock();
	float time_sec1 = 0;
	float time_sec2 = 0;
	float time_sec3 = 0;
	float time_sec4 = 0;

	size_t count = SIZE;

	size_t size_gpu = count;
	if (count % 256 != 0)
		size_gpu = (count / 256) * 256 + 256;
	float *x = new float[SIZE];
	float *y1 = new float[SIZE];
	float *y2 = new float[SIZE];
	float *y3 = new float[SIZE];
	float *y4 = new float[SIZE];


	float a = 2.0f;

	for (int i = 0; i < SIZE; i++)
	{
		x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 200));
		y1[i] = y2[i] = y3[i] = y4[i] =  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 200));
	}

	start = clock();
	saxpy(SIZE, a, x, y1);
	finish = clock();
	time_sec1 = (float(finish - start) / CLOCKS_PER_SEC);

	start = clock();
	saxpy_omp(SIZE, a, x, y3);
	finish = clock();
	time_sec3 = (float(finish - start) / CLOCKS_PER_SEC);

	{
		cl_mem inputx = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, &error);
		if (error != CL_SUCCESS) {
			std::cout << "Create buffer failed" << std::endl;
		}

		cl_mem inputy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * SIZE, NULL, &error);
		if (error != CL_SUCCESS) {
			std::cout << "Create buffer failed" << std::endl;
		}

		error = clEnqueueWriteBuffer(queue, inputx, CL_TRUE, 0, sizeof(float) * SIZE, x, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "Write buffer failed" << std::endl;
		}
		error = clEnqueueWriteBuffer(queue, inputy, CL_TRUE, 0, sizeof(float) * SIZE, y2, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "Write buffer failed" << std::endl;
		}

		error = clSetKernelArg(kernel, 0, sizeof(int), &count);
		if (error != CL_SUCCESS) {
			std::cout << "set arg1 kernel failed" << std::endl;
		}
		error = clSetKernelArg(kernel, 1, sizeof(float), &a);
		if (error != CL_SUCCESS) {
			std::cout << "set arg2 kernel failed" << std::endl;
		}

		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputx);
		if (error != CL_SUCCESS) {
			std::cout << "set arg3 kernel failed" << std::endl;
		}
		error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputy);
		if (error != CL_SUCCESS) {
			std::cout << "set arg4 kernel failed" << std::endl;
		}

		size_t group = 0;
		error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "clGetKernelWorkGroupInfo failed" << std::endl;
		}

		std::cout << "group : " << group << std::endl;
		group = 256;

		start = clock();
		error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size_gpu, &group, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "clEnqueueNDRangeKernel failed" << std::endl;
		}
		finish = clock();
		time_sec2 = (float(finish - start) / CLOCKS_PER_SEC);
		if (error != CL_SUCCESS) {
			std::cout << "clEnqueueNDRangeKernel failed" << std::endl;
		}
		clEnqueueReadBuffer(queue, inputy, CL_TRUE, 0, sizeof(float) * SIZE, y2, 0, NULL, NULL);

	}

	{
		cl_mem inputx = clCreateBuffer(context_CPU, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, &error);
		if (error != CL_SUCCESS) {
			std::cout << "Create buffer failed" << std::endl;
		}

		cl_mem inputy = clCreateBuffer(context_CPU, CL_MEM_READ_WRITE, sizeof(float) * SIZE, NULL, &error);
		if (error != CL_SUCCESS) {
			std::cout << "Create buffer failed" << std::endl;
		}

		error = clEnqueueWriteBuffer(queue_CPU, inputx, CL_TRUE, 0, sizeof(float) * SIZE, x, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "Write buffer failed" << std::endl;
		}
		error = clEnqueueWriteBuffer(queue_CPU, inputy, CL_TRUE, 0, sizeof(float) * SIZE, y4, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "Write buffer failed" << std::endl;
		}

		error = clSetKernelArg(kernel_CPU, 0, sizeof(int), &count);
		if (error != CL_SUCCESS) {
			std::cout << "set arg1 kernel failed" << std::endl;
		}
		error = clSetKernelArg(kernel_CPU, 1, sizeof(float), &a);
		if (error != CL_SUCCESS) {
			std::cout << "set arg2 kernel failed" << std::endl;
		}

		error = clSetKernelArg(kernel_CPU, 2, sizeof(cl_mem), &inputx);
		if (error != CL_SUCCESS) {
			std::cout << "set arg3 kernel failed" << std::endl;
		}
		error = clSetKernelArg(kernel_CPU, 3, sizeof(cl_mem), &inputy);
		if (error != CL_SUCCESS) {
			std::cout << "set arg4 kernel failed" << std::endl;
		}

		size_t group = 0;
		error = clGetKernelWorkGroupInfo(kernel_CPU, device_CPU, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
		if (error != CL_SUCCESS) {
			std::cout << "clGetKernelWorkGroupInfo failed" << std::endl;
		}

		std::cout << "group : " << group << std::endl;
		group = 256;


		std::chrono::time_point<std::chrono::system_clock> start_c, end_c;
		start_c = std::chrono::system_clock::now();
		start = clock();

		error = clEnqueueNDRangeKernel(queue_CPU, kernel_CPU, 1, NULL, &size_gpu, &group, 0, NULL, NULL);
		clFinish(queue_CPU);
		if (error != CL_SUCCESS) {
			std::cout << "clEnqueueNDRangeKernel failed" << std::endl;
		}
		
		end_c = std::chrono::system_clock::now();
		finish = clock();
		time_sec4 = (float(finish - start) / CLOCKS_PER_SEC);

		clEnqueueReadBuffer(queue_CPU, inputy, CL_TRUE, 0, sizeof(float) * SIZE, y4, 0, NULL, NULL);

		duration<double> sec = end_c - start_c;
		cout << sec.count() << " сек." << endl;

		std::chrono::milliseconds d = std::chrono::duration_cast<std::chrono::milliseconds >(sec);

		std::cout << sec.count() << "s\n";
		std::cout << d.count() << "ms\n";
	}

	checkf(count, y1, y2, y3, y4);

	std::cout << "seq : " << time_sec1 << std::endl;
	std::cout << "opencl gpu : " << time_sec2 << std::endl;
	std::cout << "openmp : " << time_sec3 << std::endl;
	std::cout << "opencl cpu : " << time_sec4 << std::endl;


	std::cout << std::endl;
	std::cout << std::endl;


	delete[]x;
	delete[]y1;
	delete[]y2;
	delete[]y3;

	return;
}

void daxpy_setting(cl_context context, cl_device_id device, cl_kernel kernel, cl_command_queue queue)
{
	printf("daxpy setting\n");

	srand(time(0));
	cl_int error = 0;

	clock_t start = clock();
	clock_t finish = clock();
	float time_sec1 = 0;
	float time_sec2 = 0;
	float time_sec3 = 0;
	size_t count = SIZE;

	size_t size_gpu = count;
	if (count % 256 != 0)
		size_gpu = (count / 256) * 256 + 256;

	double *x = new double[SIZE];
	double *y1 = new double[SIZE];
	double *y2 = new double[SIZE];
	double *y3 = new double[SIZE];

	double a = 2.0f;

	for (int i = 0; i < SIZE; i++)
	{
		x[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 200));
		y1[i] = y2[i] = y3[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 200));
	}

	start = clock();
	daxpy(SIZE, a, x, y1);
	finish = clock();
	time_sec1 = (float(finish - start) / CLOCKS_PER_SEC);

	start = clock();
	daxpy_omp(SIZE, a, x, y3);
	finish = clock();
	time_sec3 = (float(finish - start) / CLOCKS_PER_SEC);


	cl_mem inputx = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}

	cl_mem inputy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * SIZE, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}

	error = clEnqueueWriteBuffer(queue, inputx, CL_TRUE, 0, sizeof(double) * SIZE, x, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		std::cout << "Write buffer failed" << std::endl;
	}
	error = clEnqueueWriteBuffer(queue, inputy, CL_TRUE, 0, sizeof(double) * SIZE, y2, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		std::cout << "Write buffer failed" << std::endl;
	}

	error = clSetKernelArg(kernel, 0, sizeof(size_t), &count);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}
	error = clSetKernelArg(kernel, 1, sizeof(double), &a);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}

	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputx);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputy);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}

	size_t group = 0;
	error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
	if (error != CL_SUCCESS) {
		std::cout << "clGetKernelWorkGroupInfo failed" << std::endl;
	}

	start = clock();
	error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size_gpu, &group, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		std::cout << "clEnqueueNDRangeKernel failed" << std::endl;
	}
	finish = clock();
	time_sec2 = (float(finish - start) / CLOCKS_PER_SEC);
	if (error != CL_SUCCESS) {
		std::cout << "clEnqueueNDRangeKernel failed" << std::endl;
	}
	clEnqueueReadBuffer(queue, inputy, CL_TRUE, 0, sizeof(double) * SIZE, y2, 0, NULL, NULL);

	checkd(count, y1, y2, y3);

	std::cout << "seq : " << time_sec1 << std::endl;
	std::cout << "opencl : " << time_sec2 << std::endl;
	std::cout << "openmp : " << time_sec3 << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;



	delete[]x;
	delete[]y1;
	delete[]y2;
	delete[]y3;

	return;
}

cl_context setting_context(cl_device_type str)
{
	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
	if (0 < num_platforms) {
		cl_platform_id *platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[1];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		std::cout << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, str, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed" << std::endl;
	}
	return context;
}

cl_device_id setting_device(cl_context context)
{

	cl_int error = 0;

	size_t size = 0;

	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

	cl_device_id device = 0;
	if (size > 0) {
		cl_device_id *devices = (cl_device_id *)alloca(size);
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device = devices[0];

		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed" << std::endl;
	}
	return device;
}


cl_command_queue setting_queue(cl_context context, cl_device_id device)
{
	cl_int error = 0;

	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed" << std::endl;
	}
	return queue;
}

int main() {

	cl_int error = 0;

	cl_context context_GPU = setting_context(CL_DEVICE_TYPE_GPU);
	cl_device_id device_GPU = setting_device(context_GPU);
	cl_command_queue queue_GPU = setting_queue(context_GPU, device_GPU);

	cl_context context_CPU = setting_context(CL_DEVICE_TYPE_CPU);
	cl_device_id device_CPU = setting_device(context_CPU);
	cl_command_queue queue_CPU = setting_queue(context_CPU, device_CPU);

	read_file("OpenCLFile.cl").data();
	std::string str = read_file("OpenCLFile.cl");
	const char * t_str = str.data();
	size_t srclen[] = { str.length() };

	cl_program program_GPU = clCreateProgramWithSource(context_GPU, 1, &t_str, srclen, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}

	clBuildProgram(program_GPU, 1, &device_GPU, NULL, NULL, NULL);

	cl_kernel kernel1 = clCreateKernel(program_GPU, "saxpy", &error);
	if (error != CL_SUCCESS) {
		std::cout << error << std::endl;
		std::cout << "Create kernel failed" << std::endl;
	}

	//cl_kernel kernel2 = clCreateKernel(program_GPU, "daxpy", &error);
	if (error != CL_SUCCESS) {
		std::cout << error << std::endl;
		std::cout << "Create kernel failed" << std::endl;
	}


	cl_program program_CPU = clCreateProgramWithSource(context_CPU, 1, &t_str, srclen, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}

	clBuildProgram(program_CPU, 1, &device_CPU, NULL, NULL, NULL);

	//cl_kernel kernel3 = clCreateKernel(program_CPU, "saxpy", &error);
	if (error != CL_SUCCESS) {
		std::cout << error << std::endl;
		std::cout << "Create kernel failed" << std::endl;
	}

	//cl_kernel kernel4 = clCreateKernel(program_CPU, "daxpy", &error);
	if (error != CL_SUCCESS) {
		std::cout << error << std::endl;
		std::cout << "Create kernel failed" << std::endl;
	}
	//------------------------------------------------------------------------------------------------------------------

	saxpy_setting(context_GPU, device_GPU, kernel1, queue_GPU, context_CPU, device_CPU, kernel3, queue_CPU);

	//daxpy_setting(context, device, kernel2, queue);

	clFinish(queue_GPU);
	//clFinish(queue_CPU);

	clReleaseProgram(program_GPU);
	clReleaseProgram(program_CPU);

	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseKernel(kernel3);
	clReleaseKernel(kernel4);

	clReleaseContext(context_GPU);
	clReleaseContext(context_CPU);

	return 0;
}
