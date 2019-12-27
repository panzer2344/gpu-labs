#include <CL/cl.h>
#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <cmath>
#include <limits>
#include <ctime>
#include "omp.h"
#include <vector>
#include <algorithm> 
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
using namespace std;

std::string read_file(const std::string & path) {
	return std::string(
		(std::istreambuf_iterator<char>(
			*std::make_unique<std::ifstream>(path)
			)),
		std::istreambuf_iterator<char>()
	);
}

const char* kernelFilename = "D:\\Git\\GPU\\Kernel4.cl";

const char *getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}


bool checkError(cl_int error, const char* error_message) {
	if (error != CL_SUCCESS) {
		std::cout << error_message << std::endl;
		std::cout << error << std::endl;
		std::cout << getErrorString(error) << std::endl;
		return true;
	}
	return false;
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

float* gen_mat(size_t size) {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::uniform_real_distribution<float> dist{ -100, 100 };

	float* mem = new float[size * size];
	for (size_t i = 0; i < size; ++i) {
		float sum = 0.0f;
		for (size_t j = 0; j < size; ++j) {
			float tmp = dist(rd);
			mem[i * size + j] = tmp;
			sum += abs(tmp);
		}
		mem[i * size + i] = sum + abs(dist(rd)) + 1.0f;
	}

	return mem;
}

float * gen_vec(size_t size) {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::uniform_real_distribution<float> dist{ -100, 100 };

	float* mem = new float[size];
	for (size_t i = 0; i < size; ++i) {
		mem[i] = dist(rd);
	}

	return mem;
}

void print_mat(float * mat, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			std::cout << mat[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
}
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

cl_context setting_context(cl_device_type str)
{
	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
	if (0 < num_platforms) {
		cl_platform_id *platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);

		int id = str == CL_DEVICE_TYPE_GPU ? 0 : 1;

		platform = platforms[id];

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


float opencl(cl_device_type str_device, const string name_kernel, float* A, float* b, size_t size, size_t max_iteration, float eps)
{
	clock_t start = clock();
	clock_t finish = clock();
	float time = 0.0f;

	float * tmp = new float[size];
	for (size_t i = 0; i < size; ++i) {
		tmp[i] = rand();
	}

	float * x_old = new float[size];
	for (size_t i = 0; i < size; ++i) {
		x_old[i] = tmp[i];
	}
	float * x_new = new float[size]();

	cl_int error = 0;

	cl_context context = setting_context(str_device);
	cl_device_id device = setting_device(context);
	cl_command_queue queue = setting_queue(context, device);

	std::string str = read_file(kernelFilename);
	const char * t_str = str.data();
	size_t srclen[] = { str.length() };

	//cout << str.data() << endl;

	cl_program program_GPU = clCreateProgramWithSource(context, 1, &t_str, srclen, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}
	clBuildProgram(program_GPU, 1, &device, nullptr, nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program_GPU, name_kernel.data(), &error);
	if (error != CL_SUCCESS) {
		std::cout << error << std::endl;
		std::cout << "Create kernel failed " << str_device << std::endl;
	}

	cl_mem input1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size * size, A, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}
	cl_mem input2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, b, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}

	cl_mem output1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, x_old, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}
	cl_mem output2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, x_new, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create buffer failed" << std::endl;
	}

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output1);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output2);
	if (error != CL_SUCCESS) {
		std::cout << "set arg kernel failed" << std::endl;
	}

	const size_t global_work_offset[1] = { 0 };
	const size_t global_work_size[1] = { size };
	const size_t local_work_size[1] = { 16 };

	start = clock();
	cl_event evt;
	int iter = 0;
	float accuracy = 0.0f;

	start = clock();
	do {
		error = clEnqueueNDRangeKernel(queue, kernel, 1, global_work_offset, global_work_size, local_work_size, 0, nullptr, &evt);
		error = clWaitForEvents(1, &evt);
		error = clEnqueueReadBuffer(queue, output2, CL_TRUE, 0, size * sizeof(float), x_new, 0, nullptr,
			nullptr);

		cl_ulong start_time = 0, end_time = 0;
		error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time,nullptr);
		error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
		time += end_time - start_time;

		accuracy = 0.0f;
		for (size_t i = 0; i < size; ++i) {
			accuracy += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]);
		}

		accuracy = std::sqrt(accuracy);
		iter++;

		std::swap(x_old, x_new);
		std::swap(output1, output2);

		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output1);
		error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output2);
		if (error != CL_SUCCESS) {
			std::cout << "set arg kernel failed" << std::endl;
		}
	} while ( (accuracy > eps) && (iter < max_iteration));
	finish = clock();

	error = clFinish(queue);
	error = clEnqueueReadBuffer(queue, output1, CL_TRUE, 0, size * sizeof(float), x_new, 0, nullptr, nullptr);

	float err = 0.0f;
	for (size_t i = 0; i < size; ++i) {
		float acc = 0.0f;
		for (size_t j = 0; j < size; ++j) {
			acc += A[j * size + i] * x_new[j];
		}
		err += (acc - b[i]) * (acc - b[i]);
	}
	err = std::sqrt(err);

	//printf("error = %g\n", err);
	
	time = (float(finish - start) / CLOCKS_PER_SEC);

	return time;
}

void calculations_setting()
{
	printf("calculations setting\n");

	srand(time(0));
	size_t size = 8192;//8192*2;
	size_t max_iteration = 100;
	float eps = 1e-7;
	cout << eps << endl;

	float* A = gen_mat(size);
	float* b = gen_vec(size);

	clock_t start = clock();
	clock_t finish = clock();
	float time_gpu = 0;
	float time_cpu = 0;


	time_gpu = opencl(CL_DEVICE_TYPE_GPU, "jacobi", A, b, size, max_iteration, eps);
	time_cpu = opencl(CL_DEVICE_TYPE_CPU, "jacobi", A, b, size, max_iteration, eps);

	cout << "time gpu : " << time_gpu << endl;
	cout << "time cpu : " << time_cpu << endl;


	return;
}


int main() 
{
	calculations_setting();

	system("pause");

	return 0;
}
