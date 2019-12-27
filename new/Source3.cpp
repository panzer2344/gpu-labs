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

using namespace std;

const char* kernelFilename = "D:\\Git\\GPU\\Kernel3.cl";

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


std::string read_file(const std::string & path) {
	return std::string(
		(std::istreambuf_iterator<char>(
			*std::make_unique<std::ifstream>(path)
			)),
		std::istreambuf_iterator<char>()
	);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

cl_context setting_context(cl_device_type str)
{
	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
	std::cout << "platforms number: " << num_platforms << std::endl;
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
	if (checkError(error, "Create context from type failed")) return NULL;

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
	if (checkError(error, "Create command queue with properties failed")) return NULL;

	return device;
}

cl_command_queue setting_queue(cl_context context, cl_device_id device)
{
	cl_int error = 0;

	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
	if (checkError(error, "Create command queue with properties failed")) return NULL;

	return queue;
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


void InitMatr(int n, int m, float* matrix1, float* matrix2, float* matrix3)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			matrix1[i*m + j] = matrix2[i*m + j] = matrix3[i*m + j] = 2.0f;  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 256));
}

bool is_equal(float x, float y) {
	return std::fabs(x - y) < std::numeric_limits<float>::epsilon();
}

void check(int n, int m, float* matrix1, float* matrix2)
{
	int err = 0;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (!is_equal(matrix1[i*m + j], matrix2[i*m + j]))
			{
				printf("matrix1  %f, matrix2  %f, index  %d \n", matrix1[i*m + j], matrix2[i*m + j], i*m + j);
				err++;
			}
	if(err != 0)
		cout << "number of errors =  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << err << endl;
	return;
}

void my_print(int n, int m, float* matrix1)
{
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%15.8f  ", matrix1[i*m + j]);
		}
		printf("\n");
	}
	return;
}

void sequential_multiplication(int A_N, int A_M, int B_N, int B_M, float* matrix1, float* matrix2, float* result)
{
	double temp = 0;
	for (int i = 0; i < A_N; i++)
		for (int j = 0; j < B_M; j++)
		{
			temp = 0;
			for (int k = 0; k < A_M; k++)
			{
				temp += matrix1[i*A_M + k] * matrix2[k*B_M + j];
			}
			result[i*B_M + j] = temp;
		}
	return;
}

void openmp_multiplication(int A_N, int A_M, int B_N, int B_M, float* matrix1, float* matrix2, float* result)
{
	double temp = 0;

#pragma omp parallel for num_threads(4) firstprivate(temp)
	for (int i = 0; i < A_N; i++)
		for (int j = 0; j < B_M; j++)
		{
			temp = 0;
			for (int k = 0; k < A_M; k++)
			{
				temp += matrix1[i*A_M + k] * matrix2[k*B_M + j];
			}
			result[i*B_M + j] = temp;
		}
	return;
}


void openmp_multiplication_opt(int A_N, int A_M, int B_N, int B_M, float* matrix1, float* matrix2, float* result)
{
	int BLOCK_SIZE = 16;
#pragma omp parallel for num_threads(1)
	for (int jj = 0; jj < B_M; jj += BLOCK_SIZE) {
		for (int kk = 0; kk < A_M; kk += BLOCK_SIZE) {
			for (int i = 0; i < A_N; i++) {
				for (int j = jj; j < std::min(A_N, jj + BLOCK_SIZE); j++) {
					float acc = 0.0f;
					for (int k = kk; k < std::min(A_N, kk + BLOCK_SIZE); k++) {
						acc += matrix1[i * A_N + k] * matrix1[k * A_M + j];
					}
					result[i * A_N + j] = acc;
				}
			}
		}
	}
	return;
}



//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

float opencl(int A_N, int A_M, int B_N, int B_M, float* truematrix, float* matrix_ocl1, float* matrix_ocl2, float*  result_ocl,cl_device_type str_device, const string name_kernel)
{
	clock_t start = clock();
	clock_t finish = clock();
	float time_seq = 0;
	float time_omp = 0;
	float time_ocl = 0;

	int C_N = A_N;
	int C_M = B_M;

	cl_int error = 0;

	cl_context context = setting_context(str_device);
	cl_device_id device = setting_device(context);
	cl_command_queue queue = setting_queue(context, device);

	std::string str = read_file(kernelFilename);
	const char * t_str = str.data();
	size_t srclen[] = { str.length() };

	cl_program program_GPU = clCreateProgramWithSource(context, 1, &t_str, srclen, &error);
	if (checkError(error, "Create program with source failed")) return -1.0;

	size_t TS = 16;

	std::string build_ops = "-D TS=" + std::to_string(TS);
	clBuildProgram(program_GPU, 1, &device, build_ops.c_str(), nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program_GPU, name_kernel.data(), &error);
	if (checkError(error, "Create kernel failed ")) return -1.0;

	cl_mem input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * A_N * A_M, NULL, &error);
	if (checkError(error, "Create buffer failed")) return -1.0;

	cl_mem input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * B_N * B_M, NULL, &error);
	if (checkError(error, "Create buffer failed")) return -1.0;

	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * C_N *C_M, NULL, &error);
	if (checkError(error, "Create buffer failed")) return -1.0;

	error = clEnqueueWriteBuffer(queue, input1, CL_TRUE, 0, sizeof(float) * A_N * A_M, matrix_ocl1, 0, NULL, NULL);
	if (checkError(error, "Write buffer failed")) return -1.0;

	error = clEnqueueWriteBuffer(queue, input2, CL_TRUE, 0, sizeof(float) * B_N * B_M, matrix_ocl2, 0, NULL, NULL);
	if (checkError(error, "Write buffer failed")) return -1.0;

	error = clSetKernelArg(kernel, 0, sizeof(int), &A_M);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 1, sizeof(int), &A_N);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 2, sizeof(int), &B_M);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input1);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input2);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
	if (checkError(error, "set arg kernel failed")) return -1.0;	

	size_t group = 0;
	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);

	const size_t global_work_offset[2] = { 0, 0 };
	const size_t global_work_size[2] = { A_N, B_M };
	const size_t local_work_size[2] = { TS, TS };
	start = clock();
	cl_event evt;

	error = clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, &evt);
	if (checkError(error, "clEnqueueNDRangeKernel failed")) return -1.0;

	clFinish(queue);
	error = clWaitForEvents(1, &evt);
	if (checkError(error, "clWaitForEvents failed")) return -1.0;

	finish = clock();
	time_ocl = (float(finish - start) / CLOCKS_PER_SEC);

	error = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * C_N * C_M, result_ocl, 0, NULL, NULL);
	if (checkError(error, "clEnqueueReadBuffer failed")) return -1.0;

	check(C_N, C_M, truematrix, result_ocl);

	return time_ocl;
}

float opencl_image(int A_N, int A_M, int B_N, int B_M, float* truematrix, float* matrix_ocl1, float* matrix_ocl2, float*  result_ocl, cl_device_type str_device, const string name_kernel)
{
	clock_t start = clock();
	clock_t finish = clock();
	float time_seq = 0;
	float time_omp = 0;
	float time_ocl = 0;

	int C_N = A_N;
	int C_M = B_M;

	cl_int error = 0;

	cl_context context = setting_context(str_device);
	cl_device_id device = setting_device(context);
	cl_command_queue queue = setting_queue(context, device);

	std::string str = read_file(kernelFilename);
	const char * t_str = str.data();
	size_t srclen[] = { str.length() };

	cl_program program_GPU = clCreateProgramWithSource(context, 1, &t_str, srclen, &error);
	if (checkError(error, "Create program with source failed")) return -1.0;

	size_t TS = 16;

	std::string build_ops = "-D TS=" + std::to_string(TS);
	clBuildProgram(program_GPU, 1, &device, build_ops.c_str(), nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program_GPU, name_kernel.data(), &error);
	if (checkError(error, "Create kernel failed ")) return -1.0;

	size_t size = A_N;
	cl_image_format format = {};
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_R;
	cl_mem A_buf, B_buf, C_buf;
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = size;
	desc.image_height = size;

	cl_int cl_status;

	A_buf = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, matrix_ocl1, &error);
	if (checkError(error, "clCreateImage failed1")) return -1.0;

	B_buf = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, matrix_ocl2, &error);
	if (checkError(error, "clCreateImage failed2")) return -1.0;

	C_buf = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, result_ocl, &error);
	if (checkError(error, "clCreateImage failed3")) return -1.0;

	error = clSetKernelArg(kernel, 0, sizeof(int), &A_N);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 1, sizeof(int), &B_M);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 2, sizeof(int), &A_M);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &A_buf);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &B_buf);
	if (checkError(error, "set arg kernel failed")) return -1.0;

	error = clSetKernelArg(kernel, 5, sizeof(cl_mem), &C_buf);
	if (checkError(error, "set arg kernel failed")) return -1.0;


	const size_t global_work_offset[2] = { 0, 0 };
	const size_t global_work_size[2] = { A_N, B_M };
	const size_t local_work_size[2] = { TS, TS };


	const size_t origin[3] = { 0, 0 ,0 };
	const size_t region[3] = { size, size, 1 };

	clEnqueueWriteImage(queue, A_buf, CL_TRUE, origin, region, 0, 0, matrix_ocl1, 0, NULL, NULL);
	clEnqueueWriteImage(queue, B_buf, CL_TRUE, origin, region, 0, 0, matrix_ocl2, 0, NULL, NULL);

	size_t group = 0;
	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);

	start = clock();
	cl_event evt;

	error = clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, &evt);
	if (checkError(error, "clEnqueueNDRangeKernel failed")) return -1.0;

	clFinish(queue);
	error = clWaitForEvents(1, &evt);
	if (checkError(error, "clWaitForEvents failed")) return -1.0;

	finish = clock();
	time_ocl = (float(finish - start) / CLOCKS_PER_SEC);

	cl_status = clEnqueueReadImage(queue, C_buf, true, origin, region, 0, 0, result_ocl, 0, nullptr, nullptr);
	check(C_N, C_M, truematrix, result_ocl);

	return time_ocl;
}

void calculations_setting()
{
	printf("calculations setting\n");

	srand(time(0));
	cl_int error = 0;

	// size matrix
	int matrixSize = 1024;

	int A_N = matrixSize;
	int A_M = matrixSize;
	int B_N = A_M;
	int B_M = matrixSize;
	int C_N = A_N;
	int C_M = B_M;

	//time
	clock_t start = clock();
	clock_t finish = clock();
	float time_seq = 0;
	float time_omp = 0;
	float time_ocl_gpu_native = 0;
	float time_ocl_cpu_native = 0;
	float time_ocl_gpu_opt = 0;
	float time_ocl_cpu_opt = 0;
	float time_ocl_gpu_image = 0;
	float time_ocl_cpu_image = 0;

	//init
	cout << "init" << endl;
	float* matrix_seq1;
	float* matrix_seq2;
	float* result_seq;

	float* matrix_omp1;
	float* matrix_omp2;
	float* result_omp;

	float* matrix_ocl1;
	float* matrix_ocl2;
	float* result_ocl;
	//memory
	cout << "memory" << endl;

	matrix_seq1 = new float[A_N*A_M];
	matrix_seq2 = new float[B_N*B_M];
	result_seq = new float[C_N*C_M];

	matrix_omp1 = new float[A_N*A_M];
	matrix_omp2 = new float[B_N*B_M];
	result_omp = new float[C_N*C_M];

	matrix_ocl1 = new float[A_N*A_M];
	matrix_ocl2 = new float[B_N*B_M];
	result_ocl = new float[C_N*C_M];

	//init metrix
	cout << "init metrix" << endl;
	InitMatr(A_N, A_M, matrix_seq1, matrix_omp1, matrix_ocl1);
	InitMatr(B_N, B_M, matrix_seq2, matrix_omp2, matrix_ocl2);

	start = clock();
	sequential_multiplication(A_N, A_M, B_N, B_M, matrix_seq1, matrix_seq2, result_seq);
	finish = clock();

	cout << "sequential_multiplication" << endl;
	cout << endl;
	time_seq = (float(finish - start) / CLOCKS_PER_SEC);

	start = clock();
	openmp_multiplication(A_N, A_M, B_N, B_M, matrix_omp1, matrix_omp2, result_omp);
	finish = clock();

	cout << "openmp_multiplication" << endl;
	cout << endl;

	time_omp = (float(finish - start) / CLOCKS_PER_SEC);
	check(C_N, C_M, result_seq, result_omp);

	cout << endl;
	std::cout << "seq : " << time_seq << std::endl;
	std::cout << "omp : " << time_omp << std::endl;
	cout << endl;


	int change = 4;
	if (change >= 2)
	{
		cout << " calculation native" << endl;

		time_ocl_gpu_native = opencl(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_GPU, "mult");
		check(C_N, C_M, result_seq, result_ocl);

		time_ocl_cpu_native = opencl(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_CPU, "mult");
		check(C_N, C_M, result_seq, result_ocl);


		cout << endl;
		cout << "time native :" << endl;

		std::cout << "time_ocl_gpu native: " << time_ocl_gpu_native << std::endl;
		std::cout << "time_ocl_cpu native: " << time_ocl_cpu_native << std::endl;
		std::cout << "seq : " << time_seq << std::endl;
		std::cout << "omp : " << time_omp << std::endl;
		cout << endl;

	}
	if (change >= 3)
	{
		cout << " calculation opt" << endl;

		time_ocl_gpu_native = opencl(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_GPU, "mult_opt");
		check(C_N, C_M, result_seq, result_ocl);

		time_ocl_cpu_native = opencl(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_CPU, "mult_opt");
		check(C_N, C_M, result_seq, result_ocl);

		cout << endl;
		cout << "time opt :" << endl;
		std::cout << "time_ocl_gpu opt: " << time_ocl_gpu_native << std::endl;
		std::cout << "time_ocl_cpu opt: " << time_ocl_cpu_native << std::endl;
		std::cout << "seq : " << time_seq << std::endl;
		std::cout << "omp : " << time_omp << std::endl;
		cout << endl;

	}
	if (change >= 4)
	{
		cout << " calculation image" << endl;

		time_ocl_gpu_native = opencl_image(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_GPU, "matmul_block_image");
		check(C_N, C_M, result_seq, result_ocl);

		time_ocl_cpu_native = opencl_image(A_N, A_M, B_N, B_M, result_seq, matrix_ocl1, matrix_ocl2, result_ocl, CL_DEVICE_TYPE_CPU, "matmul_block_image");
		check(C_N, C_M, result_seq, result_ocl);


		cout << endl;
		cout << "time image :" << endl;
		std::cout << "time_ocl_gpu image: " << time_ocl_gpu_native << std::endl;
		std::cout << "time_ocl_cpu image: " << time_ocl_cpu_native << std::endl;
		std::cout << "seq : " << time_seq << std::endl;
		std::cout << "omp : " << time_omp << std::endl;

	}

	//delete memory
	delete[] matrix_seq1;
	delete[] matrix_omp1;
	delete[] matrix_seq2;
	delete[] matrix_omp2;
	delete[] result_seq;
	delete[] result_omp;

	delete[] matrix_ocl1;
	delete[] matrix_ocl2;
	delete[] result_ocl;

	return;
}


int main() {

	cl_int error = 0;

	cl_context context_GPU = setting_context(CL_DEVICE_TYPE_GPU);
	cl_device_id device_GPU = setting_device(context_GPU);
	cl_command_queue queue_GPU = setting_queue(context_GPU, device_GPU);

	std::string str = read_file(kernelFilename);
	const char * t_str = str.data();
	size_t srclen[] = { str.length() };

	cl_program program_GPU = clCreateProgramWithSource(context_GPU, 1, &t_str, srclen, &error);
	if (checkError(error, "Create program with source failed")) { system("pause"); return -1.0; }

	std::string build_ops = "-D TS=" + std::to_string(16);
	clBuildProgram(program_GPU, 1, &device_GPU, build_ops.c_str(), nullptr, nullptr);

	cl_kernel kernel1 = clCreateKernel(program_GPU, "matmul_block_image", &error);
	if (checkError(error, "Create kernel1 failed")) { system("pause"); return -1.0; }

	calculations_setting();

	clReleaseProgram(program_GPU);
	clReleaseKernel(kernel1);
	clReleaseContext(context_GPU);

	//getchar();
	system("pause");

	return 0;
}
