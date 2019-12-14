__kernel void saxpy(int n, float a, __global float* x, __global float* y)
{
	//int i = get_global_id(0);
	//if(i < n)
	//	y[i] = y[i] + a * x[i];
}

__kernel void daxpy(int n, double a, __global double* x, __global double* y)
{
	//int i = get_global_id(0);
	//if(i < n)
	//	y[i] = y[i] + a * x[i];
}