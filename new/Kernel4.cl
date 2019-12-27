__kernel void jacobi(__global float * A, __global float * b, __global float * x_old, __global float * x_new) 
{
    const size_t i = get_global_id(0);
    const size_t size = get_global_size(0);

    float acc = 0.0f;
    for (size_t j = 0; j < size; j++) {
        acc += A[j * size + i] * x_old[j] * (float)(i != j);
    }
    x_new[i] = (b[i] - acc) / A[i * size + i];
}
