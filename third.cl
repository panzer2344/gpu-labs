__kernel void mult(int rows, int cols , int size,const __global float* input1, const __global float* input2, __global float* output)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    double res = 0.0;
    for(k = 0; k < size; k++)
    {   
        res += input1[i*size+k]*input2[k*cols+j];
    }
    output[i*cols+j] = res;
}

__kernel void mult_opt(const int rows, const int cols , const int size, const __global float* input1, const __global float* input2, __global float* output) {
    
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int n = get_global_size(0);
    const int global_row = TS * get_group_id(0) + row;
    const int global_col = TS * get_group_id(1) + col;

    __local float A_sub[TS][TS];
    __local float B_sub[TS][TS];
    float acc = 0.0f;
    const int num_tiles = n / TS;
    for (int t = 0; t < num_tiles; t++) {
        const int tiled_row = TS * t + row;
        const int tiled_col = TS * t + col;
        A_sub[col][row] = input1[tiled_col * rows+ global_row];
        B_sub[col][row] = input2[global_col * size + tiled_row];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            acc += A_sub[k][row] * B_sub[col][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[global_col * rows+ global_row] = acc;
}

__kernel void matmul_block_image( const int rows,  const int cols,  const int size, __read_only image2d_t A,  __read_only image2d_t B, __write_only image2d_t C) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int n = get_global_size(0);
    const int global_row = TS * get_group_id(0) + row;
    const int global_col = TS * get_group_id(1) + col;

    __local float4 A_sub[TS][TS];
    __local float4 B_sub[TS][TS];

    float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    const int num_tiles = size / TS;
    for (int t = 0; t < num_tiles; t++) {
         const int tiled_row = TS * t + row;
         const int tiled_col = TS * t + col;
         const int2 id_A = { tiled_col, global_row};
         const int2 id_B = { global_col, tiled_row};
         A_sub[col][row] = read_imagef(A, id_A);
         B_sub[col][row] = read_imagef(B, id_B);
         barrier(CLK_LOCAL_MEM_FENCE);

         for (int k = 0; k < TS; k++) {
            acc += A_sub[k][row] * B_sub[col][k];
         }

         barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int2 id_C = { global_col, global_row};
    write_imagef(C, id_C, acc);
}
