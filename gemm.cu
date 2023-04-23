#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "/usr/local/include/cblas.h"

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 128
#define TILE_X_4 32
#define TILE_Y 128
#define TILE_Y_4 32

#define TILE_K 16

#define WPTN 8
#define WPTM 8
#define WPTN_4 2

__global__ void gemm_kernel_NN(const float* __restrict__ A, const float* __restrict__ B, float4* __restrict__ C, float alpha, float beta, int M, int N, int K);

inline double cpuSecond() {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void printMatrix(const float *a, int m, int n);

int main() {

	// parameter M, N, K
	const int M = 8192, K = 256, N = 8192;

	// allocate host memory
	float *h_A = (float *)malloc(sizeof(float) * M * K);
	float *h_B = (float *)malloc(sizeof(float) * K * N);
	float *h_C = (float *)malloc(sizeof(float) * M * N);
	float *h_C_1 = (float *)malloc(sizeof(float) * M * N);

	// random number
	for(int i = 0; i < M * K; ++i) {
                h_A[i] = (float)rand() / (float)(RAND_MAX) * 100;
        }
	for(int i = 0; i < K * N; ++i) {
                h_B[i] = (float)rand() / (float)(RAND_MAX) * 100;
        }
	memset(h_C, 0, M * N * sizeof(float));
	memset(h_C_1, 0, M * N * sizeof(float));

	// OpenBLAS implementation
	double h_start = cpuSecond();
	/*
	standard sgemm in blas library
	void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
	*/
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, h_A, K, h_B, N, 0, h_C, N);
	double h_finish = cpuSecond() - h_start;
	printf("Host time = %.3f.\n", h_finish * 1000);

	// verify the results
	//printMatrix(h_A, M, K);
	//printMatrix(h_B, K, N);
	//printMatrix(h_C, M, N);

	// allocate device memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, M * K * sizeof(float));
	cudaMalloc((float**)&d_B, K * N * sizeof(float));
	cudaMalloc((float**)&d_C, M * N * sizeof(float));
	cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C_1, M * N * sizeof(float), cudaMemcpyHostToDevice);
	
	// cuda implementation
	dim3 grid(N / TILE_X, M / TILE_Y), block(BLOCK_X * BLOCK_Y);
	double d_start = cpuSecond();
	gemm_kernel_NN<<<grid, block>>>(d_A, d_B, (float4 *)d_C, 1, 0, M, N, K);
	cudaDeviceSynchronize();
	double d_finish = cpuSecond() - d_start;
	printf("Device time = %.3f.\n", d_finish * 1000);
	
	// transfer results from device to host
	cudaMemcpy(h_C_1, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	// compare results 
	double error = 0;
	for(int i = 0; i < M * N; ++i) {
		error += (h_C[i] - h_C_1[i]);
	}
	printf("Error = %.3f.\n", error);
	//printf("Error = %.3f.\n", h_C[127] - h_C_1[127]);
	//printf("%.3f, %.3f.\n", h_C[127], h_C_1[127]);
	
	return 0;
}

__global__ void gemm_kernel_NN(const float* __restrict__ A, const float* __restrict__ B, float4* __restrict__ C, float alpha, float beta, int M, int N, int K) {

	/*
	BLOCK_X 16	BLOCK_Y 16

	TILE_X 128	TILE_X_4 32	TILE_Y 128	TILE_Y_4 32

	TILE_K 16

	WPTN 8	WPTM 8	WPTN_4 2
	*/

	// data in shared memory A -> smem_a, B -> smem_b, 2 means prefetch
	__shared__ float4 smem_a[2][TILE_K * TILE_Y_4];
	__shared__ float4 smem_b[2][TILE_K * TILE_X_4];

	// parameters: 
	int tx = threadIdx.x % 16;
	int ty = threadIdx.x / 16;

	int tx4 = threadIdx.x % 4;
	int ty4 = threadIdx.x / 4;

	int tx32 = threadIdx.x % 32;
	int ty32 = threadIdx.x / 32;

	// locate initial location
	// A : locate matrix address, K * TILE_Y * blockIdx.y : locate sub-matrix address, ty4 * K + tx4 * 4 : locate each thread's location
	/*
	   	16 * 16 threads load in upper part of first sub-block (64 * 16 of 128 * 16)
		order : 0, 1, 2, 3
			4, 5, 6, 7 ...
			16
	    |	?***?***?***?*** |
	    |	?***?***?***?*** |
	    |	?***?***?***?*** |
	128 |	?***?***?***?*** | 64
	    |       ........     | 
	    |	?***?***?***?*** |
	    |	?***?***?***?*** |
	    |	?***?***?***?*** |
	    |	?***?***?***?*** |
	    |		...
	    |		...
	    |		...
	*/
	const float* pA = (A + K * TILE_Y * blockIdx.y + ty4 * K + tx4 * 4);
	// B : locate matrix address, TILE_X * blockIdx.x : locate sub-matrix address, ty32 * N + tx32 * 4 : locate each thread's location
	/*
                16 * 16 threads load in upper part of first sub-block (8 * 128 of 16 * 128)
                order : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 31
                        32, 33, 34, 35, 36, 37, 38, ..., 63 ...
                        						128
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
        16  |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** | 8
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |   ?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?***?*** |
            |           ...
            |           ...
            |           ...
        */
	const float* pB = (B + TILE_X * blockIdx.x + ty32 * N + tx32 * 4);
	// C : locate matrix address, note pC pointer is float4, TILE_Y_4 * blockIdx.y * N + TILE_X_4 * blockIdx.x : locate 128 * 128 sub-matrix address
	float4* pC = C + TILE_Y_4 * blockIdx.y * N + TILE_X_4 * blockIdx.x;

	// locate each thread's location after transpose in shared memory
	int sts_a_offset = tx4 * 4 * TILE_Y + ty4;
	int sts_b_offset = ty32 * TILE_X_4 + tx32;

	// judgment of boundary condition
	float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
	bool valid_ld_a_0 = ((blockIdx.y * TILE_Y + ty4) < M) && ((tx4 * 4) < K);
	bool valid_ld_a_1 = ((blockIdx.y * TILE_Y + ty4 + 64) < M) && ((tx4 * 4) < K); 
	bool valid_ld_b_0 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && (ty32 < K);
	bool valid_ld_b_1 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && ((ty32 + 8) < K);

	// allocate register memory of first sub-block
	float4 ldg_a_reg[2];
	float4 ldg_b_reg[2];

	// load matrix A to register memory, pA and pA after line 64, pB and pB after line 8
	ldg_a_reg[0] = valid_ld_a_0 ? *(const float4*)(pA + 0 * K) : f4_zero;
	ldg_a_reg[1] = valid_ld_a_1 ? *(const float4*)(pA + 64 * K) : f4_zero;
	ldg_b_reg[0] = valid_ld_b_0 ? *(const float4*)(pB + 0 * N) : f4_zero;
	ldg_b_reg[1] = valid_ld_b_1 ? *(const float4*)(pB + 8 * N) : f4_zero;

	// c[8][2] to store temporary results
	float4 c[WPTM][WPTN_4] = { { f4_zero } };

	// transpose and load first sub-block of A from register memory to shared memory, 128 * 16 -> 16 * 128 
	/*
		1, 2
		3, 4	-> 1, 3, 5, 7 
		5, 6	   2, 4, 6, 8
		7, 8
	*/
	*((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
	*((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
	*((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
	*((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
	*((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
	*((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
	*((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
	*((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

	// load first sub-block of B from register memory to shared memory, 16 * 128 -> 16 * 128 without change 
	smem_b[0][sts_b_offset + 0] = ldg_b_reg[0];
	smem_b[0][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];

	__syncthreads();

	int i = 0;
	int write_stage_idx = 1;

	float4 reg_a[2][2];
	float4 reg_b[2][2];

	// load first sub-block from shared memory to register memory
	// a -> y axis because of transpose, b -> x axis
	reg_a[0][0] = smem_a[0][0 + ty];
	reg_a[0][1] = smem_a[0][16 + ty];
	reg_b[0][0] = smem_b[0][0 + tx];
	reg_b[0][1] = smem_b[0][16 + tx];

	do {
		// loop counter
		i += TILE_K;

		// judgment of boundary condition for next iteration
		valid_ld_a_0 = (valid_ld_a_0 && ((tx4 * 4 + i) < K));
		valid_ld_a_1 = (valid_ld_a_1 && ((tx4 * 4 + i) < K));
		valid_ld_b_0 = (valid_ld_b_0 && ((ty32 + i) < K));
		valid_ld_b_1 = (valid_ld_b_1 && ((ty32 + 8 + i) < K));

		// load next sub-block to register memory, pA and pA after line 64, pB and pB after line 8 with i
		ldg_a_reg[0] = (valid_ld_a_0) ? *(const float4*)(pA + i + 0) : f4_zero;
		ldg_a_reg[1] = (valid_ld_a_1) ? *(const float4*)(pA + i + 64 * K) : f4_zero;
		ldg_b_reg[0] = (valid_ld_b_0) ? *(const float4*)(pB + (i + 0) * N) : f4_zero;
		ldg_b_reg[1] = (valid_ld_b_1) ? *(const float4*)(pB + (i + 8) * N) : f4_zero;
		
		// ping-pong switch, 0 -> 1, 1 -> 0
		int load_stage_idx = write_stage_idx ^ 1;

		// each thread computes 2 * 2 of 4 * 4 mini-block of C
		#pragma unroll
		for (int j = 0; j < TILE_K - 1; j++) {

			// load next mini-block of current sub-block
			reg_a[(j + 1) % 2][0] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 0 + ty];
			reg_a[(j + 1) % 2][1] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 16 + ty];
			reg_b[(j + 1) % 2][0] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 0 + tx];
			reg_b[(j + 1) % 2][1] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 16 + tx];

			// computation work (gemm)
			c[0][0].x += reg_a[j % 2][0].x * reg_b[j % 2][0].x;
			c[0][0].y += reg_a[j % 2][0].x * reg_b[j % 2][0].y;
			c[0][0].z += reg_a[j % 2][0].x * reg_b[j % 2][0].z;
			c[0][0].w += reg_a[j % 2][0].x * reg_b[j % 2][0].w;
			c[0][1].x += reg_a[j % 2][0].x * reg_b[j % 2][1].x;
			c[0][1].y += reg_a[j % 2][0].x * reg_b[j % 2][1].y;
			c[0][1].z += reg_a[j % 2][0].x * reg_b[j % 2][1].z;
			c[0][1].w += reg_a[j % 2][0].x * reg_b[j % 2][1].w;
			c[1][0].x += reg_a[j % 2][0].y * reg_b[j % 2][0].x;
			c[1][0].y += reg_a[j % 2][0].y * reg_b[j % 2][0].y;
			c[1][0].z += reg_a[j % 2][0].y * reg_b[j % 2][0].z;
			c[1][0].w += reg_a[j % 2][0].y * reg_b[j % 2][0].w;
			c[1][1].x += reg_a[j % 2][0].y * reg_b[j % 2][1].x;
			c[1][1].y += reg_a[j % 2][0].y * reg_b[j % 2][1].y;
			c[1][1].z += reg_a[j % 2][0].y * reg_b[j % 2][1].z;
			c[1][1].w += reg_a[j % 2][0].y * reg_b[j % 2][1].w;
			c[2][0].x += reg_a[j % 2][0].z * reg_b[j % 2][0].x;
			c[2][0].y += reg_a[j % 2][0].z * reg_b[j % 2][0].y;
			c[2][0].z += reg_a[j % 2][0].z * reg_b[j % 2][0].z;
			c[2][0].w += reg_a[j % 2][0].z * reg_b[j % 2][0].w;
			c[2][1].x += reg_a[j % 2][0].z * reg_b[j % 2][1].x;
			c[2][1].y += reg_a[j % 2][0].z * reg_b[j % 2][1].y;
			c[2][1].z += reg_a[j % 2][0].z * reg_b[j % 2][1].z;
			c[2][1].w += reg_a[j % 2][0].z * reg_b[j % 2][1].w;
			c[3][0].x += reg_a[j % 2][0].w * reg_b[j % 2][0].x;
			c[3][0].y += reg_a[j % 2][0].w * reg_b[j % 2][0].y;
			c[3][0].z += reg_a[j % 2][0].w * reg_b[j % 2][0].z;
			c[3][0].w += reg_a[j % 2][0].w * reg_b[j % 2][0].w;
			c[3][1].x += reg_a[j % 2][0].w * reg_b[j % 2][1].x;
			c[3][1].y += reg_a[j % 2][0].w * reg_b[j % 2][1].y;
			c[3][1].z += reg_a[j % 2][0].w * reg_b[j % 2][1].z;
			c[3][1].w += reg_a[j % 2][0].w * reg_b[j % 2][1].w;
			c[4][0].x += reg_a[j % 2][1].x * reg_b[j % 2][0].x;
			c[4][0].y += reg_a[j % 2][1].x * reg_b[j % 2][0].y;
			c[4][0].z += reg_a[j % 2][1].x * reg_b[j % 2][0].z;
			c[4][0].w += reg_a[j % 2][1].x * reg_b[j % 2][0].w;
			c[4][1].x += reg_a[j % 2][1].x * reg_b[j % 2][1].x;
			c[4][1].y += reg_a[j % 2][1].x * reg_b[j % 2][1].y;
			c[4][1].z += reg_a[j % 2][1].x * reg_b[j % 2][1].z;
			c[4][1].w += reg_a[j % 2][1].x * reg_b[j % 2][1].w;
			c[5][0].x += reg_a[j % 2][1].y * reg_b[j % 2][0].x;
			c[5][0].y += reg_a[j % 2][1].y * reg_b[j % 2][0].y;
			c[5][0].z += reg_a[j % 2][1].y * reg_b[j % 2][0].z;
			c[5][0].w += reg_a[j % 2][1].y * reg_b[j % 2][0].w;
			c[5][1].x += reg_a[j % 2][1].y * reg_b[j % 2][1].x;
			c[5][1].y += reg_a[j % 2][1].y * reg_b[j % 2][1].y;
			c[5][1].z += reg_a[j % 2][1].y * reg_b[j % 2][1].z;
			c[5][1].w += reg_a[j % 2][1].y * reg_b[j % 2][1].w;
			c[6][0].x += reg_a[j % 2][1].z * reg_b[j % 2][0].x;
			c[6][0].y += reg_a[j % 2][1].z * reg_b[j % 2][0].y;
			c[6][0].z += reg_a[j % 2][1].z * reg_b[j % 2][0].z;
			c[6][0].w += reg_a[j % 2][1].z * reg_b[j % 2][0].w;
			c[6][1].x += reg_a[j % 2][1].z * reg_b[j % 2][1].x;
			c[6][1].y += reg_a[j % 2][1].z * reg_b[j % 2][1].y;
			c[6][1].z += reg_a[j % 2][1].z * reg_b[j % 2][1].z;
			c[6][1].w += reg_a[j % 2][1].z * reg_b[j % 2][1].w;
			c[7][0].x += reg_a[j % 2][1].w * reg_b[j % 2][0].x;
			c[7][0].y += reg_a[j % 2][1].w * reg_b[j % 2][0].y;
			c[7][0].z += reg_a[j % 2][1].w * reg_b[j % 2][0].z;
			c[7][0].w += reg_a[j % 2][1].w * reg_b[j % 2][0].w;
			c[7][1].x += reg_a[j % 2][1].w * reg_b[j % 2][1].x;
			c[7][1].y += reg_a[j % 2][1].w * reg_b[j % 2][1].y;
			c[7][1].z += reg_a[j % 2][1].w * reg_b[j % 2][1].z;
			c[7][1].w += reg_a[j % 2][1].w * reg_b[j % 2][1].w;
		}

		if(i < K) {
			// transpose and load sub-block of A of next iteration from register memory to shared memory, 128 * 16 -> 16 * 128
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
			*((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

			// load sub-block of B of next iteration from register memory to shared memory, 16 * 128 -> 16 * 128 without change 
			smem_b[write_stage_idx][sts_b_offset + 0] = ldg_b_reg[0];
			smem_b[write_stage_idx][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];
			__syncthreads();
			write_stage_idx ^= 1;
		}

		// load first mini-block of next sub-block
		reg_a[0][0] = smem_a[load_stage_idx ^ 1][0 + ty];
		reg_a[0][1] = smem_a[load_stage_idx ^ 1][16 + ty];
		reg_b[0][0] = smem_b[load_stage_idx ^ 1][0 + tx];
		reg_b[0][1] = smem_b[load_stage_idx ^ 1][16 + tx];

		// gemm of last iteration for current sub-block
		c[0][0].x += reg_a[1][0].x * reg_b[1][0].x;
		c[0][0].y += reg_a[1][0].x * reg_b[1][0].y;
		c[0][0].z += reg_a[1][0].x * reg_b[1][0].z;
		c[0][0].w += reg_a[1][0].x * reg_b[1][0].w;
		c[0][1].x += reg_a[1][0].x * reg_b[1][1].x;
		c[0][1].y += reg_a[1][0].x * reg_b[1][1].y;
		c[0][1].z += reg_a[1][0].x * reg_b[1][1].z;
		c[0][1].w += reg_a[1][0].x * reg_b[1][1].w;
		c[1][0].x += reg_a[1][0].y * reg_b[1][0].x;
		c[1][0].y += reg_a[1][0].y * reg_b[1][0].y;
		c[1][0].z += reg_a[1][0].y * reg_b[1][0].z;
		c[1][0].w += reg_a[1][0].y * reg_b[1][0].w;
		c[1][1].x += reg_a[1][0].y * reg_b[1][1].x;
		c[1][1].y += reg_a[1][0].y * reg_b[1][1].y;
		c[1][1].z += reg_a[1][0].y * reg_b[1][1].z;
		c[1][1].w += reg_a[1][0].y * reg_b[1][1].w;
		c[2][0].x += reg_a[1][0].z * reg_b[1][0].x;
		c[2][0].y += reg_a[1][0].z * reg_b[1][0].y;
		c[2][0].z += reg_a[1][0].z * reg_b[1][0].z;
		c[2][0].w += reg_a[1][0].z * reg_b[1][0].w;
		c[2][1].x += reg_a[1][0].z * reg_b[1][1].x;
		c[2][1].y += reg_a[1][0].z * reg_b[1][1].y;
		c[2][1].z += reg_a[1][0].z * reg_b[1][1].z;
		c[2][1].w += reg_a[1][0].z * reg_b[1][1].w;
		c[3][0].x += reg_a[1][0].w * reg_b[1][0].x;
		c[3][0].y += reg_a[1][0].w * reg_b[1][0].y;
		c[3][0].z += reg_a[1][0].w * reg_b[1][0].z;
		c[3][0].w += reg_a[1][0].w * reg_b[1][0].w;
		c[3][1].x += reg_a[1][0].w * reg_b[1][1].x;
		c[3][1].y += reg_a[1][0].w * reg_b[1][1].y;
		c[3][1].z += reg_a[1][0].w * reg_b[1][1].z;
		c[3][1].w += reg_a[1][0].w * reg_b[1][1].w;
		c[4][0].x += reg_a[1][1].x * reg_b[1][0].x;
		c[4][0].y += reg_a[1][1].x * reg_b[1][0].y;
		c[4][0].z += reg_a[1][1].x * reg_b[1][0].z;
		c[4][0].w += reg_a[1][1].x * reg_b[1][0].w;
		c[4][1].x += reg_a[1][1].x * reg_b[1][1].x;
		c[4][1].y += reg_a[1][1].x * reg_b[1][1].y;
		c[4][1].z += reg_a[1][1].x * reg_b[1][1].z;
		c[4][1].w += reg_a[1][1].x * reg_b[1][1].w;
		c[5][0].x += reg_a[1][1].y * reg_b[1][0].x;
		c[5][0].y += reg_a[1][1].y * reg_b[1][0].y;
		c[5][0].z += reg_a[1][1].y * reg_b[1][0].z;
		c[5][0].w += reg_a[1][1].y * reg_b[1][0].w;
		c[5][1].x += reg_a[1][1].y * reg_b[1][1].x;
		c[5][1].y += reg_a[1][1].y * reg_b[1][1].y;
		c[5][1].z += reg_a[1][1].y * reg_b[1][1].z;
		c[5][1].w += reg_a[1][1].y * reg_b[1][1].w;
		c[6][0].x += reg_a[1][1].z * reg_b[1][0].x;
		c[6][0].y += reg_a[1][1].z * reg_b[1][0].y;
		c[6][0].z += reg_a[1][1].z * reg_b[1][0].z;
		c[6][0].w += reg_a[1][1].z * reg_b[1][0].w;
		c[6][1].x += reg_a[1][1].z * reg_b[1][1].x;
		c[6][1].y += reg_a[1][1].z * reg_b[1][1].y;
		c[6][1].z += reg_a[1][1].z * reg_b[1][1].z;
		c[6][1].w += reg_a[1][1].z * reg_b[1][1].w;
		c[7][0].x += reg_a[1][1].w * reg_b[1][0].x;
		c[7][0].y += reg_a[1][1].w * reg_b[1][0].y;
		c[7][0].z += reg_a[1][1].w * reg_b[1][0].z;
		c[7][0].w += reg_a[1][1].w * reg_b[1][0].w;
		c[7][1].x += reg_a[1][1].w * reg_b[1][1].x;
		c[7][1].y += reg_a[1][1].w * reg_b[1][1].y;
		c[7][1].z += reg_a[1][1].w * reg_b[1][1].z;
		c[7][1].w += reg_a[1][1].w * reg_b[1][1].w;

	} while (i < K);

	// c = c * alpha
	#pragma unroll
	for (int wm = 0; wm < WPTM; wm++) {
		#pragma unroll
		for (int wn = 0; wn < WPTN_4; wn++) {
            	c[wm][wn].x *= alpha;
            	c[wm][wn].y *= alpha;
            	c[wm][wn].z *= alpha;
            	c[wm][wn].w *= alpha;
        	}
    	}

	// store upper part of 2 of 4 * 4 mini-block to d_C 
	#pragma unroll
    	for (int wm = 0; wm < WPTM / 2; wm++) {
		#pragma unroll
		for (int wn = 0; wn < WPTN_4; wn++) {
			if (((blockIdx.y * TILE_Y + ty * 4 + wm) < M) && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
				if (beta != 0) {
					float4 vec4c = *(pC + ((ty * 4 + wm) * N / 4 + wn * 16 + tx));
					vec4c.x = vec4c.x * beta + c[wm][wn].x;
					vec4c.y = vec4c.y * beta + c[wm][wn].y;
					vec4c.z = vec4c.z * beta + c[wm][wn].z;
					vec4c.w = vec4c.w * beta + c[wm][wn].w;
					*(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
				} else {
					*(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm][wn];
				}
			}
		}
    	}

	// store lower part of 2 of 4 * 4 mini-block to d_C 
	#pragma unroll
    	for (int wm = 0; wm < WPTM / 2; wm++) {
		#pragma unroll
        	for (int wn = 0; wn < WPTN_4; wn++) {
            		if (((blockIdx.y * TILE_Y + 64 + ty * 4 + wm) < M) && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
                		if (beta != 0) {
                    			float4 vec4c = *(pC + ((64 + ty * 4 + wm) * N / 4 + wn * 16 + tx));
                    			vec4c.x = vec4c.x * beta + c[wm + 4][wn].x;
                    			vec4c.y = vec4c.y * beta + c[wm + 4][wn].y;
                    			vec4c.z = vec4c.z * beta + c[wm + 4][wn].z;
                    			vec4c.w = vec4c.w * beta + c[wm + 4][wn].w;
                    			*(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                		} else {
                    			*(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm + 4][wn];
                		}
            		}
        	}
    	}
}

void printMatrix(const float *a, int m, int n) {
	for(int i = 0; i < m; ++i) {
		for(int j = 0; j < n; ++j) {
			printf("%.3f, ", a[i * n + j]);
		}
		printf("\n");
	}
}
