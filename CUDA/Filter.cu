#include "Filter.h"
#include "DataStructs.h"
#include "Common.h"
#include "GPUMemoryCtrl.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

__global__
void filter_kernel(
const float *d_filter,
float * output,
const CTGeom *ctg
)
{

	extern __shared__ float s[];
	float * s_row = s;
	float * s_filter = (float*)&s_row[blockDim.x];

	float conv_pix = 0;

	int is = threadIdx.x;
	int it = blockIdx.y;
	int ia = blockIdx.x;

	int first_output_pixel = is + (it + ia*gridDim.y)*blockDim.x;

	int N = ctg->ns_();
// 	if (is == 469 && it == 40 && ia == 0)
// 		printf("N=%d\n", N);

	// Load in 2*n_pix of filter data and n_pix of row data to shared memory
	s_row[is] = output[first_output_pixel];
	s_filter[is] = d_filter[is];
	s_filter[is + N] = d_filter[is + N];

	__syncthreads(); // Make sure every thread has finished copying data into shared memory

	// Compute n_pix of the convolution    
	for (int k = 0; k < N; k++){
		conv_pix += s_filter[is - k + (N)] * s_row[k];//s_filter[l-k+(N-1)]*
// 		if (is == 469 && it == 40 && ia == 0)
// 			printf("k=%d s_filter=%f s_row=%f conv_pix=%f\n", k, s_filter[is - k + (N)], s_row[k], conv_pix);
	}

	// Copy back to global memory
	output[first_output_pixel] = conv_pix;
}

extern "C"
void Filter::filter(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	CTDataGPU *data_gpu = mr->data_gpu();

	float *h = new float[2 * ctg->ns_()];
	calculate_filter(h, ctg);
	float *d_filter;
	cudaMalloc(&d_filter, 2 * ctg->ns_()*sizeof(float));
	cudaMemcpy(d_filter, h, 2 * ctg->ns_()*sizeof(float), cudaMemcpyHostToDevice);
	delete[] h;
	int shared_size = 3 * ctg->ns_()*sizeof(float);
	filter_kernel << <dim3(ctg->n_angle(), ctg->nt_(), 1), ctg->ns_(), shared_size >> >(d_filter, data_gpu->proj_gpu()->get_pointer(), data_gpu->ctg_gpu()->get_pointer());
	cudaFree(d_filter);

	//cudaMemcpy(mr->data_cpu.rebin, data_gpu.proj_gpu, ctg.ns*ctg.nt*ctg.nAngle*sizeof(float), cudaMemcpyDeviceToHost);
	data_gpu->proj_gpu()->copy_to_CPU(mr->data_cpu()->rebin_(), ctg->ns_()*ctg->nt_()*ctg->n_angle());
	cout << "Filter finished" << endl;

	return;
}
