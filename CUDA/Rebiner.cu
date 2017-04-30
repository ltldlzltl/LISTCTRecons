#include "Rebiner.h"
#include "Common.h"
#include "DataStructs.h"
#include "GPUMemoryCtrl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>
using namespace std;

texture<float, cudaTextureType3D, cudaReadModeElementType> proj_fan;

__global__
void rebin_null_kernel(
float *proj_par,
const CTGeom *ctg
)
{
	int is = threadIdx.x;
	int ia = blockIdx.x;
	int pos = is + ia*ctg->nt_()*blockDim.x;

#pragma unroll
	for (int it = 0; it < ctg->nt_(); it++)
	{
		proj_par[pos] = tex3D(proj_fan, is + 0.5f, it + 0.5f, ia + ctg->add_projections() + 0.5f);
		pos += ctg->ns_();
	}
}

extern "C"
void NullRebinerGPU::rebin(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	CTDataGPU *data_gpu = mr->data_gpu();

	cudaArray_t proj_arr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = ctg->ns_();
	extent.height = ctg->nt_();
	extent.depth = ctg->n_angle_add();
	cudaMalloc3DArray(&proj_arr, &channelDesc, extent);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)mr->data_cpu()->proj_(), ctg->ns_()*sizeof(float), ctg->ns_(), ctg->nt_());
	copyParams.dstArray = proj_arr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	if (cudaMemcpy3D(&copyParams) != (unsigned int)CUDA_SUCCESS)
	{
		printf("[ERROR] Could not copy img to 3D Array u\n");
		return;
	}
	proj_fan.filterMode = cudaFilterModeLinear;
	proj_fan.normalized = false;
	proj_fan.channelDesc = channelDesc;
	if (cudaBindTextureToArray(&proj_fan, proj_arr, &channelDesc) != (unsigned int)CUDA_SUCCESS)
	{
		//printf("[ERROR] Could not bind texture u\n");
		cout << "[ERROR] Could not bind texture proj_fan" << endl;
		return;
	}
 	proj_fan.addressMode[0] = cudaAddressModeBorder;
	proj_fan.addressMode[1] = cudaAddressModeBorder;
	proj_fan.addressMode[2] = cudaAddressModeBorder;

	//cudaMalloc(&data_gpu.proj_gpu, ctg.ns*ctg.nt*ctg.nAngle*sizeof(float));
	data_gpu->proj_gpu()->allocate_memory(ctg->ns_()*ctg->nt_()*ctg->n_angle());
	//cudaMemcpy(data_gpu.ctg_gpu, &ctg, sizeof(CTGeom), cudaMemcpyHostToDevice);
	data_gpu->ctg_gpu()->copy_from_CPU(ctg, 1);
	rebin_null_kernel << <ctg->n_angle(), ctg->ns_() >> >(data_gpu->proj_gpu()->get_pointer(), data_gpu->ctg_gpu()->get_pointer());

 	cudaThreadSynchronize();
	cudaUnbindTexture(proj_fan);
	cudaFreeArray(proj_arr);

	cout << "Rebin finished" << endl;

	return;
}

__global__
void rebin_fan2par_kernel(
float *proj_par,
const CTGeom *ctg
)
{
	int is = threadIdx.x;
	int ia = blockIdx.x;
	int pos = is + ia*ctg->nt_()*blockDim.x;

	float ss = (is - ctg->ws_())*ctg->ds_();
	float alpha = ss / ctg->dsd_();
	float alpha_idx = is;
	float beta_idx = ia + alpha / ctg->orbit_();

#pragma unroll
	for (int it = 0; it < ctg->nt_(); it++)
	{
		proj_par[pos] = tex3D(proj_fan, alpha_idx + 0.5f, it + 0.5f, beta_idx + ctg->add_projections() + 0.5f);
		pos += ctg->ns_();
	}
}

extern "C"
void HelicalConeBeamFan2ParRebinerGPU::rebin(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	CTDataGPU *data_gpu = mr->data_gpu();

	cudaArray_t proj_arr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = ctg->ns_();
	extent.height = ctg->nt_();
	extent.depth = ctg->n_angle_add();
	cudaMalloc3DArray(&proj_arr, &channelDesc, extent);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)mr->data_cpu()->proj_(), ctg->ns_()*sizeof(float), ctg->ns_(), ctg->nt_());
	copyParams.dstArray = proj_arr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	if (cudaMemcpy3D(&copyParams) != (unsigned int)CUDA_SUCCESS)
	{
		printf("[ERROR] Could not copy img to 3D Array u\n");
		return;
	}
	proj_fan.filterMode = cudaFilterModeLinear;
	proj_fan.normalized = false;
	proj_fan.channelDesc = channelDesc;
	if (cudaBindTextureToArray(&proj_fan, proj_arr, &channelDesc) != (unsigned int)CUDA_SUCCESS)
	{
		//printf("[ERROR] Could not bind texture u\n");
		cout << "[ERROR] Could not bind texture proj_fan" << endl;
		return;
	}
 	proj_fan.addressMode[0] = cudaAddressModeBorder;
	proj_fan.addressMode[1] = cudaAddressModeBorder;
	proj_fan.addressMode[2] = cudaAddressModeBorder;

	//cudaMalloc(&data_gpu.proj_gpu, ctg.ns*ctg.nt*ctg.nAngle*sizeof(float));
	data_gpu->proj_gpu()->allocate_memory(ctg->ns_()*ctg->nt_()*ctg->n_angle());
	//cudaMemcpy(data_gpu.ctg_gpu, &ctg, sizeof(CTGeom), cudaMemcpyHostToDevice);
	data_gpu->ctg_gpu()->copy_from_CPU(ctg, 1);
	rebin_fan2par_kernel << <ctg->n_angle(), ctg->ns_() >> >(data_gpu->proj_gpu()->get_pointer(), data_gpu->ctg_gpu()->get_pointer());

	cudaThreadSynchronize();
	//data_gpu->proj_gpu()->copy_to_CPU(mr->data_cpu()->rebin_(), ctg->ns_()*ctg->nt_()*ctg->n_angle());
	cudaUnbindTexture(proj_fan);
	cudaFreeArray(proj_arr);

	cout << "Rebin finished" << endl;

	return;
}
