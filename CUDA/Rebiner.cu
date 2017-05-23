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
	float alpha = ss / ctg->dso_();
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

__global__
void rebin_pi_ori_kernel(
        float *proj,
        const CTGeom *ctg
        )
{
    int ia = blockIdx.x;
    int is = threadIdx.x;
    int it = blockIdx.y;
    int pos = is+(it+ia*ctg->nt_())*ctg->ns_();

    float s = (is-ctg->ws_())*ctg->ds_();
    float gamma = asinf(s/ctg->dso_());
    float pIdx = gamma*ctg->dso_()/ctg->ds_() + ctg->ws_();
    
    float q = ((it-ctg->wt_())*ctg->pitch_()/(2*(ctg->nt_()-1))-gamma*ctg->pitch_()/(2*PI)) / __cosf(gamma);
    float qIdx = q/ctg->dt_()+ctg->wt_();

    float betaIdx = ia+gamma/ctg->orbit_() + ctg->add_projections();
    /*if(is == 450 && it == 40)
    	printf("ia=%d,s=%f,gamma=%f,pIdx=%f,q=%f,qIdx=%f,betaIdx=%f\n",ia,s,gamma,pIdx,q,qIdx,betaIdx);*/

    proj[pos] = tex3D(proj_fan, pIdx+0.5f, qIdx+0.5f, betaIdx+0.5f);
}

extern "C"
void PiOriRebiner::rebin(const ReconData *mr)
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
	rebin_pi_ori_kernel << <dim3(ctg->n_angle(), ctg->nt_(), 1), ctg->ns_() >> >(data_gpu->proj_gpu()->get_pointer(), data_gpu->ctg_gpu()->get_pointer());

	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess)
		cout<<"[Error] Kernel error"<<endl;
	//data_gpu->proj_gpu()->copy_to_CPU(mr->data_cpu()->rebin_(), ctg->ns_()*ctg->nt_()*ctg->n_angle());
	//data_gpu->proj_gpu()->copy_to_CPU(mr->data_cpu()->rebin_(), ctg->ns_()*ctg->nt_()*ctg->n_angle());
	cudaUnbindTexture(proj_fan);
	cudaFreeArray(proj_arr);

	cout << "Rebin finished" << endl;
	return;
}
