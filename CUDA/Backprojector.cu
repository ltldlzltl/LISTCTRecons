#include "Backprojector.h"
#include "DataStructs.h"
#include "GPUMemoryCtrl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>
using namespace std;

texture<float, cudaTextureType3D, cudaReadModeElementType> proj_par;

__global__
void backprojection_cbct_par_kernel(
float *img,
const float *betas,
const CTGeom *ctg,
const ImgGeom *ig,
const int ia_begin,
const int ia_step,
const int na1_half,
const float source_dz_per_view
)
{
	int iz = blockIdx.y;
	int iy = blockIdx.x;
	int ix = threadIdx.x;

	__shared__ CTGeom ctg_c;
	ctg_c = *ctg;
	__shared__ ImgGeom ig_c;
	ig_c = *ig;

	float xc = (ix - ig_c.wx_())*ig_c.dx_();	//x
	float yc = (iy - ig_c.wy_())*ig_c.dy_();	//y
	float zc = (iz - ig_c.wz_())*ig_c.dz_();	//z

	int a_middle = roundf((zc - ctg_c.source_z0()) / source_dz_per_view);	//nearest focal point
	a_middle = a_middle < 0 ? 0 : a_middle;
	a_middle = a_middle > ctg_c.n_angle()- 1 ? ctg_c.n_angle() - 1 : a_middle;
	int a_min = (a_middle - na1_half >= 0) ? (a_middle - na1_half) : 0;
	int a_max = (a_middle + na1_half < ctg_c.n_angle()) ? (a_middle + na1_half) : ctg_c.n_angle() - 1;
	a_min = a_min - a_min%ia_step;
	float proj_sum = 0.0f;
	// 	if (ix == 255 && iy == 255)
	// 			printf("iz=%d dz=%f a_middle=%d a_min=%d a_max=%d\n", iz,ig_c.dz,a_middle,a_min,a_max);
	for (int ia = a_min + ia_begin; ia < a_max; ia += ia_step)
	{
		float beta = betas[ia];
		int ia2 = ia + na1_half < a_max ? ia + na1_half : ia - na1_half;
		ia2 = ia2 < 0 ? ia + na1_half : ia2;
		float beta2 = betas[ia2];

		float phat = -xc*__cosf(beta) - yc*__sinf(beta);
		float theta = asinf(phat / ctg_c.dso_());
		float delta_p = theta*ctg_c.dso_() / ctg_c.ds_();
		float p_idx = delta_p + ctg_c.ws_();
		float p_idx2 = -delta_p + ctg_c.ws_();

		float z_ray = ctg_c.source_z0() + (ia + theta / ctg_c.orbit_())*source_dz_per_view;
		float z_ray2 = ctg_c.source_z0() + (ia2 - theta / ctg_c.orbit_())*source_dz_per_view;
		float lhat = sqrtf(ctg_c.dso_()*ctg_c.dso_() - phat*phat) + xc*__sinf(beta) - yc * __cosf(beta);
		float lhat2 = sqrtf(ctg_c.dso_()*ctg_c.dso_() - phat*phat) + xc*__sinf(beta2) - yc * __cosf(beta2);
		float qhat = (zc - z_ray)*ctg_c.dso_()*ctg_c.z_dir() / (lhat*ctg_c.dt_());
		float qhat2 = (zc - z_ray2)*ctg_c.dso_()*ctg_c.z_dir() / (lhat2*ctg_c.dt_());
		float q_idx = qhat + ctg_c.wt_();
		float q_idx2 = qhat2 + ctg_c.wt_();

		// 		if (ix == 50 && iy == 255 && iz == 20)
		// 			printf("ia=%d beta=%f phat=%f qhat=%f\n", ia, beta, p_idx, q_idx);

		float ww, ww2;
		if (abs(qhat) > ctg_c.wt_())
		{
			ww = 0.0f;
			ww2 = 1.0f;
		}
		else if (abs(qhat2) > ctg_c.wt_())
		{
			ww = 1.0f;
			ww2 = 0.0f;
		}
		else
		{
			ww = -qhat2 / (qhat - qhat2);
			ww2 = qhat / (qhat - qhat2);
		}
		float addition1 = tex3D(proj_par, p_idx + 0.5f, q_idx + 0.5f, ia + 0.5f)*ww;
		float addition2 = tex3D(proj_par, p_idx2 + 0.5f, q_idx2 + 0.5f, ia2 + 0.5f)*ww2;
		// 		if (ix == 88 && iy == 56 && iz == 20)
		// 			printf("ia=%d beta=%f p_idx=%f q_idx=%f addition1=%f\n", ia,beta, p_idx,q_idx,addition1);
		// 		if (ix == 88 && iy == 56 && iz == 20)
		// 			printf("ia2=%d p_idx2=%f q_idx2=%f addition2=%f\n", ia2, p_idx2, q_idx2, addition2);
		proj_sum += addition1 + addition2;

	}
	img[ix + (iy + iz*ig_c.ny_())*ig_c.nx_()] = proj_sum * 2.0f * PI / (float)(a_max - a_min);
	// 	if (ix == 345 && iy == 123 && iz == 20)
	// 		printf("ix=%d iy=%d img=%f\n", ix, iy, img[ix + (iy + iz*ig_c.ny)*ig_c.nx]
}

extern "C"
void HelicalConeBeamFanRebinBackprojectorGPU::backproject(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	ImgGeom *ig = mr->ig_();
	CTDataGPU *dataGpu = mr->data_gpu();

	cudaArray_t proj_arr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = ctg->ns_();
	extent.height = ctg->nt_();
	extent.depth = ctg->n_angle();
	cudaMalloc3DArray(&proj_arr, &channelDesc, extent);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)dataGpu->proj_gpu()->get_pointer(), ctg->ns_()*sizeof(float), ctg->ns_(), ctg->nt_());
	copyParams.dstArray = proj_arr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	if (cudaMemcpy3D(&copyParams) != (unsigned int)CUDA_SUCCESS)
	{
		printf("[ERROR] Could not copy proj_gpu to 3D Array proj_arr\n");
		return;
	}
	proj_par.filterMode = cudaFilterModeLinear;
	proj_par.normalized = false;
	proj_par.channelDesc = channelDesc;
	if (cudaBindTextureToArray(&proj_par, proj_arr, &channelDesc) != (unsigned int)CUDA_SUCCESS)
	{
		//printf("[ERROR] Could not bind texture u\n");
		cout << "[ERROR] Could not bind texture proj_fan" << endl;
		return;
	}
	proj_par.addressMode[0] = cudaAddressModeBorder;
	proj_par.addressMode[1] = cudaAddressModeBorder;
	proj_par.addressMode[2] = cudaAddressModeBorder;
	//cudaFree(data_gpu->proj_gpu);
	dataGpu->proj_gpu()->free_memory();
	//cudaMalloc(&dataGpu.img_gpu, ig.nx*ig.ny*ig.nz*sizeof(float));
	dataGpu->img_gpu()->allocate_memory(ig->nx_()*ig->ny_()*ig->nz_());

	// 	for (int i = 0; i < ctg.nAngle; i++)
	// 		printf("i=%d beta[i]=%f\n", i, mr->tube_angles[mr->ri.idx_pull_start + ctg.add_projections + i]);
	//cudaMemcpy(dataGpu.betas_gpu, &(mr->tube_angles[mr->ri.block_proj_start + ctg.add_projections]), ctg.nAngle*sizeof(float), cudaMemcpyHostToDevice);
	dataGpu->betas_gpu()->copy_from_CPU(&(mr->data_cpu()->tube_angles()[mr->bi_()->block_proj_start() + ctg->add_projections()]), ctg->n_angle());
	//cudaMemcpy(dataGpu.ig_gpu, &ig, sizeof(ImgGeom), cudaMemcpyHostToDevice);
	dataGpu->ig_gpu()->copy_from_CPU(ig, 1);
	int na1Half = ctg->n_proj_turn() / 2;
	float sourceDzPerView = ctg->pitch_() / ctg->n_proj_turn();

	backprojection_cbct_par_kernel << <dim3(ig->ny_(), ig->nz_(), 1), ig->nx_() >> >(dataGpu->img_gpu()->get_pointer(), dataGpu->betas_gpu()->get_pointer(), dataGpu->ctg_gpu()->get_pointer(), dataGpu->ig_gpu()->get_pointer(), 0, 1, na1Half, sourceDzPerView);

	cudaThreadSynchronize();

	cout << "backprojection finished." << endl;

	//cudaMemcpy(&(mr->data_cpu.image[ig.nx*ig.ny*ig.nz*(mr->ri.cb.block_idx - 1)]), dataGpu.img_gpu, ig.nx*ig.ny*ig.nz*sizeof(float), cudaMemcpyDeviceToHost);
	dataGpu->img_gpu()->copy_to_CPU(&mr->data_cpu()->image_()[ig->nx_()*ig->ny_()*ig->nz_()*mr->bi_()->block_idx()], ig->nx_()*ig->ny_()*ig->nz_());


	// 	for (int i = 0; i < ig.nx*ig.ny*ig.nz; i++)
	// 		cout << mr->data_cpu.image[ig.nx*ig.ny*ig.nz*(mr->ri.cb.block_idx - 1) + i] << " ";

	cudaUnbindTexture(proj_par);
	cudaFreeArray(proj_arr);
	//cudaFree(dataGpu.img_gpu);
	dataGpu->img_gpu()->free_memory();
	return;
}

__global__
void backprojection_cbct_pi_ori_kernel(
		float *img,
		const float *betas,
		const CTGeom *ctg,
		const ImgGeom *ig,
		const int na1_half,
		const float source_dz_per_view
		)
{
	int iz = blockIdx.y;
	int iy = blockIdx.x;
	int ix = threadIdx.x;

	__shared__ CTGeom ctg_c;
	ctg_c = *ctg;
	__shared__ ImgGeom ig_c;
	ig_c = *ig;

	float xc = (ix - ig_c.wx_())*ig_c.dx_();	//x
	float yc = (iy - ig_c.wy_())*ig_c.dy_();	//y
	float zc = (iz - ig_c.wz_())*ig_c.dz_();	//z

	int a_middle = roundf((zc - ctg_c.source_z0()) / source_dz_per_view);	//nearest focal point
	a_middle = a_middle < 0 ? 0 : a_middle;
	a_middle = a_middle > ctg_c.n_angle()- 1 ? ctg_c.n_angle() - 1 : a_middle;
	float proj_sum = 0.0f;

	int ia;
	for(ia = a_middle;ia >= 0;ia--)
	{
		float beta = betas[ia];
		float cb = __cosf(beta);
		float sb = __sinf(beta);

		float s = -xc*cb-yc*sb;
		float gamma = asinf(s/ctg_c.dso_());
		float temp1 = sqrtf(ctg_c.dso_()*ctg_c.dso_()-s*s);
		float temp2 = ctg_c.pitch_()/(2*PI);
		float zOff = zc-(ctg_c.source_z0()+source_dz_per_view*ia);

		float t = temp1*(zOff-gamma*temp2)/(temp1+xc*sb-yc*cb)+gamma*temp2;
		t *= ctg_c.z_dir();
		if(t > ctg_c.pitch_()/4 || t < -ctg_c.pitch_()/4)
			break;

		float is = s/ctg_c.ds_()+ctg_c.ws_();
		float it = t*2*(ctg_c.nt_()-1)/ctg_c.pitch_() + ctg_c.wt_();

		float addition = tex3D(proj_par, is+0.5f, it+0.5f, ia+0.5f);
		proj_sum += addition;
	}

	int a_min = ia;
	for(ia = a_middle+1; ia <= a_min+na1_half;ia++)
	{
		float beta = betas[ia];
		float cb = __cosf(beta);
		float sb = __sinf(beta);

		float s = -xc*cb-yc*sb;
		float gamma = asinf(s/ctg_c.dso_());
		float temp1 = sqrtf(ctg_c.dso_()*ctg_c.dso_()-s*s);
		float temp2 = ctg_c.pitch_()/(2*PI);
		float zOff = zc-(ctg_c.source_z0()+source_dz_per_view*ia);

		float t = temp1*(zOff-gamma*temp2)/(temp1+xc*sb-yc*cb)+gamma*temp2;
		t *= ctg_c.z_dir();

		float is = s/ctg_c.ds_()+ctg_c.ws_();
		float it = t*2*(ctg_c.nt_()-1)/ctg_c.pitch_() + ctg_c.wt_();

		float addition = tex3D(proj_par, is+0.5f, it+0.5f, ia+0.5f);
		proj_sum += addition;
	}
	//if(iz == 15 && ix <= 255 && iy <= 255)
		//printf("ix=%d iy=%d view number=%d\n",ix, iy, ia-a_min);

	img[ix+(iy+iz*ig_c.ny_())*ig_c.nx_()] = proj_sum*PI/na1_half;
}

extern "C"
void PiOriBackprojectorGPU::backproject(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	ImgGeom *ig = mr->ig_();
	CTDataGPU *dataGpu = mr->data_gpu();

	cudaArray_t proj_arr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = ctg->ns_();
	extent.height = ctg->nt_();
	extent.depth = ctg->n_angle();
	cudaMalloc3DArray(&proj_arr, &channelDesc, extent);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)dataGpu->proj_gpu()->get_pointer(), ctg->ns_()*sizeof(float), ctg->ns_(), ctg->nt_());
	copyParams.dstArray = proj_arr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	if (cudaMemcpy3D(&copyParams) != (unsigned int)CUDA_SUCCESS)
	{
		printf("[ERROR] Could not copy proj_gpu to 3D Array proj_arr\n");
		return;
	}
	proj_par.filterMode = cudaFilterModeLinear;
	proj_par.normalized = false;
	proj_par.channelDesc = channelDesc;
	if (cudaBindTextureToArray(&proj_par, proj_arr, &channelDesc) != (unsigned int)CUDA_SUCCESS)
	{
		cout << "[ERROR] Could not bind texture proj_fan" << endl;
		return;
	}
	proj_par.addressMode[0] = cudaAddressModeBorder;
	proj_par.addressMode[1] = cudaAddressModeBorder;
	proj_par.addressMode[2] = cudaAddressModeBorder;
	dataGpu->proj_gpu()->free_memory();
	dataGpu->img_gpu()->allocate_memory(ig->nx_()*ig->ny_()*ig->nz_());

	dataGpu->betas_gpu()->copy_from_CPU(&(mr->data_cpu()->tube_angles()[mr->bi_()->block_proj_start() + ctg->add_projections()]), ctg->n_angle());
	dataGpu->ig_gpu()->copy_from_CPU(ig, 1);
	int na1Half = ctg->n_proj_turn() / 2;
	float sourceDzPerView = ctg->pitch_() / ctg->n_proj_turn();

	backprojection_cbct_pi_ori_kernel << <dim3(ig->ny_(), ig->nz_(), 1), ig->nx_() >> >(dataGpu->img_gpu()->get_pointer(), dataGpu->betas_gpu()->get_pointer(), dataGpu->ctg_gpu()->get_pointer(), dataGpu->ig_gpu()->get_pointer(), na1Half, sourceDzPerView);

	cudaThreadSynchronize();

	cout << "backprojection finished." << endl;

	dataGpu->img_gpu()->copy_to_CPU(&mr->data_cpu()->image_()[ig->nx_()*ig->ny_()*ig->nz_()*mr->bi_()->block_idx()], ig->nx_()*ig->ny_()*ig->nz_());

	cudaUnbindTexture(proj_par);
	cudaFreeArray(proj_arr);
	dataGpu->img_gpu()->free_memory();
	return;
}
