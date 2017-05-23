#include "PreWeighting.h"
#include "DataStructs.h"
#include "GPUMemoryCtrl.h"
#include "Common.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void weighting_pi_ori_kernel(
		float *proj,
		const CTGeom *ctg
		)
{
	int is = threadIdx.x;
	int it = blockIdx.y;
	int ia = blockIdx.x;

	int pos = is+(it+ia*ctg->nt_())*ctg->ns_();

	float s = (is-ctg->ws_())*ctg->ds_();
	float t = (it-ctg->wt_())*ctg->pitch_()/(2*ctg->nt_());

	float gamma = asinf(s/ctg->dso_());
	float temp = ctg->dso_()*ctg->dso_()-s*s;
	float temp2 = t-gamma*ctg->pitch_()/(2*PI);
	float weight = sqrtf(temp)/sqrtf(temp+temp2*temp2);

	proj[pos] *= weight;
}

extern "C"
int PiOriPreWeighting::weighting(const ReconData *mr)
{
	CTGeom *ctg = mr->ctg_();
	CTDataGPU *dataGPU = mr->data_gpu();

	weighting_pi_ori_kernel<<<dim3(ctg->n_angle(), ctg->nt_(), 1), ctg->ns_()>>>(dataGPU->proj_gpu()->get_pointer(), dataGPU->ctg_gpu()->get_pointer());

	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess)
		return -1;
	else
		return 1;
}
