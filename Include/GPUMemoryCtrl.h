#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "Common.h"

template<class T>
class GPUMemoryCtrl
{
private:
	T *pData;
	size_t s;

public:
	GPUMemoryCtrl(){ s = 0; };
	~GPUMemoryCtrl()
	{
		free_memory();
	};
	bool allocate_memory(const size_t eleNum = 1)
	{
		cudaError_t err;
		if (s == 0)
		{
			err = cudaMalloc(&pData, eleNum*sizeof(T));
			if (err != cudaSuccess)
				return false;
			s = eleNum;
		}
		else
			return false;
		return true;
	};
	bool free_memory()
	{
		if (s != 0)
		{
			cudaError_t err = cudaFree(pData);
			if (err != cudaSuccess)
				return false;
			s = 0;
		}
		return true;
	};
	bool copy_from_CPU(const T *pDataCPU, const size_t eleNum, const size_t offset = 0)
	{
		if (s-offset < eleNum)
			return false;
		cudaError_t err = cudaMemcpy(pData+offset, pDataCPU, eleNum*sizeof(T), cudaMemcpyHostToDevice);
		return err == cudaSuccess;
	};
	bool copy_to_CPU(T *pDataCPU, const size_t eleNum, const size_t offset = 0)
	{
		if (s - offset < eleNum)
			return false;
		cudaError_t err = cudaMemcpy(pDataCPU, pData + offset, eleNum*sizeof(T), cudaMemcpyDeviceToHost);
		return err == cudaSuccess;
	};
	T *get_pointer(const size_t offset = 0)
	{
		return pData + offset;
	};
};
