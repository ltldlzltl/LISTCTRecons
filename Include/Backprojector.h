#pragma once
#include "Common.h"

class ReconData;

class Backprojector
{
public:
	Backprojector(){};
	virtual ~Backprojector(){};
	virtual void backproject(const ReconData *mr) = 0;
};

class HelicalConeBeamFanRebinBackprojectorGPU:public Backprojector
{
public:
	HelicalConeBeamFanRebinBackprojectorGPU(){};
	~HelicalConeBeamFanRebinBackprojectorGPU(){};
	void backproject(const ReconData *mr);
};

class PiOriBackprojectorGPU:public Backprojector
{
public:
	PiOriBackprojectorGPU(){};
	~PiOriBackprojectorGPU(){};
	void backproject(const ReconData *mr);
};
