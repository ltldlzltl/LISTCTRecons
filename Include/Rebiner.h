#pragma once

class ReconData;

class Rebiner
{
public:
	Rebiner(){};
	virtual ~Rebiner(){};
	virtual void rebin(const ReconData *mr) = 0;
};

class NullRebinerGPU:public Rebiner
{
public:
	NullRebinerGPU(){};
	~NullRebinerGPU(){};
	void rebin(const ReconData *mr);
};

class HelicalConeBeamFan2ParRebinerGPU:public Rebiner
{
public:
	HelicalConeBeamFan2ParRebinerGPU(){};
	~HelicalConeBeamFan2ParRebinerGPU(){};
	void rebin(const ReconData *mr);
};
