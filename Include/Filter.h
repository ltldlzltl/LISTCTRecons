#pragma once

class ReconData;
class CTGeom;

class Filter
{
private:
	virtual void calculate_filter(float *h, const CTGeom *ctg) = 0;

public:
	Filter(){};
	virtual ~Filter(){};
	void filter(const ReconData *mr);
};

class RampFilterGPU :public Filter
{
private:
	void calculate_filter(float *h, const CTGeom *ctg);
public:
	RampFilterGPU(){};
	~RampFilterGPU(){};
};
