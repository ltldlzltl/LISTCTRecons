#pragma once

class ReconData;

class PreWeighting
{
public:
	PreWeighting(){};
	virtual ~PreWeighting(){};
	virtual int weighting(const ReconData *mr) = 0;
};

class NULLPreWeighting:public PreWeighting
{
public:
	NULLPreWeighting(){};
	~NULLPreWeighting(){};
	int weighting(const ReconData *mr){ return 1; };
};

class PiOriPreWeighting:public PreWeighting
{
public:
	PiOriPreWeighting(){};
	~PiOriPreWeighting(){};
	int weighting(const ReconData *mr);
};
