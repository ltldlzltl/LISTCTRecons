#pragma once

class Rebiner;
class Filter;
class Backprojector;
class DataReader;
class ReconData;
class PreWeighting;

class Reconstructor
{
public:
	Reconstructor(){};
	virtual ~Reconstructor(){};
	virtual void reconstruct(const ReconData *mr) = 0;
};

class FDKReconstructor: public Reconstructor
{
private:
	Rebiner *rbn;
	Filter *flt;
	Backprojector *bp;
	DataReader *dr;
	PreWeighting *pw;

public:
	FDKReconstructor(Rebiner *rbn, Filter *flt, Backprojector *bp, PreWeighting *pw);
	~FDKReconstructor();
	void reconstruct(const ReconData *mr);
};
