#pragma once
#include "Common.h"

class ReconData;

class DataReader
{
public:
	DataReader(){};
	virtual ~DataReader(){};
	virtual void readData(const ReconData *mr) = 0;
};

class RawDataReader: public DataReader
{
public:
	RawDataReader(){};
	~RawDataReader(){};
	void readData(const ReconData *mr);
};
