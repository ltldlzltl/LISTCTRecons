#pragma once
#include <string>

class Reconstructor;

class ReconstrctorAbstractFactory
{
public:
	ReconstrctorAbstractFactory(){};
	~ReconstrctorAbstractFactory(){};
	Reconstructor *createReconstructor(const std::string &str);
};
