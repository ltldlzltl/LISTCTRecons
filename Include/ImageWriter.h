#pragma once
#include "Common.h"

class ReconData;

class ImageWriter
{
public:
	ImageWriter(){};
	virtual ~ImageWriter(){};
	virtual void writeImage(const ReconData *mr) = 0;
};

class RawImageWriter :public ImageWriter
{
public:
	RawImageWriter(){};
	~RawImageWriter(){};
	void writeImage(const ReconData *mr);
};
