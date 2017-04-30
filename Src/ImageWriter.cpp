#include "ImageWriter.h"
#include "DataStructs.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;

void RawImageWriter::writeImage(const ReconData *mr)
{
	string filename = mr->fi_()->output_dir() + mr->fi_()->output_name();
	fstream file(filename.c_str(), ios::out | ios::binary);
	if (!file.is_open())
	{
		cout << "Opening image file " << mr->fi_()->output_dir() << mr->fi_()->output_name() << " failed." << endl;
		exit(1);
	}

	cout << "Writing image to " << mr->fi_()->output_dir() << mr->fi_()->output_name() << endl;
	int imageSize = mr->ig_()->nx_()*mr->ig_()->ny_()*mr->ri_()->n_slices_recon()*sizeof(float);
	char *p = new char[imageSize];
	memcpy(p, mr->data_cpu()->image_(), imageSize);
	file.write(p, imageSize);
	file.close();
	delete[] p;

	string filename2 = mr->fi_()->output_dir()+"rebin.raw";
	fstream file2(filename2.c_str(), ios::out | ios::binary);
	if (!file2.is_open())
	{
		cout << "Opening image file " << mr->fi_()->output_dir() << "rebin.raw failed." << endl;
		exit(1);
	}

	cout << "Writing image to " << mr->fi_()->output_dir() << "rebin.raw" << endl;
	int rebinSize = mr->ctg_()->ns_()*mr->ctg_()->nt_()*mr->ctg_()->n_angle()*sizeof(float);
	char *p2 = new char[rebinSize];
	memcpy(p2, mr->data_cpu()->rebin_(), rebinSize);
	file2.write(p2, rebinSize);
	file2.close();
	cout << "Writing finished." << endl;
	delete[] p2;
}
