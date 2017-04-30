#include "DataReader.h"
#include "DataStructs.h"
#include <cstdio>
using namespace std;

#define _FILE_OFFSET_BITS 64

void RawDataReader::readData(const ReconData *mr)
{
	FILE *fp;
	string filename = mr->fi_()->in_file_dir() + mr->fi_()->data_file_name();
	fp = fopen(filename.c_str(), "rb");
	// Seek to beginning of requested frame
	size_t offset = (mr->bi_()->block_proj_start() + mr->ri_()->data_offset())*mr->ctg_()->ns_()*mr->ctg_()->nt_()*sizeof(float);
	fseek(fp, offset, SEEK_SET);

	// Fread the frame into our frame holder
	fread(mr->data_cpu()->proj_(), sizeof(float), mr->ctg_()->ns_()*mr->ctg_()->nt_()*mr->ctg_()->n_angle_add(), fp);
}
