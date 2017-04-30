#include "DataStructs.h"
#include "Common.h"
#include "FileParams.h"
#include "GPUMemoryCtrl.h"
#include <fstream>
#include <cmath>
#include <cstring>
using namespace std;

int BlockInfo::blockId = 0;

int array_search(float key, float * array, int numel_array, int search_type){
	int idx = 0;

	switch (search_type){
	case -1:{// Array descending
		while (key < array[idx] && idx < numel_array){
			idx++;
		}
		break; }
	case 0:{// Find where we're equal
		while (key != array[idx] && idx < numel_array){
			idx++;
		}
		break; }
	case 1:{// Array ascending
		while (key > array[idx] && idx < numel_array){
			idx++;
		}
		break; }
	}

	return idx;
}

CTDataCPU::CTDataCPU(const float *tubeAngle, const float *tablePosition, const int sizeProj, const int sizeRebin, const int sizeImage, const int nReadings)
{
	this->proj = new float[sizeProj]();
	this->rebin = new float[sizeRebin]();
	this->image = new float[sizeImage]();
	this->tubeAngles = new float[nReadings]();
	this->tablePositions = new float[nReadings]();

	memcpy(tubeAngles, tubeAngle, nReadings*sizeof(float));
	memcpy(tablePositions, tablePosition, nReadings*sizeof(float));
}

CTDataCPU::~CTDataCPU()
{
	delete[] this->proj;
	delete[] this->rebin;
	delete[] this->image;
	delete[] this->tubeAngles;
	delete[] this->tablePositions;
}

CTDataGPU::CTDataGPU(){
	ctgGpu = new GPUMemoryCtrl<CTGeom>();
	igGpu = new GPUMemoryCtrl<ImgGeom>();
	projGpu = new GPUMemoryCtrl<float>;
	imgGpu = new GPUMemoryCtrl<float>();
	betasGpu = new GPUMemoryCtrl<float>();

	ctgGpu->allocate_memory();
	igGpu->allocate_memory();
};

CTDataGPU::~CTDataGPU()
{
	this->projGpu->free_memory();
	this->imgGpu->free_memory();
	this->betasGpu->free_memory();
	this->ctgGpu->free_memory();
	this->igGpu->free_memory();
}

void BlockInfo::update(const ReconInfo *ri, const CTGeom *ctg, const ImgGeom *ig, const CTDataCPU *dataCPU)
{
	blockId++;

	/* --- Figure out how many and which projections to grab --- */
	int reconDirection = fabs(ri->end_pos() - ri->start_pos()) / (ri->end_pos() - ri->start_pos());
	if (reconDirection != 1 && reconDirection != -1) // user requests one slice (end_pos==start_pos)
		reconDirection = 1;

	float blockStart = ri->recon_start_pos() + reconDirection*blockId*ig->dz_()*ri->n_slices_block();
	float blockEnd = blockStart + reconDirection*(ri->n_slices_block() - 1)*ig->dz_();
	int array_direction = ctg->z_dir();
	int blockProjStart = array_search(blockStart, dataCPU->table_positions(), ri->n_readings(), array_direction);
	int blockProjEnd = array_search(blockEnd, dataCPU->table_positions(), ri->n_readings(), array_direction);

	if (blockProjStart > blockProjEnd)
		blockProjStart = blockProjEnd - ctg->n_proj_turn() / 2 - ctg->add_projections();
	else
		blockProjStart = blockProjStart - ctg->n_proj_turn() / 2 - ctg->add_projections();
	blockProjEnd = blockProjStart + ctg->n_angle_add();

	// copy this info into our recon metadata
	this->blockStart = blockStart;
	this->blockEnd = blockEnd;
	this->blockSliceStart = BLOCK_SLICES*blockId;
	this->blockSliceEnd = blockSliceStart + BLOCK_SLICES - 1;

	this->blockProjStart = blockProjStart;
	this->blockProjEnd = blockProjEnd;

	return;
}

ReconData::~ReconData()
{
	delete this->bi;
	delete this->ig;
	delete this->ctg;
	delete this->dataCPU;
	delete this->dataGPU;
	delete this->fi;
	delete this->ri;
}

void ReconData::convert_from_file_params(const FileParams &fp)
{
	int addProjections = fp.ns()*fp.ds()*fp.n_proj_turn() / (4.0f*fp.r_src_to_det()*PI) + 10;

	// Allocate memory
	float *tubeAngles = new float[fp.readings()];
	float *tablePositions = new float[fp.readings()];

	// Examine if data file exists
	string datapath = fp.raw_file_dir() + fp.data_interp_file();
	fstream rawFile(datapath.c_str(), ios::in|ios::binary);
	if (!rawFile.is_open()){
		perror("Raw data file not found.");
		exit(1);
	}
	rawFile.close();

	// Read view angle file
	string anglepath = fp.raw_file_dir() + fp.view_angle_file();
	fstream angleFile(anglepath.c_str(), ios::in|ios::binary);
	if (!angleFile.is_open()){
		perror("Raw data file not found.");
		exit(1);
	}
	char *p = new char[(fp.readings() + fp.view_offset())*sizeof(float)];
	angleFile.read(p, (fp.readings() + fp.view_offset())*sizeof(float));
	memcpy(tubeAngles, p + fp.view_offset()*sizeof(float), fp.readings()*sizeof(float));
	angleFile.close();

	// Read couch pos file
	string pospath = fp.raw_file_dir() + fp.couch_pos_file();
	fstream posFile(pospath.c_str(), ios::in|ios::binary);
	if (!posFile.is_open()){
		perror("Raw data file not found.");
		exit(1);
	}

	posFile.read(p, (fp.readings() + fp.view_offset())*sizeof(float));
	memcpy(tablePositions, p + fp.view_offset()*sizeof(float), fp.readings()*sizeof(float));
	posFile.close();
	delete[] p;

	/* --- Figure out how many and which projections to grab --- */
	int nSlicesBlock = BLOCK_SLICES;
	int nz = BLOCK_SLICES;

	//float recon_start_pos=rp.start_pos;
	//float recon_end_pos=rp.start_pos+recon_direction*(n_slices_recon-1)*rp.coll_slicewidth;
	int arrayDirection = fabs(tablePositions[100] - tablePositions[0]) / (tablePositions[100] - tablePositions[0]);
	int zDir = arrayDirection;
	float orbitStart = tubeAngles[0], orbit;
	if ((tubeAngles[1] < 1 && tubeAngles[0]>6) || (tubeAngles[1] > 6 && tubeAngles[0] < 1))
		orbit = (tubeAngles[2] - tubeAngles[1]);
	else
		orbit = (tubeAngles[1] - tubeAngles[0]);
	float sourceZ0 = -fp.pitch_value() / 2.0f;

	// Decide if the user has requested a valid range for reconstruction
	float dataBeginPos = tablePositions[0];
	float dataEndPos = tablePositions[fp.readings() - 1];
	float projection_padding = fp.pitch_value() * (fp.n_proj_turn() / 2 + addProjections + 256) / fp.n_proj_turn();
	float allowedBegin = dataBeginPos + arrayDirection*projection_padding;
	float allowedEnd = dataEndPos - arrayDirection*projection_padding;

	float startPos, endPos;
	if (arrayDirection > 0)
	{
		startPos = ceil(allowedBegin + 1 * arrayDirection);
		endPos = floor(allowedEnd - 1 * arrayDirection);
	}
	else
	{
		startPos = floor(allowedBegin + 1 * arrayDirection);
		endPos = ceil(allowedEnd - 1 * arrayDirection);
	}
	int reconDirection = fabs(endPos - startPos) / (endPos - startPos);
	if (reconDirection != 1 && reconDirection != -1) // user request one slice (end_pos==start_pos)
		reconDirection = 1;

	float reconStartPos = startPos - reconDirection*fp.dz();
	float reconEndPos = endPos + reconDirection*fp.dz();//rp.start_pos+recon_direction*(n_slices_recon-1)*rp.coll_slicewidth;

	int nSlicesRequested = floor(fabs(reconEndPos - reconStartPos) / fp.dz()) + 1;//floor(fabs(rp.end_pos-rp.start_pos)/rp.coll_slicewidth)+1;
	int nSlicesRecon = (nSlicesRequested - 1) - (nSlicesRequested - 1) % nSlicesBlock;

	reconEndPos = reconStartPos + reconDirection*(nSlicesRecon - 1)*fp.dz();

	int nBlocks = nSlicesRecon / nSlicesBlock;

	if (((startPos > allowedBegin) && (startPos > allowedEnd)) || ((startPos < allowedBegin) && (startPos < allowedEnd))){
		printf("Requested reconstruction is outside of allowed data range: %.2f to %.2f\n", allowedBegin, allowedEnd);
		exit(1);
	}

	if (((endPos > allowedBegin) && (endPos > allowedEnd)) || ((endPos < allowedBegin) && (endPos < allowedEnd))){
		printf("Requested reconstruction is outside of allowed data range: %.2f to %.2f\n", allowedBegin, allowedEnd);
		exit(1);
	}

	printf("Reconstructing couch position from %f to %f, %d slices in total.\n", reconStartPos, reconEndPos, nSlicesRecon);

	float blockStart = reconStartPos;
	float blockEnd = blockStart + reconDirection*(nSlicesBlock - 1)*fp.dz();
	int blockProjStart = array_search(blockStart, tablePositions, fp.readings(), arrayDirection);
	int blockProjEnd = array_search(blockEnd, tablePositions, fp.readings(), arrayDirection);
	int blockSliceStart = 0;
	int blockSliceEnd = nSlicesBlock - 1;

	if (blockProjStart > blockProjEnd)
	{
		blockProjStart = blockProjEnd - fp.n_proj_turn() / 2 - addProjections;
		blockProjEnd = blockProjStart + fp.n_proj_turn() / 2 + addProjections;
	}
	else
	{
		blockProjStart = blockProjStart - fp.n_proj_turn() / 2 - addProjections;
		blockProjEnd = blockProjEnd + fp.n_proj_turn() / 2 + addProjections;
	}
	int nAngleAdd = blockProjEnd - blockProjStart + 100;
	int nAngle = nAngleAdd - 2 * addProjections;

	this->ctg = new CTGeom(orbit, orbitStart, fp.ns(), fp.nt(), nAngle, nAngleAdd, fp.ds(), fp.dt(), fp.central_channel(), (fp.nt() - 1) / 2.0f, fp.pitch_value(), sourceZ0, fp.r_src_to_det(), fp.r_src_to_iso(), fp.n_proj_turn(), zDir, addProjections);
	this->ig = new ImgGeom(fp.nx(), fp.ny(), nz, fp.dx(), fp.dx(), fp.dz(), fp.offset_x(), fp.offset_y(), 0);
	this->fi = new FileInfo(fp.raw_file_dir(), fp.data_interp_file(), fp.output_dir(), fp.output_name());
	this->ri = new ReconInfo(fp.readings(), dataBeginPos, dataEndPos, allowedBegin, allowedEnd, startPos, endPos, reconStartPos, reconEndPos, nSlicesRecon, nSlicesBlock, nBlocks, fp.data_offset());
	this->dataCPU = new CTDataCPU(tubeAngles, tablePositions, ctg->ns_()*ctg->nt_()*ctg->n_angle_add(), ctg->ns_()*ctg->nt_()*ctg->n_angle(), ig->nx_()*ig->ny_()*ri->n_slices_recon(), fp.readings());
	this->dataGPU = new CTDataGPU();
	dataGPU->betas_gpu()->allocate_memory(fp.readings());
	this->bi = new BlockInfo(blockProjStart, blockProjEnd, blockStart, blockEnd, blockSliceStart, blockSliceEnd);
}
