#pragma once
#include <string>
#include <cuda_runtime.h> 

class FileParams;
template<class T>
class GPUMemoryCtrl;

class CTGeom
{
private:
	float orbit, orbitStart;	//����view����ת�ĽǶȣ���0��view�ĽǶ�
	int ns, nt;	//ͶӰ���width��height
	float ws, wt;	//ͶӰ���ԭ��λ��
	int nAngle, nAngleAdd;	//��ͶӰ�õ���view������par2fan_rebin��Ҫ��������view��������Щview���ܹ���view��
	float ds, dt;	//ͶӰ��ݼ��
	float pitch;	//����Դ��������ת��ͬһ�Ƕ�ʱ��z�����ϵľ���
	float sourceZ0;	//��0��view������Դ���0��ͼ��֮��ľ���
	float dsd, dso;	//����Դ��̽�����ľ��룬����Դ���������ϵԭ��ľ���
	int nProjTurn;	//����Դ��תһȦ��view��
	int zDir;		//couch_pos�ı䷽��1��-1
	int addProjections;	//rebin����Ķ����view��nAngleAdd=nAngle+2*add_projections

public:
	CTGeom(){};
	CTGeom(const float orbit, const float orbitStart, const int ns, const int nt, const int na, const int naa, const float ds, const float dt, const float ws, const float wt, const float pitch, const float sourceZ, const float dsd, const float dso, const int nProjTurn, const int zDir, const int addProjections)
	{
		this->orbit = orbit;
		this->orbitStart = orbitStart;
		this->ns = ns;
		this->nt = nt;
		this->nAngle = na;
		this->ds = ds;
		this->dt = dt;
		this->pitch = pitch;
		this->sourceZ0 = sourceZ;
		this->dsd = dsd;
		this->dso = dso;

		this->ws = ws;
		this->wt = wt;

		this->nProjTurn = nProjTurn;
		this->zDir = zDir;
		this->addProjections = addProjections;
		this->nAngleAdd = naa;
	}
	__device__ __host__
		CTGeom &operator=(const CTGeom & ctg)
	{
		this->orbit = ctg.orbit;
		this->orbitStart = ctg.orbitStart;
		this->ns = ctg.ns;
		this->nt = ctg.nt;
		this->nAngle = ctg.nAngle;
		this->ds = ctg.ds;
		this->dt = ctg.dt;
		this->pitch = ctg.pitch;
		this->sourceZ0 = ctg.sourceZ0;
		this->dsd = ctg.dsd;
		this->dso = ctg.dso;

		this->ws = ctg.ws;
		this->wt = ctg.wt;
		this->nProjTurn = ctg.nProjTurn;
		this->zDir = ctg.zDir;
		this->addProjections = ctg.addProjections;

		return *this;
	}
	__device__ __host__
		float orbit_() const { return this->orbit; };
	__device__ __host__
		float orbit_start() const { return this->orbitStart; };
	__device__ __host__
		int ns_() const { return this->ns; };
	__device__ __host__
		int nt_() const { return this->nt; };
	__device__ __host__
		float ws_() const { return this->ws; };
	__device__ __host__
		float wt_() const { return this->wt; };
	__device__ __host__
		int n_angle() const { return this->nAngle; };
	__device__ __host__
		int n_angle_add() const { return this->nAngleAdd; };
	__device__ __host__
		float ds_() const { return this->ds; };
	__device__ __host__
		float dt_() const { return this->dt; };
	__device__ __host__
		float pitch_() const { return this->pitch; };
	__device__ __host__
		float source_z0() const { return this->sourceZ0; };
	__device__ __host__
		float dso_() const { return this->dso; };
	__device__ __host__
		float dsd_() const { return this->dsd; };
	__device__ __host__
		int n_proj_turn() const { return this->nProjTurn; };
	__device__ __host__
		int z_dir() const { return this->zDir; };
	__device__ __host__
		int add_projections() const { return this->addProjections; };
};

class ImgGeom
{
private:
	int nx, ny, nz;	//�ؽ�ͼ�������������
	float wx, wy, wz;	//ԭ��λ��
	float dx, dy, dz;	//��������

public:
	ImgGeom(){};
	ImgGeom(int nx, int ny, int nz, float dx, float dy, float dz, float offX, float offY, float offZ)
	{
		this->nx = nx;
		this->ny = ny;
		this->nz = nz;
		this->dx = dx;
		this->dy = dy;
		this->dz = dz;
		this->wx = (nx - 1) / 2.0f + offX;
		this->wy = (ny - 1) / 2.0f + offY;
		this->wz = offZ;
	}
	__device__ __host__
		ImgGeom &operator=(const ImgGeom & ig)
	{
		this->nx = ig.nx;
		this->ny = ig.ny;
		this->nz = ig.nz;
		this->dx = ig.dx;
		this->dy = ig.dy;
		this->dz = ig.dz;
		this->wx = ig.wx;
		this->wy = ig.wy;
		this->wz = ig.wz;
		return *this;
	}
	__device__ __host__
		int nx_() const { return this->nx; };
	__device__ __host__
		int ny_() const { return this->ny; };
	__device__ __host__
		int nz_() const { return this->nz; };
	__device__ __host__
		float wx_() const { return this->wx; };
	__device__ __host__
		float wy_() const { return this->wy; };
	__device__ __host__
		float wz_() const { return this->wz; };
	__device__ __host__
		float dx_() const { return this->dx; };
	__device__ __host__
		float dy_() const { return this->dy; };
	__device__ __host__
		float dz_() const { return this->dz; };
};

class FileInfo
{
private:
	std::string inFileDir;	//�������·��
	std::string dataFileName;	//proj����ļ���
	std::string outputDir;	//����ļ�·��
	std::string outputName;	//����ļ���

public:
	FileInfo(const std::string inFileDir, const std::string dataFileName, const std::string outputDir, const std::string outputName)
	{
		this->inFileDir = inFileDir;
		this->dataFileName = dataFileName;
		this->outputDir = outputDir;
		this->outputName = outputName;
	}
	std::string in_file_dir() const { return this->inFileDir; };
	std::string data_file_name() const { return this->dataFileName; };
	std::string output_dir() const { return this->outputDir; };
	std::string output_name() const { return this->outputName; };
};

class CTDataCPU
{
private:
	float *proj;
	float *rebin;
	float *image;
	float *tubeAngles;
	float *tablePositions;

public:
	CTDataCPU()
	{
		proj = 0;
		rebin = 0;
		image = 0;
		tubeAngles = 0;
		tablePositions = 0;
	}
	CTDataCPU(const float *tubeAngle, const float *tablePosition, const int sizeProj, const int sizeRebin, const int sizeImage, const int nReadings);
	~CTDataCPU();
	float *proj_() const { return this->proj; };
	float *rebin_() const { return this->rebin; };
	float *image_() const { return this->image; };
	float *tube_angles() const { return this->tubeAngles; };
	float *table_positions() const { return this->tablePositions; };
};

class CTDataGPU
{
private:
	GPUMemoryCtrl<CTGeom> *ctgGpu;
	GPUMemoryCtrl<ImgGeom> *igGpu;
	GPUMemoryCtrl<float> *projGpu;
	GPUMemoryCtrl<float> *imgGpu;
	GPUMemoryCtrl<float> *betasGpu;
public:
	CTDataGPU();
	~CTDataGPU();
	GPUMemoryCtrl<CTGeom> *ctg_gpu() const { return this->ctgGpu; };
	GPUMemoryCtrl<ImgGeom> *ig_gpu() const { return this->igGpu; };
	GPUMemoryCtrl<float> *proj_gpu() const { return this->projGpu; };
	GPUMemoryCtrl<float> *img_gpu() const { return this->imgGpu; };
	GPUMemoryCtrl<float> *betas_gpu() const { return this->betasGpu; };
};

class ReconInfo
{
private:
	int nReadings;		//�ܹ���ȡ��view��
	float dataBeginPos;
	float dataEndPos;
	float allowBegin;
	float allowEnd;
	float startPos;
	float endPos;
	float reconStartPos;
	float reconEndPos;
	int nSlicesRecon;
	int nSlicesBlock;
	int nBlocks;
	int dataOffset;
	ReconInfo(){};

public:
	ReconInfo(const int nReadings, const float dataBeginPos, const float dataEndPos, const float allowBegin, const float allowEnd, const float startPos, const float endPos, const float reconStartPos, const float reconEndPos, const int nSlicesRecon, const int nSlicesBlock, const int nBlocks, const int dataOffset)
	{
		this->nReadings = nReadings;
		this->dataBeginPos = dataBeginPos;
		this->dataEndPos = dataEndPos;
		this->allowBegin = allowBegin;
		this->allowEnd = allowEnd;
		this->startPos = startPos;
		this->endPos = endPos;
		this->reconStartPos = reconStartPos;
		this->reconEndPos = reconEndPos;
		this->nSlicesRecon = nSlicesRecon;
		this->nSlicesBlock = nSlicesBlock;
		this->nBlocks = nBlocks;
		this->dataOffset = dataOffset;
	}
	int n_readings() const { return this->nReadings; };
	float data_begin_pos() const { return this->dataBeginPos; };
	float data_end_pos() const { return this->dataEndPos; };
	float allow_begin() const { return this->allowBegin; };
	float allow_end() const { return this->allowEnd; };
	float start_pos() const { return this->startPos; };
	float end_pos() const { return this->endPos; };
	float recon_start_pos() const { return this->reconStartPos; };
	float recon_end_pos() const { return this->reconEndPos; };
	int n_slices_recon() const { return this->nSlicesRecon; };
	int n_slices_block() const { return this->nSlicesBlock; };
	int n_blocks() const { return this->nBlocks; };
	int data_offset() const { return this->dataOffset; };
};

class BlockInfo
{
private:
	static int blockId;
	int blockProjStart;
	int blockProjEnd;
	float blockStart;
	float blockEnd;
	int blockSliceStart;
	int blockSliceEnd;

public:
	BlockInfo(const int blockProjStart, const int blockProjEnd, const float blockStart, const float blockEnd, const int blockSliceStart, const int blockSliceEnd)
	{
		blockId = 0;
		this->blockProjStart = blockProjStart;
		this->blockProjEnd = blockProjEnd;
		this->blockStart = blockStart;
		this->blockEnd = blockEnd;
		this->blockSliceStart = blockSliceStart;
		this->blockSliceEnd = blockSliceEnd;
	}
	int block_idx() const { return blockId; };
	int block_proj_start() const { return this->blockProjStart; };
	int block_proj_end() const { return this->blockProjEnd; };
	float block_start() const { return this->blockStart; };
	float block_end() const { return this->blockEnd; };
	int block_slice_start() const { return this->blockSliceStart; };
	int block_slice_end() const { return this->blockSliceEnd; };
	void update(const ReconInfo *ri, const CTGeom *ctg, const ImgGeom *ig, const CTDataCPU *dataCPU);
};

class ReconData
{
private:
	CTGeom *ctg;
	ImgGeom *ig;
	FileInfo *fi;
	CTDataCPU *dataCPU;
	CTDataGPU *dataGPU;
	ReconInfo *ri;
	BlockInfo *bi;
	void convert_from_file_params(const FileParams &fp);
	ReconData(){};
	ReconData(const ReconData &){};

public:
	ReconData(const FileParams &fp)
	{
		convert_from_file_params(fp);
	}
	~ReconData();
	CTGeom *ctg_() const { return this->ctg; };
	ImgGeom *ig_() const { return this->ig; };
	FileInfo *fi_() const { return this->fi; };
	CTDataCPU *data_cpu() const { return this->dataCPU; };
	CTDataGPU *data_gpu() const { return this->dataGPU; };
	ReconInfo *ri_() const { return this->ri; };
	BlockInfo *bi_() const { return this->bi; };
};
