#pragma once
#include <string>

class FileParams
{
private:
	std::string RawFileDir;
	std::string DataInterpFile;
	std::string ViewAngleFile;
	std::string CouchPosFile;
	std::string OutputDir;
	std::string OutputName;
	int Ns, Nt;
	float Ds, Dt;
	float PitchValue;
	int Nx, Ny;
	float Dx, Dz;
	float OffsetX, OffsetY;
	int Readings;
	int ViewOffset, DataOffset;
	float RSrcToIso, RSrcToDet;
	float CentralChannel;
	int NProjTurn;
	FileParams(){};
	bool enough_data(const bool haveData[]);
	void readPrmFile(const std::string prmFileName);

public:
	FileParams(const std::string prmFileName);
	std::string raw_file_dir() const { return this->RawFileDir; };
	std::string data_interp_file() const { return this->DataInterpFile; };
	std::string view_angle_file() const { return this->ViewAngleFile; };
	std::string couch_pos_file() const { return this->CouchPosFile; };
	std::string output_dir() const { return this->OutputDir; };
	std::string output_name() const { return this->OutputName; };
	int ns() const { return this->Ns; };
	int nt() const { return this->Nt; };
	float ds() const { return this->Ds; };
	float dt() const { return this->Dt; };
	float pitch_value() const { return this->PitchValue; };
	int nx() const { return this->Nx; };
	int ny() const { return this->Ny; };
	float dx() const { return this->Dx; };
	float dz() const { return this->Dz; };
	float offset_x() const { return this->OffsetX; };
	float offset_y() const { return this->OffsetY; };
	int readings() const { return this->Readings; };
	int view_offset() const { return this->ViewOffset; };
	int data_offset() const { return this->DataOffset; };
	float r_src_to_iso() const { return this->RSrcToIso; };
	float r_src_to_det() const { return this->RSrcToDet; };
	float central_channel() const { return this->CentralChannel; };
	int n_proj_turn() const { return this->NProjTurn; };
};
