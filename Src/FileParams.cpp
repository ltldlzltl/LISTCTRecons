#include "FileParams.h"
#include "Common.h"
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

string TokenNames[24] = { "RawDataDir", "DataInterpFile", "ViewAngleFile", "CouchPosFile", "OutputDir", "OutputName", "Ns", "Nt", "Ds", "Dt", "PitchValue", "Nx", "Ny", "Dx", "Dz", "OffsetX", "OffsetY", "Readings", "ViewOffset", "DataOffset", "RSrcToIso", "RSrcToDet", "CentralChannel", "NProjTurn" };

bool FileParams::enough_data(const bool haveData[24])
{
	for (int i = 0; i < 24; i++)
	{
		if (!haveData[i])
		{
			cout << "Parameter not found in prm file:" << TokenNames[i] << endl;
			exit(1);
			return false;
		}
	}
	return true;
}

void FileParams::readPrmFile(const string prmFileName)
{
	bool haveData[24] = { false };

	fstream file(prmFileName.c_str(), ios::in);
	if (!file.is_open())
	{
		cout << "Parameter file not found. " << endl;
		exit(1);
	}
	string token;
	while (file >> token)
	{
		if (token == "RawDataDir:"){
			file >> RawFileDir;
			haveData[0] = true;
		}
		else if (token == "DataInterpFile:"){
			file >> DataInterpFile;
			haveData[1] = true;
		}
		else if (token == "ViewAngleFile:" ){
			file >> ViewAngleFile;
			haveData[2] = true;
		}
		else if (token == "CouchPosFile:" ){
			file >> CouchPosFile;
			haveData[3] = true;
		}
		else if (token == "OutputDir:" ){
			file >> OutputDir;
			haveData[4] = true;
		}
		else if (token == "OutputName:" ){
			file >> OutputName;
			haveData[5] = true;
		}
		else if (token == "Ns:" ){
			file >> Ns;
			haveData[6] = true;
		}
		else if (token == "Nt:" ){
			file >> Nt;
			haveData[7] = true;
		}
		else if (token == "Ds:" ){
			file >> Ds;
			haveData[8] = true;
		}
		else if (token == "Dt:" ){
			file >> Dt;
			haveData[9] = true;
		}
		else if ((token == "PitchValue:" ) || (token == "TableFeed:" )){
			file >> PitchValue;
			haveData[10] = true;
		}
		else if (token == "Nx:" ){
			file >> Nx;
			haveData[11] = true;
		}
		else if (token == "Ny:" ){
			file >> Ny;
			haveData[12] = true;
		}
		else if (token == "Dx:" ){
			file >> Dx;
			haveData[13] = true;
		}
		else if (token == "Dz:" ){
			file >> Dz;
			haveData[14] = true;
		}
		else if (token == "OffsetX:" ){
			file >> OffsetX;
			haveData[15] = true;
		}
		else if (token == "OffsetY:" ){
			file >> OffsetY;
			haveData[16] = true;
		}
		else if (token == "Readings:" ){
			file >> Readings;
			haveData[17] = true;
		}
		else if (token == "ViewOffset:" ){
			file >> ViewOffset;
			haveData[18] = true;
		}
		else if (token == "DataOffset:" ){
			file >> DataOffset;
			haveData[19] = true;
		}
		if (token == "RSrcToIso:" ){
			file >> RSrcToIso;
			haveData[20] = true;
		}
		else if (token == "RSrcToDet:" ){
			file >> RSrcToDet;
			haveData[21] = true;
		}
		else if (token == "CentralChannel:" ){
			file >> CentralChannel;
			haveData[22] = true;
		}
		else if (token == "NProjTurn:" ){
			file >> NProjTurn;
			haveData[23] = true;
		}
	}
	file.close();

	if(!enough_data(haveData));

	return;
}

FileParams::FileParams(const string prmFileName)
{
	this->readPrmFile(prmFileName);
}
