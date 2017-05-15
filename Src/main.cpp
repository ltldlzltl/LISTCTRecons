#include "FileParams.h"
#include "DataStructs.h"
#include "ReconstructorAbstractFactory.h"
#include "Reconstructor.h"
#include "ImageWriter.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
	//log_file.open("recons.log", ios::app);
	/*cout << "Parameter file name:";
	string prmFile;
	cin >> prmFile;*/
    if(argc < 2)
    {
        cout<<"Usage LISTCTRecons <PRM filename>"<<endl;
        return 1;
    }
    string prmFile = argv[1];


	cout << "Reading prm file..." << endl;
	FileParams *fp = new FileParams(prmFile);

	cout << "Converting parameters..." << endl;
	ReconData *mr = new ReconData(*fp);

	string recon_type = "Helicalfdkramprebingpu";

	cout << "Creating reconstructor..." << endl;
	ReconstrctorAbstractFactory *rFac = new ReconstrctorAbstractFactory();
	Reconstructor *recon = rFac->createReconstructor(recon_type);

	recon->reconstruct(mr);

	ImageWriter *imWriter = new RawImageWriter();
	imWriter->writeImage(mr);

	delete fp;
	delete mr;
	delete rFac;
	delete recon;
	delete imWriter;
	//log_file.close();

	return 0;
}
