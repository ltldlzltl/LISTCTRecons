#include "Reconstructor.h"
#include "Common.h"
#include "DataStructs.h"
#include "Rebiner.h"
#include "Filter.h"
#include "Backprojector.h"
#include "DataReader.h"
#include <iostream>
using namespace std;

FDKReconstructor::FDKReconstructor(Rebiner *rbn, Filter *flt, Backprojector *bp)
{
	this->rbn = rbn;
	this->flt = flt;
	this->bp = bp;
	this->dr = new RawDataReader();
}

FDKReconstructor::~FDKReconstructor()
{
	delete this->rbn;
	delete this->flt;
	delete this->bp;
	delete this->dr;
}

void FDKReconstructor::reconstruct(const ReconData *mr)
{
	cout << "Start reconstruction..." << endl;
	for (int i = 0; i < mr->ri_()->n_blocks(); i++)
	{
		cout << endl << "Start processing block " << i << "..." << endl;
		cout << "Reading data..." << endl;
		dr->readData(mr);
		cout << "Rebining..." << endl;
		rbn->rebin(mr);
		cout << "Filtering..." << endl;
		flt->filter(mr);
		cout << "Backprojecting..." << endl;
		bp->backproject(mr);
		mr->bi_()->update(mr->ri_(), mr->ctg_(), mr->ig_(), mr->data_cpu());
		cout << "Block " << i << " finished." << endl;
	}
}
