#include "Reconstructor.h"
#include "ReconstructorAbstractFactory.h"
#include "Rebiner.h"
#include "Backprojector.h"
#include "Filter.h"
#include <iostream>
#include <cstdlib>
using namespace std;

Reconstructor *ReconstrctorAbstractFactory::createReconstructor(const std::string &str)
{
	if (str.find("fdk"))
	{
		Rebiner *rbn = 0;
		Filter *flt = 0;
		Backprojector *bp = 0;
		if (str.find("GPU") || str.find("gpu"))
		{
			if (str.find("rebin"))
			{
				rbn = new HelicalConeBeamFan2ParRebinerGPU();
				bp = new HelicalConeBeamFanRebinBackprojectorGPU();
			}
			if (str.find("ramp"))
				flt = new RampFilterGPU();
		}
		if (rbn == 0 || flt == 0 || bp == 0)
		{
			cout << "Error happens when building reconstructor." << endl;
			exit(1);
		}
		return (new FDKReconstructor(rbn, flt, bp));
	}

	cout << "Error happens when building reconstructor." << endl;
	exit(1);
	return NULL;
}
