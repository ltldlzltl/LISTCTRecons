#include "Reconstructor.h"
#include "ReconstructorAbstractFactory.h"
#include "Rebiner.h"
#include "Backprojector.h"
#include "Filter.h"
#include "PreWeighting.h"
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
		PreWeighting *pw = 0;
		/*rbn = new PiOriRebiner();
		pw = new PiOriPreWeighting();
		flt = new RampFilterGPU();
		bp = new PiOriBackprojectorGPU();*/
		rbn = new HelicalConeBeamFan2ParRebinerGPU();
		pw = new NULLPreWeighting();
		flt = new RampFilterGPU();
		bp = new HelicalConeBeamFanRebinBackprojectorGPU();
		return (new FDKReconstructor(rbn, flt, bp, pw));
	}

	cout << "Error happens when building reconstructor." << endl;
	exit(1);
	return NULL;
}
