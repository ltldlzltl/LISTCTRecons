#include "Filter.h"
#include "DataStructs.h"
#include "Common.h"
#include <cmath>
#include <omp.h>
using namespace std;

void RampFilterGPU::calculate_filter(float *h, const CTGeom *ctg)
{
#pragma omp parallel for
	for (int i = 0; i < ctg->ns_() * 2; i++)
	{
		int n = i - ctg->ns_();
		if (n == 0)
			h[i] = 0.25 / ctg->ds_();
		else
		{
			h[i] = (sin(n*PI) / (n*PI) + (cos(n*PI) - 1) / (n*n*PI*PI)) / (2 * ctg->ds_());
		}
	}
// 	for (int i = 0; i < ctg->ns_() * 2; i++)
// 		cout << "i=" << i << " " << "h[i]=" << h[i] << endl;
}
