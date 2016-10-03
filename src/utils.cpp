#include <cmath>
#include <iostream>

#include "utils.hpp"


void matrix_generator(double *X, size_t dimensions[3])
{
	// retrieve dimensions
	int N0 = dimensions[0];
	int N1 = dimensions[1];
	int N2 = dimensions[2];

	double tti, ttj, ttk, dti, dtj, dtk, A0i, A0j, A0k, A1i, A1j, A1k;
	dti = 0.1;  
	dtj = 2.0*dti;  
	dtk = dtj;
	
	A0i = exp(-N0*dti);
	A0j = exp(-N1*dtj);
	A0k = exp(-N2*dtk);

	//printf("Source\n");
	for (int i = 0; i<N0; i++)
	{
		tti = (double)i*dti;
		A1i = exp(-tti);
		for (int j = 0; j<N1; j++)
		{
			ttj = (double)j*dtj;
			A1j = exp(-ttj);
			for (int k = 0; k<N2; k++)
			{
				ttk = (double)k*dtk;
				A1k = exp(-ttk);

				size_t idx = 2 * (k + j*N2 + i*N1*N2);
				X[idx] = (A1i + A0i / A1i)*dti*(A1j + A0j / A1j)*dtj*(A1k + A0k / A1k)*dtk;
				X[idx + 1] = 0.0;

				//printf("(%f, %f) ", X[idx], X[idx + 1]);
			}
			//printf("\n");
		}
		//printf("\n");
	}
}

void matrix_generator(double *X, int dimensions[3])
{
	// retrieve dimensions
	int N0 = dimensions[0];
	int N1 = dimensions[1];
	int N2 = dimensions[2];

	double tti, ttj, ttk, dti, dtj, dtk, A0i, A0j, A0k, A1i, A1j, A1k;
	dti = 0.1;
	dtj = 2.0*dti;
	dtk = dtj;

	A0i = exp(-N0*dti);
	A0j = exp(-N1*dtj);
	A0k = exp(-N2*dtk);

	//printf("Source\n");
	for (int i = 0; i<N0; i++)
	{
		tti = (double)i*dti;
		A1i = exp(-tti);
		for (int j = 0; j<N1; j++)
		{
			ttj = (double)j*dtj;
			A1j = exp(-ttj);
			for (int k = 0; k<N2; k++)
			{
				ttk = (double)k*dtk;
				A1k = exp(-ttk);

				size_t idx = 2 * (k + j*N2 + i*N1*N2);
				X[idx] = (A1i + A0i / A1i)*dti*(A1j + A0j / A1j)*dtj*(A1k + A0k / A1k)*dtk;
				X[idx + 1] = 0.0;

				//printf("(%f, %f) ", X[idx], X[idx + 1]);
			}
			//printf("\n");
		}
		//printf("\n");
	}
}