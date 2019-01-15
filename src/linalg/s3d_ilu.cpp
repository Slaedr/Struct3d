/** \file
 * \brief ILU implementations
 */

#include "s3d_ilu.hpp"

StrILU_preconditioner::StrILU_preconditioner(const SMat& lhs, const PreconParams parms)
	: SGS_like_preconditioner(lhs, parms)
{
	updateOperator();
}

void StrILU_preconditioner::updateOperator()
{
	sreal starttime = MPI_Wtime();

	// initialize
#pragma omp parallel for simd default(shared)
	for(size_t i = 0; i < diaginv.size(); i++)
		diaginv[i] = A.vals[STENCIL_DIAG][i];

#pragma omp parallel default(shared) if(params.threadedbuild)
	{
		for(int iswp = 0; iswp < params.nbuildsweeps; iswp++)
		{
#pragma omp for collapse(3) nowait
			for(sint k = A.start; k < A.start + A.sz[2]; k++)
				for(sint j = A.start; j < A.start + A.sz[1]; j++)
					for(sint i = A.start; i < A.start + A.sz[0]; i++)
					{
						const sint idxr = A.m->localFlattenedIndexReal(k,j,i);
						const sint jdx[] = { A.m->localFlattenedIndexReal(k,j,i-1),
						                     A.m->localFlattenedIndexReal(k,j-1,i),
						                     A.m->localFlattenedIndexReal(k-1,j,i)};
						// diag
						diaginv[idxr] = A.vals[STENCIL_DIAG][idxr];
						// i-dir
						if(i > 0)
							diaginv[idxr] -= A.vals[0][idxr] * A.vals[4][jdx[0]]
								/ diaginv[jdx[0]];
						// j-dir
						if(j > 0)
							diaginv[idxr] -= A.vals[1][idxr] * A.vals[5][jdx[1]]
								/ diaginv[jdx[1]];
						// k-dir
						if(k > 0)
							diaginv[idxr] -= A.vals[2][idxr] * A.vals[6][jdx[2]]
								/ diaginv[jdx[2]];
					}
		}
	}

	// invert
#pragma omp parallel for simd default(shared)
	for(size_t i = 0; i < diaginv.size(); i++)
		diaginv[i] = 1.0/diaginv[i];

	sreal etime = MPI_Wtime() - starttime;
	printf(" >> Time taken to compute ILU = %f.\n", etime);
}
