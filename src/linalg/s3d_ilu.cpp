/** \file
 * \brief ILU implementations
 */

#include "s3d_ilu.hpp"

StrILU_preconditioner::StrILU_preconditioner(const SMat& lhs, const PreconParams parms)
	: SGS_like_preconditioner(lhs, parms)
{
}

void StrILU_preconditioner::updateOperator()
{
	updateOperatorWithBranchingInLoop();
	//updateOperatorWithSeparateLoops();
}

void StrILU_preconditioner::updateOperatorWithBranchingInLoop()
{
	sreal starttime = MPI_Wtime();

	/* NOTE: We need to use a signed index type for LLVM's vectorizer to vectorize loops
	 */
	// initialize
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = A.vals[STENCIL_DIAG][i];

#pragma omp parallel default(shared) if(params.threadedbuild)
	{
		for(int iswp = 0; iswp < params.nbuildsweeps; iswp++)
		{
#pragma omp for collapse(2) nowait
			for(sint k = A.start; k < A.start + A.sz[2]; k++)
				for(sint j = A.start; j < A.start + A.sz[1]; j++)
#pragma omp simd
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
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = 1.0/diaginv[i];

	sreal etime = MPI_Wtime() - starttime;
	printf(" >> Time taken to compute ILU = %f.\n", etime);
}

void StrILU_preconditioner::updateOperatorWithSeparateLoops()
{
	sreal starttime = MPI_Wtime();

	// initialize
	/* NOTE: We need to use a signed index type for LLVM's vectorizer to vectorize loops
	 */
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = A.vals[STENCIL_DIAG][i];

#pragma omp parallel default(shared) if(params.threadedbuild)
	{
		for(int iswp = 0; iswp < params.nbuildsweeps; iswp++)
		{
			// corners

#pragma omp for simd nowait
			for(sint i = A.start+1; i < A.start + A.sz[0]; i++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(0,0,i);
				const sint jdx =  A.m->localFlattenedIndexReal(0,0,i-1);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[0][idxr]*A.vals[4][jdx]/diaginv[jdx];
			}
#pragma omp for nowait
			for(sint j = A.start+1; j < A.start + A.sz[1]; j++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(0,j,0);
				const sint jdx =  A.m->localFlattenedIndexReal(0,j-1,0);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[1][idxr]*A.vals[5][jdx]/diaginv[jdx];
			}
#pragma omp for nowait
			for(sint k = A.start+1; k < A.start + A.sz[2]; k++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(k,0,0);
				const sint jdx =  A.m->localFlattenedIndexReal(k-1,0,0);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[2][idxr]*A.vals[6][jdx]/diaginv[jdx];
			}

			// faces

#pragma omp for nowait
			for(sint j = A.start+1; j < A.start + A.sz[1]; j++)
#pragma omp simd
				for(sint i = A.start+1; i < A.start + A.sz[0]; i++)
				{
					const sint idxr = A.m->localFlattenedIndexReal(0,j,i);
					const sint jdx[] = { A.m->localFlattenedIndexReal(0,j,i-1),
					                     A.m->localFlattenedIndexReal(0,j-1,i) };
					// diag
					diaginv[idxr] = A.vals[STENCIL_DIAG][idxr];
					// i-dir
					diaginv[idxr] -= A.vals[0][idxr] * A.vals[4][jdx[0]]
						/ diaginv[jdx[0]];
					// j-dir
					diaginv[idxr] -= A.vals[1][idxr] * A.vals[5][jdx[1]]
						/ diaginv[jdx[1]];
				}

#pragma omp for nowait
			for(sint k = A.start+1; k < A.start + A.sz[2]; k++)
#pragma omp simd
				for(sint i = A.start+1; i < A.start + A.sz[0]; i++)
				{
					const sint idxr = A.m->localFlattenedIndexReal(k,0,i);
					const sint jdx[] = { A.m->localFlattenedIndexReal(k,0,i-1),
					                     -1,                                    //< dummy
					                     A.m->localFlattenedIndexReal(k-1,0,i)};
					// diag
					diaginv[idxr] = A.vals[STENCIL_DIAG][idxr];
					// i-dir
					diaginv[idxr] -= A.vals[0][idxr] * A.vals[4][jdx[0]]
						/ diaginv[jdx[0]];
					// k-dir
					diaginv[idxr] -= A.vals[2][idxr] * A.vals[6][jdx[2]]
						/ diaginv[jdx[2]];
				}

#pragma omp for nowait
			for(sint k = A.start+1; k < A.start + A.sz[2]; k++)
#pragma omp simd
				for(sint j = A.start+1; j < A.start + A.sz[1]; j++)
				{
					const sint idxr = A.m->localFlattenedIndexReal(k,j,0);
					const sint jdx[] = { -1,                                     //< dummy
					                     A.m->localFlattenedIndexReal(k,j-1,0),
					                     A.m->localFlattenedIndexReal(k-1,j,0)};
					// diag
					diaginv[idxr] = A.vals[STENCIL_DIAG][idxr];
					// j-dir
					diaginv[idxr] -= A.vals[1][idxr] * A.vals[5][jdx[1]]
						/ diaginv[jdx[1]];
					// k-dir
					diaginv[idxr] -= A.vals[2][idxr] * A.vals[6][jdx[2]]
						/ diaginv[jdx[2]];
				}

			// interior 
#pragma omp for collapse(2) nowait
			for(sint k = A.start+1; k < A.start + A.sz[2]; k++)
				for(sint j = A.start+1; j < A.start + A.sz[1]; j++)
#pragma omp simd
					for(sint i = A.start+1; i < A.start + A.sz[0]; i++)
					{
						const sint idxr = A.m->localFlattenedIndexReal(k,j,i);
						const sint jdx[] = { A.m->localFlattenedIndexReal(k,j,i-1),
						                     A.m->localFlattenedIndexReal(k,j-1,i),
						                     A.m->localFlattenedIndexReal(k-1,j,i)};
						// diag
						diaginv[idxr] = A.vals[STENCIL_DIAG][idxr];
						// i-dir
						diaginv[idxr] -= A.vals[0][idxr] * A.vals[4][jdx[0]]
							/ diaginv[jdx[0]];
						// j-dir
						diaginv[idxr] -= A.vals[1][idxr] * A.vals[5][jdx[1]]
							/ diaginv[jdx[1]];
						// k-dir
						diaginv[idxr] -= A.vals[2][idxr] * A.vals[6][jdx[2]]
							/ diaginv[jdx[2]];
					}
		}
	}

	// invert
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = 1.0/diaginv[i];

	sreal etime = MPI_Wtime() - starttime;
	printf(" >> Time taken to compute ILU = %f.\n", etime);
}
