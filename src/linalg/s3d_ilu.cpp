/** \file
 * \brief ILU implementations
 */

#include "s3d_ilu.hpp"

void async_ilu0(const SMat& A, const PreconParams params, s3d::vector<sreal>& diaginv)
{
	// initialize
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = A.vals[STENCIL_DIAG][i];
	
	const sint idxmax[3] = {A.start + A.sz[0], A.start + A.sz[1], A.start + A.sz[2]};

#pragma omp parallel default(shared) if(params.threadedbuild)
	{
		for(int iswp = 0; iswp < params.nbuildsweeps; iswp++)
		{
#pragma omp for collapse(2) nowait schedule(dynamic, params.thread_chunk_size)
			for(sint k = A.start; k < idxmax[2]; k++)
				for(sint j = A.start; j < idxmax[1]; j++)
#pragma omp simd
					for(sint i = A.start; i < idxmax[0]; i++)
					{
						const sint idx = A.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = { A.m->localFlattenedIndexAll(k,j,i-1),
						                     A.m->localFlattenedIndexAll(k,j-1,i),
						                     A.m->localFlattenedIndexAll(k-1,j,i)};
						// diag
						diaginv[idx] = A.vals[STENCIL_DIAG][idx];
						// i-dir
						if(i > A.start)
							diaginv[idx] -= A.vals[0][idx] * A.vals[4][jdx[0]]
								/ diaginv[jdx[0]];
						// j-dir
						if(j > A.start)
							diaginv[idx] -= A.vals[1][idx] * A.vals[5][jdx[1]]
								/ diaginv[jdx[1]];
						// k-dir
						if(k > A.start)
							diaginv[idx] -= A.vals[2][idx] * A.vals[6][jdx[2]]
								/ diaginv[jdx[2]];
					}
		}
	}

	// invert
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = 1.0/diaginv[i];
}

StrILU_preconditioner::StrILU_preconditioner(const SMat& lhs, const PreconParams parms)
	: SGS_like_preconditioner(lhs, parms)
{
}

void StrILU_preconditioner::updateOperator()
{
	async_ilu0(A, params, diaginv);
}

AILU_ISAI_preconditioner::AILU_ISAI_preconditioner(const SMat& lhs, const PreconParams parms)
	: ISAI_preconditioner(lhs, parms)
{
}

void AILU_ISAI_preconditioner::updateOperator()
{
	async_ilu0(A, params, diaginv);
}

#if 0
void StrILU_preconditioner::updateOperatorWithSeparateLoops()
{
	// initialize
	/* NOTE: We need to use a signed index type for LLVM's vectorizer to vectorize loops
	 */
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < static_cast<sint>(diaginv.size()); i++)
		diaginv[i] = A.vals[STENCIL_DIAG][i];
	
	const sint idxmax[3] = {A.start + A.sz[0], A.start + A.sz[1], A.start + A.sz[2]};

#pragma omp parallel default(shared) if(params.threadedbuild)
	{
		for(int iswp = 0; iswp < params.nbuildsweeps; iswp++)
		{
			// corners

#pragma omp for simd nowait
			for(sint i = A.start+1; i < idxmax[0]; i++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(0,0,i);
				const sint jdx =  A.m->localFlattenedIndexReal(0,0,i-1);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[0][idxr]*A.vals[4][jdx]/diaginv[jdx];
			}
#pragma omp for nowait
			for(sint j = A.start+1; j < idxmax[1]; j++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(0,j,0);
				const sint jdx =  A.m->localFlattenedIndexReal(0,j-1,0);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[1][idxr]*A.vals[5][jdx]/diaginv[jdx];
			}
#pragma omp for nowait
			for(sint k = A.start+1; k < idxmax[2]; k++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(k,0,0);
				const sint jdx =  A.m->localFlattenedIndexReal(k-1,0,0);
				diaginv[idxr] = A.vals[STENCIL_DIAG][idxr]
					- A.vals[2][idxr]*A.vals[6][jdx]/diaginv[jdx];
			}

			// faces

#pragma omp for nowait
			for(sint j = A.start+1; j < idxmax[1]; j++)
#pragma omp simd
				for(sint i = A.start+1; i < idxmax[0]; i++)
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
			for(sint k = A.start+1; k < idxmax[2]; k++)
#pragma omp simd
				for(sint i = A.start+1; i < idxmax[0]; i++)
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
			for(sint k = A.start+1; k < idxmax[2]; k++)
#pragma omp simd
				for(sint j = A.start+1; j < idxmax[1]; j++)
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
			for(sint k = A.start+1; k < idxmax[2]; k++)
				for(sint j = A.start+1; j < idxmax[1]; j++)
#pragma omp simd
					for(sint i = A.start+1; i < idxmax[0]; i++)
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
}
#endif
