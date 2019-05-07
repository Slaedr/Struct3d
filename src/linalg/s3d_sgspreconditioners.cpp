/** \file
 * \brief Implementation of SGS-like application and SGS preconditioner
 */

#include <cassert>
#include "s3d_sgspreconditioners.hpp"

SGS_like_preconditioner::SGS_like_preconditioner(const SMat& lhs, const PreconParams parms)
	: SolverBase(lhs, nullptr), params(parms)
{
	assert(A.m->gnPoinTotal() == A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2));

	diaginv.resize(A.m->gnPoinTotal());
}

SolveInfo SGS_like_preconditioner::apply(const SVec& r, SVec& z) const
{
	if(r.m != z.m || r.m != A.m)
		throw std::runtime_error("apply: Vectors and matrix must be defined over the same mesh!");

	/* Temporary vector for application. Same size as arguments to apply().
	 * It's best for this vector to be initialized to zero before every application.
	 */
	s3d::vector<sreal> y(A.m->gnPoinTotal());
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < A.m->gnPoinTotal(); i++)
		y[i] = 0;
	
	const sint idxmax[3] = {r.start + r.sz[0], r.start + r.sz[1], r.start + r.sz[2]};

#pragma omp parallel default(shared) if(params.threadedapply)
	{
		for(int iswp = 0; iswp < params.napplysweeps; iswp++)
		{
			/* The loop below speeds up a little from auto-vectorization by LLVM 6.0 and still gives
			 * the (exact) same answer as un-vectorized code (using a loop over the stencil for y;
			 * LLVM says it's not been vectorized in that case).
			 * However, adding an omp simd pragma over the i-loop leads to the solver taking more
			 * iterations (again using LLVM/Clang 6.0).
			 *
			 * In case of GCC 8.2, the situation is similar. Auto-vectorization leads to no change in
			 * number of iterations while omp simd is faster per iteration but takes more iterations.
			 * 
			 * My guess is that any vectorization in GCC produces two versions of the vectorized loop,
			 * and it always ends up using the slower version with Not Much Vectorization (TM).
			 *   In case of Clang, while it never mentions any versioning, I suspect its LLVM vectorizer
			 * still generates versions and uses the slower version because it has detected dependence or
			 * something. However, the LLVM OMP simd seems to use the proper vectorized version, because
			 * now the solver takes more iterations.
			 */

#pragma omp for collapse(2) nowait schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i-1),
							r.m->localFlattenedIndexAll(k,j-1,i),
							r.m->localFlattenedIndexAll(k-1,j,i),
							r.m->localFlattenedIndexAll(k,j,i),
						};

						// y[jdx[3]] = r.vals[jdx[3]];
						// for(int is = 0; is < STENCIL_DIAG; is++)
						// 	y[jdx[3]] -= A.vals[is][idxr] * y[jdx[is]];

						y[idx] = r.vals[idx] - A.vals[0][idx]*y[jdx[0]]
							- A.vals[1][idx]*y[jdx[1]] - A.vals[2][idx]*y[jdx[2]];

						y[idx] *= diaginv[idx];
					}
		}

// #pragma omp parallel for simd default(shared)
// 		for(sint i = 0; i < A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2); i++)
// 			z.vals[i] = y[i];

		for(int iswp = 0; iswp < params.napplysweeps; iswp++)
		{
#pragma omp for collapse(2) nowait schedule(static, params.thread_chunk_size)
			for(sint k = idxmax[2]-1; k >= r.start; k--)
				for(sint j = idxmax[1]-1; j >= r.start; j--)
#pragma omp simd
					for(sint i = idxmax[0]-1; i >= r.start; i--)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							-1, -1, -1,
							r.m->localFlattenedIndexAll(k,j,i),
							r.m->localFlattenedIndexAll(k,j,i+1),
							r.m->localFlattenedIndexAll(k,j+1,i),
							r.m->localFlattenedIndexAll(k+1,j,i)
						};

						// z.vals[jdx[3]] = y[jdx[3]];
						// for(int is = STENCIL_DIAG + 1; is < NSTENCIL; is++)
						// 	z.vals[jdx[3]] -= diaginv[idxr]*A.vals[is][idxr] * z.vals[jdx[is]];

						z.vals[idx] = y[idx] - diaginv[idx] * ( A.vals[4][idx]*z.vals[jdx[4]]
						                                        + A.vals[5][idx]*z.vals[jdx[5]]
						                                        + A.vals[6][idx]*z.vals[jdx[6]]);
					}
		}
	}

	return {false, 1, -1.0};
}

SGS_preconditioner::SGS_preconditioner(const SMat& lhs, const PreconParams parms)
	: SGS_like_preconditioner(lhs, parms)
{
}

void SGS_preconditioner::updateOperator()
{
	const sint idxmax[3] = {A.start + A.sz[0], A.start + A.sz[1], A.start + A.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = A.start; k < idxmax[2]; k++)
		for(sint j = A.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = A.start; i < idxmax[0]; i++)
			{
				const sint idx = A.m->localFlattenedIndexAll(k,j,i);
				diaginv[idx] = 1.0/A.vals[STENCIL_DIAG][idx];
			}
}

