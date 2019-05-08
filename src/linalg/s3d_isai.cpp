
#include "s3d_isai.hpp"

ISAI_preconditioner::ISAI_preconditioner(const SMat& lhs, const PreconParams parms)
	: SolverBase(lhs, nullptr), params(parms)
{
	diaginv.resize(A.m->gnPoinTotal());

	tres.resize(A.m->gnPoinTotal());
	y.resize(A.m->gnPoinTotal());
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < A.m->gnPoinTotal(); i++) {
		tres[i] = 0;
		y[i] = 0;
	}
}

SolveInfo ISAI_preconditioner::apply(const SVec& r, SVec& z) const
{
	if(r.m != z.m || r.m != A.m)
		throw std::runtime_error("apply: Vectors and matrix must be defined over the same mesh!");

	/** It is somewhat more reliable to initialize y to zero. It may sometimes work better
	 * without that, though.
	 * The z vector needs to be initialized to zero.
	 */
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < A.m->gnPoinTotal(); i++) {
		y[i] = 0;
		z.vals[i] = 0;
	}
	
	const sint idxmax[3] = {r.start + r.sz[0], r.start + r.sz[1], r.start + r.sz[2]};

	// Lower triangular solve
	for(int iswp = 0; iswp < params.napplysweeps; iswp++)
	{
#pragma omp parallel default(shared) if(params.threadedapply)
		{
			// Compute the triangular solve residual
			//  tres = r - y - L D^{-1} y
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
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
						};

						tres[idx] = r.vals[idx] - y[idx]
							- A.vals[0][idx]*y[jdx[0]] * diaginv[jdx[0]]
							- A.vals[1][idx]*y[jdx[1]] * diaginv[jdx[1]]
							- A.vals[2][idx]*y[jdx[2]] * diaginv[jdx[2]];
					}

			// multiply the residual temp with the approximate inverse and update, as
			//  y = y + M tres
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
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
						};

						y[idx] += tres[idx]
							- A.vals[0][idx]*tres[jdx[0]]*diaginv[jdx[0]]
							- A.vals[1][idx]*tres[jdx[1]]*diaginv[jdx[1]]
							- A.vals[2][idx]*tres[jdx[2]]*diaginv[jdx[2]];
					}
		}

// 		sreal tresnorm = 0;
// #pragma omp parallel for simd reduction(+:tresnorm)
// 		for(sint i = 0; i < A.m->gnPoinTotal(); i++)
// 			tresnorm += tres[i]*tres[i];
// 		tresnorm = std::sqrt(tresnorm);
// 		printf("   ISAI lower: Iter %d, tresnorm = %6g\n", iswp, tresnorm);
	}

// #pragma omp parallel for simd default(shared)
// 	for(sint i = 0; i < A.m->gnPoinTotal(); i++) {
// 		z.vals[i] = 0;
// 	}

		// Upper triangular solve

	for(int iswp = 0; iswp < params.napplysweeps; iswp++)
	{
#pragma omp parallel default(shared) if(params.threadedapply)
		{
			// tres = y - D z - U z
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i+1),
							r.m->localFlattenedIndexAll(k,j+1,i),
							r.m->localFlattenedIndexAll(k+1,j,i)
						};

						tres[idx] = y[idx] - z.vals[idx]/diaginv[idx]
							- A.vals[4][idx]*z.vals[jdx[0]]
							- A.vals[5][idx]*z.vals[jdx[1]]
							- A.vals[6][idx]*z.vals[jdx[2]];
					}

			// z = z + M tres
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i+1),
							r.m->localFlattenedIndexAll(k,j+1,i),
							r.m->localFlattenedIndexAll(k+1,j,i)
						};

						z.vals[idx] += diaginv[idx]
							* ( tres[idx] - (A.vals[4][idx]   *tres[jdx[0]]*diaginv[jdx[0]]
							                 + A.vals[5][idx] *tres[jdx[1]]*diaginv[jdx[1]]
							                 + A.vals[6][idx] *tres[jdx[2]]*diaginv[jdx[2]] ));
					}
		}

// 		sreal tresnorm = 0;
// #pragma omp parallel for simd reduction(+:tresnorm)
// 		for(sint i = 0; i < A.m->gnPoinTotal(); i++)
// 			tresnorm += tres[i]*tres[i];
// 		tresnorm = std::sqrt(tresnorm);
// 		printf("   ISAI upper: Iter %d, tresnorm = %6g\n", iswp, tresnorm);
	}

	return {false, 1, -1.0};
}

SGS_ISAI_preconditioner::SGS_ISAI_preconditioner(const SMat& lhs, const PreconParams parms)
	: ISAI_preconditioner(lhs, parms)
{
}

void SGS_ISAI_preconditioner::updateOperator()
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
