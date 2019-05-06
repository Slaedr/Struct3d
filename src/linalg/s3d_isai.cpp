
#include "s3d_isai.hpp"

ISAI_preconditioner::ISAI_preconditioner(const SMat& lhs, const PreconParams parms)
	: SolverBase(lhs, nullptr), params(parms)
{
	// FIXME: This size of diaginv will not do! Need padding for ghost points.
	diaginv.reize((A.m->gnpoind(0)-2)*(A.m->gnpoind(1)-2)*(A.m->gnpoind(2)-2));
	temp.init(A.m);
}

sreal ISAI_preconditioner::multiply_isai_lower(const sint idx, const sint idxr, const sint jdx[3],
                                               const SVec& x) const
{
	return x[idx] - A.vals[0][idxr]*x[jdx[0]] - A.vals[1][idxr]*x[jdx[1]]
		- A.vals[2][idxr]*x[jdx[2]];
}

sreal ISAI_preconditioner::multiply_isai_upper(const sint idx,const sint idxr,
                                               const sint jdx[3], const sint jdxr[3],
                                               const SVec& x) const
{
	return diaginv[idxr] * ( x[idx]
	                         - A.vals[4][idxr]*x[jdx[0]] * diaginv[jdxr[0]]
	                         - A.vals[5][idxr]*x[jdx[1]] * diaginv[jdxr[1]]
	                         - A.vals[6][idxr]*x[jdx[2]] * diaginv[jdxr[2]] );
}

SolveInfo ISAI_preconditioner::apply(const SVec& r, SVec& z) const
{
	if(r.m != z.m || r.m != A.m)
		throw std::runtime_error("apply: Vectors and matrix must be defined over the same mesh!");

	const int ng = r.nghost;
	assert(ng == 1);

	/* Temporary vector for application. Same size as arguments to apply().
	 * It's best for this vector to be initialized to zero before every application.
	 */
	s3d::vector<sreal> y(A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2));
#pragma omp parallel for simd default(shared)
	for(sint i = 0; i < A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2); i++)
		y[i] = 0;
	
	const sint idxmax[3] = {r.start + r.sz[0], r.start + r.sz[1], r.start + r.sz[2]};

#pragma omp parallel default(shared) if(params.threadedapply)
	{
		for(int iswp = 0; iswp < params.napplysweeps; iswp++)
		{
			// Compute the triangular solve residual
			//  tres = r - y - L D^{-1} y
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
					//#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i-1),
							r.m->localFlattenedIndexAll(k,j-1,i),
							r.m->localFlattenedIndexAll(k-1,j,i),
						};
						const sint idxr = r.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);
						const sint jdxr[] = {
							x.m->localFlattenedIndexReal(k-ng,j-ng,i-ng-1),
							x.m->localFlattenedIndexReal(k-ng,j-ng-1,i-ng),
							x.m->localFlattenedIndexReal(k-ng-1,j-ng,i-ng),
						};

						tres[idx] = r.vals[idx] - y.vals[idx]
							- A.vals[0][idxr]*y[jdx[0]] * diaginv[jdxr[0]]
							- A.vals[1][idxr]*y[jdx[1]] * diaginv[jdxr[1]]
							- A.vals[2][idxr]*y[jdx[2]] * diaginv[jdxr[2]];
					}

			// multiply the residual temp with the approximate inverse and update, as
			//  y = y + M tres
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
					//#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i-1),
							r.m->localFlattenedIndexAll(k,j-1,i),
							r.m->localFlattenedIndexAll(k-1,j,i),
						};
						const sint idxr = r.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);

						y[idx] += tres[idx]
							- A.vals[0][idxr]*tres[jdx[0]]
							- A.vals[1][idxr]*tres[jdx[1]]
							- A.vals[2][idxr]*tres[jdx[2]];
					}
		}

#pragma omp parallel for simd default(shared)
		for(sint i = 0; i < A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2); i++)
			z.vals[i] = y[i];

		for(int iswp = 0; iswp < params.napplysweeps; iswp++)
		{
			// tres = y - D z - U z
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
					//#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i+1),
							r.m->localFlattenedIndexAll(k,j+1,i),
							r.m->localFlattenedIndexAll(k+1,j,i)
						};
						const sint idxr = r.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);

						tres[idx] = y.vals[idx] - z.vals[idx]/diaginv[idxr]
							- A.vals[0][idxr]*z[jdx[0]]
							- A.vals[1][idxr]*z[jdx[1]]
							- A.vals[2][idxr]*z[jdx[2]];
					}

			// z = z + M tres
#pragma omp for collapse(2) schedule(static, params.thread_chunk_size)
			for(sint k = r.start; k < idxmax[2]; k++)
				for(sint j = r.start; j < idxmax[1]; j++)
					//#pragma omp simd
					for(sint i = r.start; i < idxmax[0]; i++)
					{
						const sint idx = r.m->localFlattenedIndexAll(k,j,i);
						const sint jdx[] = {
							r.m->localFlattenedIndexAll(k,j,i+1),
							r.m->localFlattenedIndexAll(k,j+1,i),
							r.m->localFlattenedIndexAll(k+1,j,i)
						};
						const sint jdxr[] = {
							x.m->localFlattenedIndexReal(k-ng,j-ng,i-ng+1),
							x.m->localFlattenedIndexReal(k-ng,j-ng+1,i-ng),
							x.m->localFlattenedIndexReal(k-ng+1,j-ng,i-ng),
						};

						z.vals[jdx[3]] += diaginv[idxr] * ( tres[idx]
						                                    - A.vals[4][idxr]*tres.vals[jdx[0]]
						                                    - A.vals[5][idxr]*tres.vals[jdx[1]]
						                                    - A.vals[6][idxr]*tres.vals[jdx[2]] );
					}
		}
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
				const sint idxr = A.m->localFlattenedIndexReal(k,j,i);
				diaginv[idxr] = 1.0/A.vals[STENCIL_DIAG][idxr];
			}
}
