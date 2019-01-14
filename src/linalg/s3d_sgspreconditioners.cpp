/** \file
 * \brief Implementation of SGS-like application and SGS preconditioner
 */

#include "s3d_sgspreconditioners.hpp"

SGS_like_preconditioner::SGS_like_preconditioner(const SMat& lhs, const PreconParams parms)
	: SolverBase(lhs, nullptr), params(parms)
{
	diaginv.resize((A.m->gnpoind(0)-2)*(A.m->gnpoind(1)-2)*(A.m->gnpoind(2)-2));
	y.assign(A.m->gnpoind(0)*A.m->gnpoind(1)*A.m->gnpoind(2), 0);
}

SolveInfo SGS_like_preconditioner::apply(const SVec& r, SVec& z) const
{
	if(r.m != z.m || r.m != A.m)
		throw std::runtime_error("apply: Vectors and matrix must be defined over the same mesh!");

	const int ng = r.nghost;

	for(int iswp = 0; iswp < params.napplysweeps; iswp++)
	{
		for(sint k = r.start; k < r.start + r.sz[2]; k++)
			for(sint j = r.start; j < r.start + r.sz[1]; j++)
				for(sint i = r.start; i < r.start + r.sz[0]; i++)
				{
					const sint idxr = r.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);
					const sint jdx[] = {
						r.m->localFlattenedIndexAll(k,j,i-1),
						r.m->localFlattenedIndexAll(k,j-1,i),
						r.m->localFlattenedIndexAll(k-1,j,i),
						r.m->localFlattenedIndexAll(k,j,i),
					};

					y[jdx[3]] = r.vals[jdx[3]];
					for(int is = 0; is < STENCIL_DIAG; is++)
						y[jdx[3]] -= A.vals[is][idxr] * y[jdx[is]];
					y[jdx[3]] *= diaginv[idxr];
				}

		for(sint k = r.start; k < r.start + r.sz[2]; k++)
			for(sint j = r.start; j < r.start + r.sz[1]; j++)
				for(sint i = r.start; i < r.start + r.sz[0]; i++)
				{
					const sint idxr = r.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);
					const sint jdx[] = {
						-1, -1, -1,
						r.m->localFlattenedIndexAll(k,j,i),
						r.m->localFlattenedIndexAll(k,j,i+1),
						r.m->localFlattenedIndexAll(k,j+1,i),
						r.m->localFlattenedIndexAll(k+1,j,i)
					};

					z.vals[jdx[3]] = y[jdx[3]];
					for(int is = STENCIL_DIAG + 1; is < NSTENCIL; is++)
						z.vals[jdx[3]] -= diaginv[idxr]*A.vals[is][idxr] * z.vals[jdx[is]];
				}
	}

	return {false, 1, -1.0};
}

SGS_preconditioner::SGS_preconditioner(const SMat& lhs, const PreconParams parms)
	: SGS_like_preconditioner(lhs, parms)
{
	updateOperator();
}

void SGS_preconditioner::updateOperator()
{
	for(sint k = A.start; k < A.start + A.sz[2]; k++)
		for(sint j = A.start; j < A.start + A.sz[1]; j++)
			for(sint i = A.start; i < A.start + A.sz[0]; i++)
			{
				const sint idxr = A.m->localFlattenedIndexReal(k,j,i);
				diaginv[idxr] = 1.0/A.vals[STENCIL_DIAG][idxr];
			}
}

