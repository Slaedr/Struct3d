/** \file
 * \brief Jacobi implementation
 */

#include "s3d_jacobi.hpp"

JacobiPreconditioner::JacobiPreconditioner(const SMat& lhs)
	: SolverBase(lhs, nullptr)
{ }

void JacobiPreconditioner::updateOperator()
{ }

SolveInfo JacobiPreconditioner::apply(const SVec& r, SVec& z) const
{
	const sint idxmax[3] = {r.start + r.sz[0], r.start + r.sz[1], r.start + r.sz[2]};
#pragma omp parallel for collapse(2) default(shared)
	for(sint k = r.start; k < idxmax[2]; k++)
		for(sint j = r.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = r.start; i < idxmax[0]; i++)
			{
				const sint idxr = r.m->localFlattenedIndexReal(k-1,j-1,i-1);
				const sint idx = r.m->localFlattenedIndexAll(k,j,i);
				z.vals[idx] = r.vals[idx] / A.vals[3][idxr];
			}

	return {false, 1, 1.0};
}
