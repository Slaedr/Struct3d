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
	for(sint k = r.start; k < r.start + r.sz[2]; k++)
		for(sint j = r.start; j < r.start + r.sz[1]; j++)
			for(sint i = r.start; i < r.start + r.sz[0]; i++)
			{
				const sint idxr = r.m->localFlattenedIndexReal(k-1,j-1,i-1);
				const sint idx = r.m->localFlattenedIndexAll(k,j,i);
				z.vals[idx] = r.vals[idx] / A.vals[3][idxr];
			}

	return {false, 1, 1.0};
}
