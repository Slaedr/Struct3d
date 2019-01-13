/** \file
 * \brief Implementation of some matrix and vector operations
 */

#include <cassert>
#include "matvec.hpp"

SVec::SVec(const CartMesh *const mesh) : m{mesh}, start{1}, nghost{1},
                                         sz{m->gnpoind(0)-2, m->gnpoind(1)-2, m->gnpoind(2)-2}
{
	// allocate space for all points, real and ghost
	vals.resize((m->gnpoind(2))*(m->gnpoind(1))*(m->gnpoind(0)));

	// initialize to zero
	for(size_t i = 0; i < vals.size(); i++)
		vals[i] = 0;
}

/** Assumes compact stencil. No extra space is allocated.
 */
SMat::SMat(const CartMesh *const mesh) : m{mesh}, start{0}, nghost{1},
                                         sz{m->gnpoind(0)-2, m->gnpoind(1)-2, m->gnpoind(2)-2}
{
	for(int i = 0; i < NSTENCIL; i++)
		vals[i].resize((m->gnpoind(2)-2)*(m->gnpoind(1)-2)*(m->gnpoind(0)-2));
}

void SMat::apply(const SVec& x, SVec& y) const
{
	if(x.m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");

	for(sint k = x.start; k < x.start + x.sz[2]; k++)
		for(sint j = x.start; j < x.start + x.sz[1]; j++)
			for(sint i = x.start; i < x.start + x.sz[0]; i++)
			{
				const sint idxr = x.m->localFlattenedIndexReal(k,j,i);
				const sint jdx[] = {
					x.m->localFlattenedIndexAll(k,j,i-1),
					x.m->localFlattenedIndexAll(k,j-1,i),
					x.m->localFlattenedIndexAll(k-1,j,i),
					x.m->localFlattenedIndexAll(k,j,i),
					x.m->localFlattenedIndexAll(k,j,i+1),
					x.m->localFlattenedIndexAll(k,j+1,i),
					x.m->localFlattenedIndexAll(k+1,j,i)
				};

				for(int is = 0; is < NSTENCIL; is++)
					y.vals[jdx[3]] += vals[is][idxr] * x.vals[jdx[is]];
			}
}

void SMat::apply_res(const SVec& b, const SVec& x, SVec& y) const
{
	if(x.m != y.m || x.m != b.m || x.m != m)
		throw std::runtime_error("All vectors and matrix should be defined over the same mesh!");

	const int ng = x.nghost;
	assert(ng == 1);

	for(sint k = x.start; k < x.start + x.sz[2]; k++)
		for(sint j = x.start; j < x.start + x.sz[1]; j++)
			for(sint i = x.start; i < x.start + x.sz[0]; i++)
			{
				const sint idxr = x.m->localFlattenedIndexReal(k-ng,j-ng,i-ng);
				const sint jdx[] = {
					x.m->localFlattenedIndexAll(k,j,i-1),
					x.m->localFlattenedIndexAll(k,j-1,i),
					x.m->localFlattenedIndexAll(k-1,j,i),
					x.m->localFlattenedIndexAll(k,j,i),
					x.m->localFlattenedIndexAll(k,j,i+1),
					x.m->localFlattenedIndexAll(k,j+1,i),
					x.m->localFlattenedIndexAll(k+1,j,i)
				};

				y.vals[jdx[3]] = b.vals[jdx[3]];
				for(int is = 0; is < NSTENCIL; is++)
					y.vals[jdx[3]] -= vals[is][idxr] * x.vals[jdx[is]];
			}
}

void vecaxpy(const sreal a, const SVec& x, SVec& y)
{
	if(x.m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");

	for(sint k = x.start; k < x.start + x.sz[2]; k++)
		for(sint j = x.start; j < x.start + x.sz[1]; j++)
			for(sint i = x.start; i < x.start + x.sz[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				y.vals[idx] += a*x.vals[idx];
			}
}

sreal norm_L2(const SVec& x)
{
	sreal norm = 0;

	for(sint k = x.start; k < x.start + x.sz[2]; k++)
		for(sint j = x.start; j < x.start + x.sz[1]; j++)
			for(sint i = x.start; i < x.start + x.sz[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				const sreal vol = 1.0/8.0*(x.m->gcoords(0,i+1)-x.m->gcoords(0,i-1))
					*(x.m->gcoords(1,j+1)-x.m->gcoords(1,j-1))*(x.m->gcoords(2,k+1)-x.m->gcoords(2,k-1));
				norm += x.vals[idx]*x.vals[idx]*vol;
			}
	return sqrt(norm);
}

sreal norm_vector_l2(const SVec& x)
{
	sreal norm = 0;

	for(sint k = x.start; k < x.start + x.sz[2]; k++)
		for(sint j = x.start; j < x.start + x.sz[1]; j++)
			for(sint i = x.start; i < x.start + x.sz[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				norm += x.vals[idx]*x.vals[idx];
			}
	return sqrt(norm);
}
