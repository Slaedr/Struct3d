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

	const int ng = x.nghost;
	assert(ng == 1);

	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
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

				/* It turns out both GCC 8.2 and Clang 7 need the following loop manually unrolled
				 * in order to vectorize the i loop.
				 */ 
				// for(int is = 0; is < NSTENCIL; is++)
				// 	y.vals[jdx[3]] += vals[is][idxr] * x.vals[jdx[is]];

				y.vals[jdx[3]] += vals[0][idxr]*x.vals[jdx[0]]
					+ vals[1][idxr]*x.vals[jdx[1]]
					+ vals[2][idxr]*x.vals[jdx[2]]
					+ vals[3][idxr]*x.vals[jdx[3]]
					+ vals[4][idxr]*x.vals[jdx[4]]
					+ vals[5][idxr]*x.vals[jdx[5]]
					+ vals[6][idxr]*x.vals[jdx[6]];
			}
}

void SMat::apply_res(const SVec& b, const SVec& x, SVec& y) const
{
	if(x.m != y.m || x.m != b.m || x.m != m)
		throw std::runtime_error("All vectors and matrix should be defined over the same mesh!");

	const int ng = x.nghost;
	assert(ng == 1);
	
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
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

				/* It turns out both GCC 8.2 and Clang 7 need the following loop manually unrolled
				 * in order to vectorize the i loop.
				 */ 
				// for(int is = 0; is < NSTENCIL; is++)
				// 	y.vals[jdx[3]] -= vals[is][idxr] * x.vals[jdx[is]];

				y.vals[jdx[3]] -= vals[0][idxr]*x.vals[jdx[0]]
					+ vals[1][idxr]*x.vals[jdx[1]]
					+ vals[2][idxr]*x.vals[jdx[2]]
					+ vals[3][idxr]*x.vals[jdx[3]]
					+ vals[4][idxr]*x.vals[jdx[4]]
					+ vals[5][idxr]*x.vals[jdx[5]]
					+ vals[6][idxr]*x.vals[jdx[6]];
			}
}

void vecaxpy(const sreal a, const SVec& x, SVec& y)
{
	if(x.m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");
	
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				y.vals[idx] += a*x.vals[idx];
			}
}

sreal norm_L2(const SVec& x)
{
	sreal norm = 0;
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared) reduction(+:norm)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd reduction(+:norm)
			for(sint i = x.start; i < idxmax[0]; i++)
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
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared) reduction(+:norm)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd reduction(+:norm)
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				norm += x.vals[idx]*x.vals[idx];
			}
	return sqrt(norm);
}

sreal compute_error_L2(const SVec& x, const SVec& y)
{
	assert(x.m == y.m);
	SVec diff = y;
	vecaxpy(-1.0, x, diff);
	//return norm_vector_l2(diff)/sqrt(x.m->gnpointotal());
	return norm_L2(diff);
}
