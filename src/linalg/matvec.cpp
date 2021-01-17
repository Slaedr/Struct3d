/** \file
 * \brief Implementation of some matrix and vector operations
 */

#include <cassert>
#include "matvec.hpp"

SVec::SVec(const CartMesh *const mesh) : m{mesh}, start{1}, nghost{1},
                                         sz{m->gnpoind(0)-2, m->gnpoind(1)-2, m->gnpoind(2)-2}
{
	if(m->gnghost() != nghost) {
		throw std::runtime_error("Num ghost points don't match between SVec and mesh!");
	}

	// allocate space for all points, real and ghost
	const sint ssize = (m->gnpoind(2))*(m->gnpoind(1))*(m->gnpoind(0));
	vals.resize(ssize);

	// initialize to zero
#pragma omp parallel for simd
	for(sint i = 0; i < ssize; i++)
		vals[i] = 0;
}

SVec::SVec() : m{nullptr}, start{1}, nghost{1}
{ }

void SVec::init(const CartMesh *const mesh)
{
	m = mesh;
	sz = {m->gnpoind(0)-2, m->gnpoind(1)-2, m->gnpoind(2)-2 };

	if(m->gnghost() != nghost) {
		throw std::runtime_error("Num ghost points don't match between SVec and mesh!");
	}

	// allocate space for all points, real and ghost
	vals.resize(m->gnPoinTotal());

	// initialize to zero
#pragma omp parallel for simd
	for(sint i = 0; i < m->gnPoinTotal(); i++)
		vals[i] = 0;
}

/** Assumes compact stencil. A row is allocated for each point, real or ghost.
 */
SMat::SMat(const CartMesh *const mesh) : m{mesh}, start{1}, nghost{1},
                                         sz{m->gnpoind(0)-2, m->gnpoind(1)-2, m->gnpoind(2)-2}
{
	if(m->gnghost() != nghost) {
		throw std::runtime_error("Num ghost points don't match between SMat and mesh!");
	}
	for(int i = 0; i < NSTENCIL; i++)
		vals[i].resize(m->gnPoinTotal());

	for(int i = 0; i < NSTENCIL; i++)
		for(int j = 0; j < m->gnPoinTotal(); j++)
			vals[i][j] = 0;
}

void SMat::apply(const SVec& x, SVec& y) const
{
	if(x.m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");

	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for default(shared) collapse(2)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
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
				// y.vals[jdx[3]] = 0;
				// for(int is = 0; is < NSTENCIL; is++)
				// 	y.vals[jdx[3]] += vals[is][idxr] * x.vals[jdx[is]];

				y.vals[idx] = vals[0][idx]*x.vals[jdx[0]]
					+ vals[1][idx]*x.vals[jdx[1]]
					+ vals[2][idx]*x.vals[jdx[2]]
					+ vals[3][idx]*x.vals[jdx[3]]
					+ vals[4][idx]*x.vals[jdx[4]]
					+ vals[5][idx]*x.vals[jdx[5]]
					+ vals[6][idx]*x.vals[jdx[6]];
			}
}

void SMat::apply_res(const SVec& b, const SVec& x, SVec& y) const
{
	if(x.m != y.m || x.m != b.m || x.m != m)
		throw std::runtime_error("All vectors and matrix should be defined over the same mesh!");

	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
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
				// y.vals[idx] = b.vals[idx];
				// for(int is = 0; is < NSTENCIL; is++)
				// 	y.vals[jdx[3]] -= vals[is][idxr] * x.vals[jdx[is]];

				y.vals[idx] = b.vals[idx] - ( vals[0][idx]*x.vals[jdx[0]]
				                              + vals[1][idx]*x.vals[jdx[1]]
				                              + vals[2][idx]*x.vals[jdx[2]]
				                              + vals[3][idx]*x.vals[jdx[3]]
				                              + vals[4][idx]*x.vals[jdx[4]]
				                              + vals[5][idx]*x.vals[jdx[5]]
				                              + vals[6][idx]*x.vals[jdx[6]] );
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

void vec_multi_axpy(const int numvecs, const sreal *const a, const SVec *const x, SVec& y)
{
	if(numvecs == 0)
		return;

	if(x[0].m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");
	
	const sint idxmax[3] = {y.start + y.sz[0], y.start + y.sz[1], y.start + y.sz[2]};

	for(int l = 0; l < numvecs; l++)
#pragma omp parallel for collapse(2) default(shared)
		for(sint k = y.start; k < idxmax[2]; k++)
			for(sint j = y.start; j < idxmax[1]; j++)
#pragma omp simd
				for(sint i = y.start; i < idxmax[0]; i++)
				{
					const sint idx = y.m->localFlattenedIndexAll(k,j,i);
					y.vals[idx] += a[l]*x[l].vals[idx];
				}
}

void vecset(const sreal a, SVec& x)
{
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				x.vals[idx] = a;
			}
}

void vecassign(const SVec& x, SVec& y)
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
				y.vals[idx] = x.vals[idx];
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

sreal inner_vector_l2(const SVec& x, const SVec& y)
{
	if(x.m != y.m)
		throw std::runtime_error("Both vectors should be defined over the same mesh!");
	if(x.sz[0] != y.sz[0] || x.sz[1] != y.sz[1] || x.sz[2] != y.sz[2])
		throw std::runtime_error("Sizes don't match!");
	assert(x.nghost == y.nghost);
	assert(x.start == y.start);

	sreal dot = 0;
	const sint idxmax[3] = {x.start + x.sz[0],x.start + x.sz[1], x.start + x.sz[2]};

#pragma omp parallel for collapse(2) default(shared) reduction(+:dot)
	for(sint k = x.start; k < idxmax[2]; k++)
		for(sint j = x.start; j < idxmax[1]; j++)
#pragma omp simd reduction(+:dot)
			for(sint i = x.start; i < idxmax[0]; i++)
			{
				const sint idx = x.m->localFlattenedIndexAll(k,j,i);
				dot += x.vals[idx]*y.vals[idx];
			}
	return dot;
}

sreal compute_error_L2(const SVec& x, const SVec& y)
{
	assert(x.m == y.m);
	SVec diff = y;
	vecaxpy(-1.0, x, diff);
	//return norm_vector_l2(diff)/sqrt(x.m->gnpointotal());
	return norm_L2(diff);
}
