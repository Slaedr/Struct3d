/** \file
 * \brief Implementation of some matrix and vector operations
 */

#include "matvec.hpp"

SVec::SVec(const CartMesh *const mesh) : m{mesh}
{
	nghost = 1;                       // assuming compact stencil
	start = 1;                        // leave space for one ghost layer

	// the formal sizes only reflect real points:
	for(int i = 0; i < NDIM; i++) {
		sz[i] = m->gnpoind(i)-2;
	}

	// allocate space for all points, real and ghost
	vals.resize((m->gnpoind(2))*(m->gnpoind(1))*(m->gnpoind(0)));
}

SMat::SMat(const CartMesh *const mesh) : m{mesh}
{
	nghost = 1;                       // assuming compact stencil
	start = 0;
	for(int i = 0; i < NDIM; i++) {
		sz[i] = m->gnpoind(i)-2;
	}

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
	if(x.m != y.m || x.m != b.m)
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
					y.vals[jdx[3]] = b.vals[jdx[3]] - vals[is][idxr] * x.vals[jdx[is]];
			}
}
