/** \file
 * \brief Vectors and matrices defined over Cartesian grids
 */

#ifndef STRUCT3D_MATVEC_H
#define STRUCT3D_MATVEC_H

#include <valarray>
#include <array>
#include "cartmesh.hpp"

/// Vector format for a 3D structured grid
/** The convention is that 'i' is the fastest-changing index
 */
struct SVec
{
	/// Construct the vector over the given Cartesian grid
	SVec(const CartMesh *const mesh);

	/// Associated mesh
	const CartMesh *const m;

	/// Starting position of 'real' points along each direction; those before this are ghosts
	/** This depends on the number of ghost points needed. Note that this is not a flattened index -
	 * its value will usually be 1 or 2.
	 */
	sint start;
	/// Number of 'real' points in each direction (0 is i, 1 is j and 2 is k)
	std::array<sint,NDIM> sz;
	/// Number of ghost points per boundary
	int nghost;

	/// 3D storage - access the value at point (i,j,k) as vals[localFlattenedIndex(k,j,i)]
	/** \sa CartMesh::localFlattenedIndex
	 */
	std::valarray<sreal> vals;
};

/// Matrix format for a stencil of size NSTENCIL on a 3D structured grid
/** The convention is that 'i' is the fastest-changing index while the stencil location is the
 * slowest.
 * 'Left' (i-1) entries for all points are stored first, followed by 'down' (j-1) entries for all
 * points, and so on.
 */
struct SMat
{
	/// Construct the matrix over the given Cartesian grid
	SMat(const CartMesh *const mesh);

	/// Associated mesh
	const CartMesh *const m;

	/// Starting position of 'real' points in each direction
	/** This depends on the number of ghost points needed. Note that this is not a flattened index -
	 * its value will usually be 0,1 or 2.
	 */
	sint start;
	/// Number of 'real' points in each of the directions (0 is i, 1 is j and 2 is k)
	std::array<sint,NDIM> sz;
	/// Number of ghost points per boundary
	int nghost;

	/// Non-zero values of the matrix - access as vals[<neighbor>][localFlattenedIndex(k,j,i)].
	/** \sa CartMesh::localFlattenedIndex
	 */
	std::valarray<sreal> vals[NSTENCIL];

	/// Compute a matrix-vector product y += Ax (\warning the result is ADDED TO, not overwritten)
	void apply(const SVec& x, SVec& y) const;

	/// Computes y = b - Ax (the results is overwritten)
	void apply_res(const SVec& b, const SVec& x, SVec& y) const;
};

#endif
