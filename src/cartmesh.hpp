/** \file cartmesh.hpp
 * \brief Specification of single-block but distributed, non-uniform Cartesian grid using PETSc
 * \author Aditya Kashi
 */

#ifndef STRUCT3D_CARTMESH_H
#define STRUCT3D_CARTMESH_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include <petscdm.h>
#include <petscdmda.h>

#include "struct3d_config.h"

/// Non-uniform Cartesian grid
/** 
 * We store the on-dimensional locations of points along 3 orthogonal vectors whose
 * tensor product defines the grid. So, we store only 
 * - the x-coordinates of points lying along the "back lower horizontal" line of the cube
 *     containing the mesh
 * - the y-coordinates of points lying along the "back upper vertical" line of the cube
 * - the z-coordinates of points lying along the line "coming out" of the "bottom-left" corner 
 *     of the "back" plane
 */
class CartMesh
{
protected:
	/// Array storing the number of points on each coordinate axis
	sint npoind[NDIM];

	/// Stores an array for each of the 3 axes 
	/// coords[i][j] refers to the j-th node along the i-axis
	sreal ** coords;			
	
	sint npointotal;                ///< Total number of points in the grid
	sint nDomPoin;                   ///< Number of internal (non-boundary) points
	PetscReal h;                    ///< Mesh size parameter

	// Stuff related to multiprocess
	PetscMPIInt nprocs[NDIM];		///< Number of processors in each dimension
	PetscMPIInt ntprocs;			///< Total number of processors

	sint nghost;                    ///< Number of ghost points at each boundary, currently 1

	/// Computes the mesh size parameter h
	/** Sets h as the length of the longest diagonal of all cells.
	 */
	void computeMeshSize();

public:
	CartMesh();

	/// Create the mesh and set up the DMDA for it
	/** \param comm The communicator of all ranks across which to distribute the computation
	 * \param npdim Number of points along each dimension
	 */
	PetscErrorCode createMeshAndDMDA(const MPI_Comm comm, const PetscInt npdim[NDIM], 
	                                 PetscInt ndofpernode, PetscInt stencil_width,
	                                 DMBoundaryType bx, DMBoundaryType by, DMBoundaryType bz,
	                                 DMDAStencilType stencil_type,
	                                 DM *const dap);

	/// Simpler mesh setup for single-processor runs independent of PETSc
	int createMesh(const sint npdim[NDIM]);

	~CartMesh();

	/// Returns the number of points along a coordinate direction
	PetscInt gnpoind(const int idim) const
	{
		assert(idim < NDIM);
		return npoind[idim];
	}

	sint gnghost() const { return nghost; }

	/// Returns a coordinate of a grid point
	/** \param[in] idim The coordinate line along which the point to be queried lies
	 * \param[in] ipoin Index of the required point in the direction idim
	 */
	PetscReal gcoords(const int idim, const sint ipoin) const
	{
		assert(idim < NDIM);
		assert(ipoin < npoind[idim]);
		return coords[idim][ipoin];
	}

	PetscInt gnPoinTotal() const { return npointotal; }
	PetscInt gnDomPoin() const { return nDomPoin; }
	PetscReal gh() const { return h; }

	const sint *pointer_npoind() const
	{
		return npoind;
	}

	const sreal *const *pointer_coords() const
	{
		return coords;
	}

	/// Generate a non-uniform mesh in a cuboid corresponding to Chebyshev points in each direction
	/** For interval [a,b], a Chebyshev distribution of N points including a and b is computed as
	 * x_i = (a+b)/2 + (a-b)/2 * cos(pi - i*theta)
	 * where theta = pi/(N-1)
	 *
	 * \param[in] rmin Array containing the lower bounds of the domain along each coordinate
	 * \param[in] rmax Array containing the upper bounds of the domain along each coordinate
	 * \param[in] rank The MPI rank of the current process
	 */
	void generateMesh_ChebyshevDistribution(const sreal rmin[NDIM], const sreal rmax[NDIM]);
	
	/// Generates grid with uniform spacing
	/**
	 * \param[in] rmin Array containing the lower bounds of the domain along each coordinate
	 * \param[in] rmax Array containing the upper bounds of the domain along each coordinate
	 * \param[in] rank The MPI rank of the current process
	 */
	void generateMesh_UniformDistribution(const sreal rmin[NDIM], const sreal rmax[NDIM]);

	/// Returns the flattened 1D index of point at index (i,j,k). Excludes ghost points.
	/// \warning Note the reversed argument order.
	[[deprecated]]
	sint localFlattenedIndexReal(const sint k, const sint j, const sint i) const
		__attribute__((always_inline))
	{
		constexpr int ng = 1;
		return k*(npoind[1]-2*ng)*(npoind[0]-2*ng) + j*(npoind[0]-2*ng) + i;
	}

	/// Returns the flattened 1D index of point at index (i,j,k), including ghost points.
	/// \warning Note the reversed argument order.
	sint localFlattenedIndexAll(const sint k, const sint j, const sint i) const
		__attribute__((always_inline))
	{
		return k*npoind[1]*npoind[0] + j*npoind[0] + i;
	}
};


#endif
