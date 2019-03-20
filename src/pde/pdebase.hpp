/** \file
 * \brief Base class for PDE-based problems
 */

#ifndef PDEBASE_H
#define PDEBASE_H

#include <array>
#include <functional>
#include "cartmesh.hpp"
#include "linalg/matvec.hpp"

/// Abstract class for all PDE discretizations on Cartesian grids
class PDEBase
{
public:
	PDEBase(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals);

	virtual ~PDEBase() { }

	/// Computes a grid vector from a given function of space coordinates
	/** Only interior point values are set.
	 * \param m The mesh
	 * \param da The DMDA that the output Vec has been build with
	 * \param func The function whose discrete representation is desired
	 * \param[in,out] f The vec that must be written to
	 */
	int computeVectorPetsc(const CartMesh *const m, DM da,
	                       const std::function<sreal(const sreal[NDIM])> func, Vec f) const;

	/// Computes a grid vector same as computeVectorPetsc, but using our native SVec instead
	SVec computeVector(const CartMesh *const m,
	                   const std::function<sreal(const sreal[NDIM])> func) const;

	/// Prescribes computation of the left-hand side operator for specific PDEs
	int computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const;

	/// Prescribes computation of the left-hand side operator for specific PDEs, using native format
	SMat computeLHS(const CartMesh *const m) const;

	/// Prescribes generation of a pair of functions: the first being the solution and
	///  the second being the right hand side
	virtual std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const = 0;

protected:
	/// Boundary conditions at each of the 6 faces
	std::array<BCType,6> bctypes;
	/// Boundary values at each of the 6 faces
	std::array<sreal,6> bvals;

	/// Kernel used for assembling the matrix
	virtual void lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
	                           const sint nghost, sreal *const __restrict v) const = 0;

	/// Should return the value of the right-hand side vector at the (i,j,k) node
	virtual sreal rhs_kernel(const CartMesh *const m,
	                         const std::function<sreal(const sreal[NDIM])>& source_function,
	                         const sint i, const sint j, const sint k) const = 0;
};

#endif
