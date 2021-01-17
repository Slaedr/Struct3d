/** \file
 * \brief Base class for PDE-based problems
 */

#ifndef STRUCT3D_PDEBASE_H
#define STRUCT3D_PDEBASE_H

#include <array>
#include <functional>
#include "cartmesh.hpp"
#include "linalg/matvec.hpp"

/// Abstract class for all PDE discretizations on Cartesian grids
class PDEBase
{
public:
	PDEBase(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals)
		: bctypes(bc_types), bvals(bc_vals)
	{
		printf("Boundary conditions:\n");
		for(int i = 0; i < 6; i++)
			printf("  %c : %f\n", (bctypes[i]==S3D_DIRICHLET ? 'D':'O'), bvals[i]);
	}

	virtual ~PDEBase() { }

	/// Computes a grid vector from a given function of space coordinates
	/** Only interior point values are set.
	 * \param m The mesh
	 * \param da The DMDA that the output Vec has been build with
	 * \param func The function whose discrete representation is desired
	 * \param[in,out] f The vec that must be written to
	 */
	virtual int computeVectorPetsc(const CartMesh *const m, DM da,
	                               const std::function<sreal(const sreal[NDIM])> func, Vec f) const = 0;

	/// Computes a grid vector same as computeVectorPetsc, but using our native SVec instead
	virtual SVec computeVector(const CartMesh *const m,
	                           const std::function<sreal(const sreal[NDIM])> func) const = 0;

	/// Prescribes computation of the left-hand side operator for specific PDEs
	virtual int computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const = 0;

	/// Prescribes computation of the left-hand side operator for specific PDEs, using native format
	virtual SMat computeLHS(const CartMesh *const m) const = 0;

	/// Prescribes generation of a pair of functions: the first being the solution and
	///  the second being the right hand side
	virtual std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const = 0;

	/// A source term function to make for a relatively challenging test case
	virtual std::function<sreal(const sreal[NDIM])> test_rhs() const = 0;

protected:
	/// Boundary conditions at each of the 6 faces
	std::array<BCType,6> bctypes;
	/// Boundary values at each of the 6 faces
	std::array<sreal,6> bvals;
};

#endif
