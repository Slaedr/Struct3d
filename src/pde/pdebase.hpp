/** \file
 * \brief Base class for PDE-based problems
 */

#ifndef PDEBASE_H
#define PDEBASE_H

#include <array>
#include <functional>
#include "cartmesh.hpp"

/// Abstract class for all PDE discretizations on Cartesian grids
class PDEBase
{
public:
	PDEBase() { }
	virtual ~PDEBase() { }

	/// Computes a grid vector from a given function of space coordinates
	/** Only interior point values are set.
	 * \param m The mesh
	 * \param da The DMDA that the output Vec has been build with
	 * \param func The function whose discrete representation is desired
	 * \param[in,out] f The vec that must be written to
	 */
	virtual int computeVector(const CartMesh *const m, DM da,
	                          const std::function<sreal(const sreal[NDIM])> func,
	                          Vec f) const;

	/// Prescribes computation of the left-hand side operator for specific PDEs
	virtual int computeLHS(const CartMesh *const m, DM da, Mat A) const = 0;

	/// Prescribes generation of a pair of functions: the first being the solution and
	///  the second being the right hand side
	virtual std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const = 0;
};

#endif