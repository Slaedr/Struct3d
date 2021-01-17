/** \file
 * \brief Class for convection-diffusion problem
 */

#ifndef STRUCT3D_CONVDIFF_H
#define STRUCT3D_CONVDIFF_H

#include "pdeimpl.hpp"

class ConvDiff;

/// Solves b.grad u - p div grad u = f
class ConvDiff : public PDEImpl<ConvDiff>
{
public:
	/**
	 * \param advel Advection velocity vector
	 * \param diffusion_coeff Diffusion coefficient
	 */
	ConvDiff(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals,
	         const std::array<sreal,NDIM> advel, const sreal diffusion_coeff);

	/// Return the solution sin(2pi x)sin(2pi y)sin(2pi z) and the corresponding source term
	/** The source term is, of course, dependent on the diffusion coeff and advection velocity.
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;

	std::function<sreal(const sreal[NDIM])> test_rhs() const;

	/// Kernel used for assembling the matrix
	//#pragma omp declare simd uniform(this,m,nghost,j,k) linear(i:1) notinbranch
	void lhsmat_kernel(const CartMesh *m, sint i, sint j, sint k,
	                   sint nghost, sreal *__restrict v) const;

	/// Kernel used for assembling the right hand side vector using source and BCs
	sreal rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& sourcef,
	                 const sint i, const sint j, const sint k) const;

protected:
	const sreal mu;                      ///< Diffusion coeff
	const std::array<sreal,NDIM> b;      ///< Advection velocity (normalized)
};

#endif
