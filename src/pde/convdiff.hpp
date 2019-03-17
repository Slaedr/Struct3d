/** \file
 * \brief Class for convection-diffusion problem
 */

#ifndef STRUCT3D_CONVDIFF_H
#define STRUCT3D_CONVDIFF_H

#include "pdebase.hpp"

/// Solves b.grad u - p div grad u = f
class ConvDiff : public PDEBase
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

protected:
	const sreal mu;                      ///< Diffusion coeff
	const std::array<sreal,NDIM> b;      ///< Advection velocity (normalized)

	/// Kernel used for assembling the matrix
	void lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
	                   const sint nghost,
	                   sreal& v0, sreal& v1, sreal& v2, sreal& v3, sreal& v4, sreal& v5, sreal& v6)
		const;

	/// Kernel used for assembling the right hand side vector using source and BCs
	sreal rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& sourcef,
	                 const sint i, const sint j, const sint k) const;
};

#endif
