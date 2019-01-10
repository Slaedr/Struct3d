/** \file
 * \brief Class for convection-diffusion problem
 */

#ifndef STRUCT3D_CONVDIFF_H
#define STRUCT3D_CONVDIFF_H

#include "pdebase.hpp"

/// Solves b.grad u - p div grad u = f where b = (1/v3,1/v3,1/v3) and p is the diffusion coefficient
class ConvDiff : public PDEBase
{
public:
	/**
	 * \param advel Advection velocity vector
	 * \param diffusion_coeff Diffusion coefficient
	 */
	ConvDiff(const std::array<sreal,NDIM> advel, const sreal diffusion_coeff);

	int computeLHS(const CartMesh *const m, DM da, Mat A) const;

	/// Return the solution sin(2pi x)sin(2pi y)sin(2pi z) and the corresponding source term
	/** The source term is, of course, dependent on the diffusion coeff and advection velocity.
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;

protected:
	const sreal mu;                      ///< Diffusion coeff
	const std::array<sreal,NDIM> b;      ///< Advection velocity (normalized)
};

#endif
