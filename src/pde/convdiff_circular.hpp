/** \file
 * \brief Class for convection-diffusion problem with rotational advection
 */

#ifndef STRUCT3D_CONVDIFF_CIRCULAR_H
#define STRUCT3D_CONVDIFF_CIRCULAR_H

#include "pdebase.hpp"

/// Solves b.grad u - p div grad u = f where b is a circular (but 3D) velocity field
class ConvDiffCirc : public PDEBase
{
public:
	/**
	 * \param advel Advection velocity magnitude
	 * \param diffusion_coeff Diffusion coefficient
	 */
	ConvDiffCirc(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals,
	             const sreal advelmag, const sreal diffusion_coeff);

	/// Return the solution sin(2pi x)sin(2pi y)sin(2pi z) and the corresponding source term
	/** The first component is the solution and the second is the right hand side.
	 * The source term is, of course, dependent on the diffusion coeff and advection velocity.
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;

protected:
	const sreal mu;                      ///< Diffusion coeff
	const sreal bmag;                    ///< Magnitude of advection velocity

	/// Velocity as a function of space coordinates
	__attribute__((always_inline))
	std::array<sreal,NDIM> advectionVel(const sreal r[NDIM]) const;

	/// Kernel used for assembling the matrix
	void lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
	                   const sint nghost,
	                   sreal& v0, sreal& v1, sreal& v2, sreal& v3, sreal& v4, sreal& v5, sreal& v6)
		const;

	sreal rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& func,
	                 const sint i, const sint j, const sint k) const;
};

#endif
