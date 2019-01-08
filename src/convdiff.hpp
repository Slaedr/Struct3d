/** \file
 * \brief Class for convection-diffusion problem
 */

#ifndef STRUCT3D_CONVDIFF_H
#define STRUCT3D_CONVDIFF_H

#include "pdebase.hpp"

/// Solves b.grad u - p div grad u = f where b = (1/v3,1/v3,1/v3) and p is the Peclet number
/** The Peclet number if currently one, but the solution still has a kind of boudary layer:
 * u = sin(2*pi*x^5)*sin(2*pi*y^5)
 */
class ConvDiff : public PDEBase
{
public:
	ConvDiff(const sreal peclet_number);

	int computeLHS(const CartMesh *const m, DM da, Mat A) const;

	/// Return the solution sin(2pi x)sin(2pi y)sin(2pi z) and the corresponding source term
	/** The source term is, of course, dependent on the Peclet number and advection velocity.
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;

protected:
	const sreal peclet;                  ///< Peclet number
	const std::array<sreal,NDIM> b;      ///< Advection velocity (normalized)
};

#endif
