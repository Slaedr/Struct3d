/** \file
 * \brief Class for convection-diffusion problem
 */

#ifndef STRUCT3D_CONVDIFF_H
#define STRUCT3D_CONVDIFF_H

/// Solves b.grad u - p div grad u = f where b = (1/v3,1/v3,1/v3) and p is the Peclet number
/** The Peclet number if currently one, but the solution still has a kind of boudary layer:
 * u = sin(2*pi*x^5)*sin(2*pi*y^5)
 */
class ConvDiff : public PDEBase
{
public:
	ConvDiff(const sreal peclet_number);

	int computeRHS(const CartMesh *const m, DM da, Vec f, Vec uexact) const;
	int computeLHS(const CartMesh *const m, DM da, Mat A) const;

protected:
	const sreal peclet;                  ///< Peclet number
	const sreal b[NDIM];                 ///< Advection velocity (normalized)
};

#endif
