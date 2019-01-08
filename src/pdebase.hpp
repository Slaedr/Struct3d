/** \file
 * \brief Base class for PDE-based problems
 */

#ifndef PDEBASE_H
#define PDEBASE_H

class PDEBase
{
public:
	PDEBase() { }
	virtual ~PDEBase() { }

	virtual int computeRHS(const CartMesh *const m, DM da, Vec f, Vec uexact) const = 0;
	virtual int computeLHS(const CartMesh *const m, DM da, Mat A) const = 0;
};

#endif
