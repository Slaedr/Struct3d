/** \file
 * \brief Jacobi relaxation and preconditioner
 */

#ifndef STRUCT3D_JACOBI_H
#define STRUCT3D_JACOBI_H

#include "s3d_solverbase.hpp"

class JacobiPreconditioner : public SolverBase
{
public:
	JacobiPreconditioner(const SMat& lhs);
	void updateOperator();
	SolveInfo apply(const SVec& b, SVec& x) const;
};

#endif
