/** \file
 * \brief Flexible GCR solver
 */

#ifndef S3D_GCR_H
#define S3D_GCR_H

#include "s3d_solverbase.hpp"

/// GCR iteration
class GCR : public SolverBase
{
public:
	/// Sets parameters
	GCR(const SMat& lhs, SolverBase *const precond, const SolveParams params);
	/// Updates the preconditioner
	void updateOperator();
	/// Solve
	SolveInfo apply(const SVec& b, SVec& x) const;

protected:
	SolveParams sparams;
};

#endif
