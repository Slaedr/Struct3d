/** \file
 * \brief Minimal residual (line-search) solver
 */

#ifndef S3D_MRES_H
#define S3D_MRES_H

#include "s3d_solverbase.hpp"

/// Minimum residual iteration with right preconditioning
class MinRes : public SolverBase
{
public:
	/// Sets parameters
	MinRes(const SMat& lhs, SolverBase *const precond, const SolveParams params);
	/// Updates the preconditioner
	void updateOperator();
	/// Solve
	SolveInfo apply(const SVec& b, SVec& x) const;

protected:
	SolveParams sparams;
};

#endif
