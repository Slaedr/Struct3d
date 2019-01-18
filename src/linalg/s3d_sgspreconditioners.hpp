/** \file
 * \brief SGS-like preconditioners
 */

#ifndef STRUCT3D_SGS_PRECONDITIONERS_H
#define STRUCT3D_SGS_PRECONDITIONERS_H

#include "s3d_solverbase.hpp"

/// Base class for preconditioners of the form (D+L) D^(-1) (D+U)
class SGS_like_preconditioner : public SolverBase
{
public:
	SGS_like_preconditioner(const SMat& lhs, const PreconParams parms);
	SolveInfo apply(const SVec& r, SVec& z) const;

protected:
	/// Inverse of diagonal entries to be used as D^(-1) in the application
	s3d::vector<sreal> diaginv;
	/// Parameters for preconditioner application
	const PreconParams params;
};

/// Symmetric Gauss-Seidel (SGS) preconditioner
class SGS_preconditioner : public SGS_like_preconditioner
{
public:
	/// Sets data and computes the preconditioner by calling updateOperator
	SGS_preconditioner(const SMat& lhs, const PreconParams parms);
	/// Updates the preconditioner - in this case, computes and stores inverses of diagonal entries
	void updateOperator();
};

#endif
