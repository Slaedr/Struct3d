/** \file
 * \brief ISAI preconditioners
 */

#ifndef STRUCT3D_ISAI_PRECONDITIONERS_H
#define STRUCT3D_ISAI_PRECONDITIONERS_H

#include "s3d_solverbase.hpp"

/// Base class for preconditioners of the form M ~= ((D+L) D^(-1) (D+U))^-1 computed by ISAI
class ISAI_preconditioner : public SolverBase
{
public:
	ISAI_preconditioner(const SMat& lhs, const PreconParams parms);
	SolveInfo apply(const SVec& r, SVec& z) const;

protected:
	/// Inverse of diagonal entries to be used as D^(-1) in the application
	s3d::vector<sreal> diaginv;
	/// Parameters for preconditioner application
	const PreconParams params;

private:
	/// Temporary vector
	mutable std::vector<sreal> tres;
};

/// Symmetric Gauss-Seidel (SGS) preconditioner approximately applied by ISAI
class SGS_ISAI_preconditioner : public ISAI_preconditioner
{
public:
	/// Sets data and computes the preconditioner by calling updateOperator
	SGS_ISAI_preconditioner(const SMat& lhs, const PreconParams parms);
	/// Updates the preconditioner - in this case, computes and stores inverses of diagonal entries
	void updateOperator();
};

#endif
