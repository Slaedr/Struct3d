/** \file
 * \brief ISAI preconditioners
 */

#ifndef STRUCT3D_ISAI_PRECONDITIONERS_H
#define STRUCT3D_ISAI_PRECONDITIONERS_H

#include "s3d_solverbase.hpp"

/// Base class for preconditioners of the form M ~= ((D+L) D^(-1) (D+U))^-1 computed by ISAI
/** Incomplete Sparse Approximate Inverse
 */
class ISAI_preconditioner : public SolverBase
{
public:
	ISAI_preconditioner(const SMat& lhs, const PreconParams parms);

	/// Apply lower- followed by upper-triangular factors by ISAI-preconditioned Jacobi iterations
	/** The entries of the ISAI are computed on-the-fly during application
	 */
	SolveInfo apply(const SVec& r, SVec& z) const;

protected:
	/// Inverse of diagonal entries to be used for D^(-1) in the application
	s3d::vector<sreal> diaginv;
	/// Parameters for preconditioner application
	const PreconParams params;

private:
	/// Temporary vector to store triangular system residual
	mutable s3d::vector<sreal> tres;
	/// Solution of forward triangular solve
	mutable s3d::vector<sreal> y;
};

/// Symmetric Gauss-Seidel (SGS) preconditioner approximately applied by ISAI
class SGS_ISAI_preconditioner : public ISAI_preconditioner
{
public:
	/// Sets data
	SGS_ISAI_preconditioner(const SMat& lhs, const PreconParams parms);
	/// Updates the preconditioner - in this case, computes and stores inverses of diagonal entries
	void updateOperator();
};

#endif
