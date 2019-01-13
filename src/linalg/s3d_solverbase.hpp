/** \file
 * \brief Base class for solvers
 */

#ifndef STRUCT3D_SOLVER_BASE_H
#define STRUCT3D_SOLVER_BASE_H

#include "cartmesh.hpp"
#include "matvec.hpp"

struct SolveParams {
	sreal rtol;
	int maxiter;
};

struct PreconParams {
	int nbuildsweeps;
	int napplysweeps;
	int thread_chunk_size;
	bool threadedbuild;
	bool threadedapply;
};

/// Abstract class for cartesian-mesh dependent solvers
class SolverBase
{
public:
	/// Set the matrix for the LHS and the preconditioner
	SolverBase(const SMat& lhs, SolverBase *const precond);

	/// For updating the solver when the LHS operator changes
	virtual void updateOperator() = 0;

	/// Solve a problem given a RHS vector
	virtual void apply(const SVec& b, SVec& x) const = 0;

protected:
	/// LHS matrix for the solver
	const SMat& A;
	/// Preconditioner context
	SolverBase *const prec;
};

/// Identity preconditioner
/** A "do-nothing" solver, for use as a dummy preconditioner in un-preconditioned solvers
 */
class NoSolver : public SolverBase
{
public:
	NoSolver(const SMat& lhs, SolverBase *const precond = nullptr);
	/// Does nothing
	void updateOperator();
	/// Just copies b into x
	void apply(const SVec& b, SVec& x) const;
};

class Richardson : public SolverBase
{
public:
	/// Builds the preconditioner
	Richardson(const SMat& lhs, SolverBase *const precond, const SolveParams params);
	/// Updates the preconditioner
	void updateOperator();
	/// Solve
	void apply(const SVec& b, SVec& x) const;

protected:
	SolveParams sparams;
};

#endif
