/** \file
 * \brief Implementation of Richardson iteration
 */

#include <cassert>
#include "s3d_solverbase.hpp"

SolverBase::SolverBase(const SMat& lhs, SolverBase *const precond) : A(lhs), prec{precond}
{ }

NoSolver::NoSolver(const SMat& lhs, SolverBase *const precond) : SolverBase(lhs,nullptr)
{ }

void NoSolver::updateOperator() { }

void NoSolver::apply(const SVec& b, SVec& x) const
{
	if(b.m != x.m)
		throw std::runtime_error("Arguments must be defined over a common mesh!");
	assert(b.start == x.start);
	assert(b.nghost == x.nghost);
	assert(b.vals.size() == x.vals.size());

	for(size_t i = 0; i < b.vals.size(); i++)
		x.vals[i] = b.vals[i];
}

Richardson::Richardson(const SMat& lhs, SolverBase *const precond, const SolveParams params)
	: SolverBase(lhs, precond), sparams(params)
{
	updateOperator();
}

void Richardson::updateOperator()
{
	prec->updateOperator();
}

void Richardson::apply(const SVec& b, SVec& x) const
{
	SVec res(A.m), dx(A.m);
	const sreal bnorm = norm_vector_l2(b);

	A.apply_res(b, x, res);
	sreal resnorm = norm_vector_l2(res);

	int step = 0;
	while(resnorm/bnorm > sparams.rtol && step < sparams.maxiter)
	{
		prec->apply(res, dx);
		vecaxpy(1.0, dx, x);

		A.apply_res(b, x, res);
		resnorm = norm_vector_l2(res);
	}
}