/** \file
 * \brief Implementation of Richardson iteration
 */

#include <cassert>
#include "s3d_solverbase.hpp"

SolverBase::SolverBase(const SMat& lhs, SolverBase *const precond) : A(lhs), prec{precond}
{ }

SolverBase::~SolverBase()
{
	delete prec;
}

NoSolver::NoSolver(const SMat& lhs, SolverBase *const precond) : SolverBase(lhs,nullptr)
{ }

void NoSolver::updateOperator() { }

SolveInfo NoSolver::apply(const SVec& b, SVec& x) const
{
	if(b.m != x.m)
		throw std::runtime_error("Arguments must be defined over a common mesh!");
	assert(b.start == x.start);
	assert(b.nghost == x.nghost);
	assert(b.vals.size() == x.vals.size());

	for(size_t i = 0; i < b.vals.size(); i++)
		x.vals[i] = b.vals[i];

	return {false, 1, 1.0};
}

Richardson::Richardson(const SMat& lhs, SolverBase *const precond, const SolveParams params)
	: SolverBase(lhs, precond), sparams(params)
{ }

void Richardson::updateOperator()
{
	if(prec)
		prec->updateOperator();
}

SolveInfo Richardson::apply(const SVec& b, SVec& x) const
{
	SVec res(A.m), dx(A.m);
	const sreal bnorm = norm_vector_l2(b);

	A.apply_res(b, x, res);
	sreal resnorm = norm_vector_l2(res);

	int step = 0;

	SolveInfo info;
	info.precapplywtime = 0;

	printf("      Step        Rel res    \n");
	printf("-----------------------------\n");
	fflush(stdout);

	while(resnorm/bnorm > sparams.rtol && step < sparams.maxiter)
	{
		sreal starttime = MPI_Wtime();
		prec->apply(res, dx);
		info.precapplywtime += MPI_Wtime()-starttime;

		vecaxpy(1.0, dx, x);

		A.apply_res(b, x, res);
		resnorm = norm_vector_l2(res);

		if(step % 10 == 0) {
			printf("        %4d          %.6e\n", step, resnorm/bnorm);
			fflush(stdout);
		}
		step++;
	}

	info.converged = resnorm/bnorm <= sparams.rtol ? true : false;
	info.iters = step;
	info.resnorm = resnorm;
	return info;
}
