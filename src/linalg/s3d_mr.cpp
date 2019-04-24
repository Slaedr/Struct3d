
#include "s3d_mr.hpp"

MinRes::MinRes(const SMat& lhs, SolverBase *const precond, const SolveParams params)
	: SolverBase(lhs, precond), sparams(params)
{ }

void MinRes::updateOperator()
{
	if(prec)
		prec->updateOperator();
}

SolveInfo MinRes::apply(const SVec& b, SVec& x) const
{
	SolveInfo info;
	info.precapplywtime = 0;

	SVec res(A.m), p(A.m), v(A.m);

	const sreal bnorm = norm_vector_l2(b);
	sreal resnorm = 1.0;

	int step = 0;

	A.apply_res(b, x, res);
	resnorm = norm_vector_l2(res);

	while(step < sparams.maxiter)
	{
		if(resnorm/bnorm < sparams.rtol)
			break;

		sreal starttime = MPI_Wtime();
		prec->apply(res,v);
		info.precapplywtime += MPI_Wtime()-starttime;

		A.apply(v,p);

		const sreal alpha = inner_vector_l2(res, p) / inner_vector_l2(p,p);

		vecaxpy(alpha, v, x);
		vecaxpy(-alpha, p, res);

		resnorm = norm_vector_l2(res);

		if(step % 5 == 0) {
			printf("      Step %d: Rel res = %g\n", step, resnorm/bnorm);
			fflush(stdout);
		}
		step++;
	}

	info.converged = resnorm/bnorm <= sparams.rtol ? true : false;
	info.iters = step;
	info.resnorm = resnorm;
	return info;
}
