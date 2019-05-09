
#include "s3d_mr.hpp"

MinRes::MinRes(const SMat& lhs, SolverBase *const precond, const SolveParams params)
	: SolverBase(lhs, precond), sparams(params)
{ }

void MinRes::updateOperator()
{
	if(prec)
		prec->updateOperator();
}

/** The step-length is capped at 2.0.
 */
SolveInfo MinRes::apply(const SVec& b, SVec& x) const
{
	SolveInfo info;
	info.precapplywtime = 0;

	SVec res(A.m), p(A.m), v(A.m);

	printf("      Step       Rel res     step length\n");
	printf("----------------------------------------\n");
	fflush(stdout);

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

		sreal alpha = inner_vector_l2(res, p) / inner_vector_l2(p,p);

		if(alpha > 2.0) alpha = 2.0;

		vecaxpy(alpha, v, x);
		vecaxpy(-alpha, p, res);

		resnorm = norm_vector_l2(res);

		if(step % 10 == 0) {
			printf("      %4d       %.6e         %5.3g\n", step, resnorm/bnorm, alpha);
			fflush(stdout);
		}
		step++;
	}

	info.converged = resnorm/bnorm <= sparams.rtol ? true : false;
	info.iters = step;
	info.resnorm = resnorm;
	return info;
}
