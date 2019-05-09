
#include "s3d_gcr.hpp"

GCR::GCR(const SMat& lhs, SolverBase *const precond, const SolveParams params)
	: SolverBase(lhs, precond), sparams(params)
{ }

void GCR::updateOperator()
{
	if(prec)
		prec->updateOperator();
}

SolveInfo GCR::apply(const SVec& b, SVec& x) const
{
	SolveInfo info;
	info.precapplywtime = 0;

	const int north = sparams.restart;

	SVec res(A.m), z(A.m);
	std::vector<SVec> p(north), q(north);
	for(int i = 0; i < north; i++) {
		p[i].init(A.m);
		q[i].init(A.m);
	}

	const sreal bnorm = norm_vector_l2(b);
	sreal resnorm = 1.0;

	int step = 0;

	while(step < sparams.maxiter)
	{
		A.apply_res(b, x, res);
		resnorm = norm_vector_l2(res);

		sreal starttime = MPI_Wtime();
		prec->apply(res,p[0]);
		info.precapplywtime += MPI_Wtime()-starttime;

		A.apply(p[0],q[0]);

		for(int k = 0; k < north; k++)
		{
			const sreal alpha = inner_vector_l2(res,q[k])/inner_vector_l2(q[k],q[k]);
			vecaxpy(alpha, p[k], x);
			vecaxpy(-alpha, q[k], res);

			resnorm = norm_vector_l2(res);
			if(step % 5 == 0) {
				printf("      Step %d: Rel res = %g\n", step, resnorm/bnorm);
				fflush(stdout);
			}
			step++;

			if(resnorm/bnorm < sparams.rtol)
				break;
			if(k == north-1)
				break;
			if(step >= sparams.maxiter)
				break;

			starttime = MPI_Wtime();
			prec->apply(res, z);
			info.precapplywtime += MPI_Wtime()-starttime;

			A.apply(z, q[k+1]);
			vecassign(z, p[k+1]);

			std::vector<sreal> beta(k+1);
			for(int i = 0; i < k+1; i++) {
				beta[i] = -inner_vector_l2(q[k+1], q[i]) / inner_vector_l2(q[i], q[i]);
			}

			vec_multi_axpy(k+1, &beta[0], &p[0], p[k+1]);
			vec_multi_axpy(k+1, &beta[0], &q[0], q[k+1]);
		}

		if(resnorm/bnorm < sparams.rtol)
			break;
	}

	info.converged = resnorm/bnorm <= sparams.rtol ? true : false;
	info.iters = step;
	info.resnorm = resnorm;
	return info;
}
