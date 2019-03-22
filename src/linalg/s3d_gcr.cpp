
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

	const int north = sparams.restart+1;

	SVec res(A.m), z(A.m), az(A.m);
	std::vector<SVec> p(north), q(north);
	for(int i = 0; i < north; i++) {
		p[i].init(A.m);
		q[i].init(A.m);
	}

	const sreal bnorm = norm_vector_l2(b);

	int step = 0;

	while(step < sparams.maxiter)
	{
		A.apply_res(b, x, res);
		sreal resnorm = norm_vector_l2(res);

		sreal starttime = MPI_Wtime();
		prec->apply(res,p[0]);
		info.precapplywtime += MPI_Wtime()-starttime;

		vecset(0,q[0]);
		A.apply(p[0],q[0]);

		for(int k = 0; k < north; k++)
		{
			const sreal alpha = inner_vector_l2(res,q[k])/inner_vector_l2(q[k],q[k]);
			vecaxpy(alpha, p[k], x);
			vecaxpy(-alpha, q[k], res);

			resnorm = norm_vector_l2(res);
			if(resnorm/bnorm < sparams.rtol)
				break;

			starttime = MPI_Wtime();
			prec->apply(res, z);
			info.precapplywtime += MPI_Wtime()-starttime;

			//std::vector<sreal> beta(k);

			step++;
		}

		if(resnorm/bnorm < sparams.rtol)
			break;
	}

	return info;
}
