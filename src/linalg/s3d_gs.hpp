/** \file
 * \brief Gauss-Seidel solvers
 */

#ifndef STRUCT3D_GS_H
#define STRUCT3D_GS_H

#include "s3d_solverbase.hpp"

class GaussSeidelRelaxation : public SolverBase
{
public:
	GaussSeidelRelaxation(const CartMesh& mesh, const int nappplysweeps, const int threadchunksize);

	/// Does nothing, as pre-inversion of diagonal entries is not needed for scalar problems
	void compute();

	/// Solves Ax=b using relaxation
	void apply(const sreal *const b, sreal *const x) const;

protected:
	const int napplysweeps;
	const int threadchunksize;
};

#endif
