/** \file
 * \brief Implementation for mesh-based Gauss-Seidel
 */

#include "s3d_gs.hpp"

GaussSeidelRelaxation::GaussSeidelRelaxation(const CartMesh& mesh, const int nas, const int tcs)
	: SolverBase(mesh), napplysweeps{nas}, threadchunksize{tcs}
{ }

void GaussSeidelRelaxation::compute()
{ }

void GaussSeidelRelaxation::apply(const sreal *const b, sreal *const x) const
{
	for(int iswp = 0; iswp < napplysweeps; iswp++)
	{
	}
}
