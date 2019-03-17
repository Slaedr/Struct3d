/** \file
 * \brief Implementation of PDE factory
 */

#include "pdefactory.hpp"
#include "poisson.hpp"
#include "convdiff.hpp"
#include "convdiff_circular.hpp"

PDEBase *construct_pde(const CaseData& cdata)
{
	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson(cdata.btypes, cdata.bvals);
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff(cdata.btypes, cdata.bvals, cdata.vel, cdata.diffcoeff);
	else if(cdata.pdetype == "convdiff_circular")
		pde = new ConvDiffCirc(cdata.btypes, cdata.bvals, cdata.vel[0], cdata.diffcoeff);
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	return pde;
}
