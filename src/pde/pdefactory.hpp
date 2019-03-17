/** \file
 * \brief Construct a PDE from config settings
 */

#ifndef STRUCT3D_PDEFACTORY_H
#define STRUCT3D_PDEFACTORY_H

#include "case.hpp"
#include "pdebase.hpp"

PDEBase *construct_pde(const CaseData& cdata);

#endif
