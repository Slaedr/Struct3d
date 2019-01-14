/** \file
 * \brief Implementation of a solver factory
 */

#include <cstring>
#include <petscsys.h>
#include "common_utils.hpp"
#include "s3d_jacobi.hpp"
#include "solverfactory.hpp"

SolverBase *createSolver(const SMat& lhs)
{
	SolverBase *solver = nullptr;
	SolverBase *prec = nullptr;

	const std::string tlsolver = petscoptions_get_string("-s3d_ksp_type", 20);

	std::string precstr;
	try {
		precstr = petscoptions_get_string("-s3d_pc_type", 30);
	} catch (NonExistentPetscOpion& e) {
		printf("No preconditioner\n");
		precstr = "none";
	}

	SolveParams params;
	params.rtol = petscoptions_get_real("-ksp_rtol");
	params.maxiter = petscoptions_get_int("-ksp_max_it");

	if(precstr == "jacobi")
	{
		prec = new JacobiPreconditioner(lhs);
	}
	else {
		prec = new NoSolver(lhs);
		printf("WARNING: createSolver: Using no preconditioner");
	}

	if(tlsolver == "richardson")
	{
		solver = new Richardson(lhs, prec, params);
	}
	else {
		throw std::runtime_error("Need a valid solver option!");
	}

	return solver;
}
