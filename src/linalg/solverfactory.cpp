/** \file
 * \brief Implementation of a solver factory
 */

#include <cstring>
#include <petscsys.h>
#include "common_utils.hpp"
#include "s3d_jacobi.hpp"
#include "s3d_sgspreconditioners.hpp"
#include "s3d_ilu.hpp"
#include "s3d_gcr.hpp"
#include "s3d_mr.hpp"
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
	try {
		params.restart = petscoptions_get_int("-ksp_restart");
	} catch(NonExistentPetscOpion& e) {
		printf(" No KSP restart specified, using 30 if necessary.\n");
		params.restart = 30;
	}

	if(precstr == "jacobi")
	{
		printf("  Using Jacobi preconditioner\n");
		prec = new JacobiPreconditioner(lhs);
	}
	else if(precstr == "sgs")
	{
		printf("  Using SGS preconditioner\n");
		PreconParams pparams;
		pparams.nbuildsweeps = 0;
		pparams.napplysweeps = petscoptions_get_int("-s3d_pc_apply_sweeps");
		pparams.thread_chunk_size = petscoptions_get_int("-s3d_thread_chunk_size");
		pparams.threadedbuild = false;
		try {
			pparams.threadedapply = petscoptions_get_bool("-s3d_pc_use_threaded_apply");
		} catch(NonExistentPetscOpion& e) {
			pparams.threadedapply = true;
			printf(" WARNING: SolverFactory: Using threaded apply.\n");
		}
		prec = new SGS_preconditioner(lhs, pparams);
	}
	else if(precstr == "strilu")
	{
		printf("  Using StrILU preconditioner\n");
		PreconParams pparams;
		pparams.nbuildsweeps = petscoptions_get_int("-s3d_pc_build_sweeps");
		pparams.napplysweeps = petscoptions_get_int("-s3d_pc_apply_sweeps");
		pparams.thread_chunk_size = petscoptions_get_int("-s3d_thread_chunk_size");
		try {
			pparams.threadedbuild = petscoptions_get_bool("-s3d_pc_use_threaded_build");
		} catch(NonExistentPetscOpion& e) {
			pparams.threadedbuild = true;
			printf(" WARNING: SolverFactory: Using threaded build.\n");
		}
		try {
			pparams.threadedapply = petscoptions_get_bool("-s3d_pc_use_threaded_apply");
		} catch(NonExistentPetscOpion& e) {
			pparams.threadedapply = true;
			printf(" WARNING: SolverFactory: Using threaded apply.\n");
		}
		prec = new StrILU_preconditioner(lhs, pparams);
	}
	else {
		prec = new NoSolver(lhs);
		printf("WARNING: createSolver: Using no preconditioner");
	}

	if(tlsolver == "richardson")
	{
		printf("  Using Richardson solver\n");
		solver = new Richardson(lhs, prec, params);
	}
	else if(tlsolver == "gcr") {
		printf("  Using GCR solver\n");
		solver = new GCR(lhs, prec, params);
	}
	else if(tlsolver == "mr") {
		printf("  Using MinRes solver\n");
		solver = new MinRes(lhs, prec, params);
	}
	else {
		throw std::runtime_error("Need a valid solver option!");
	}

	return solver;
}
