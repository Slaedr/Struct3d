/** \file
 * \brief Test scaling w.r.t. threads on CPU
 */

#ifndef STRUCT3D_SCALING_OPENMP_H
#define STRUCT3D_SCALING_OPENMP_H

#include <vector>
#include "pde/pdebase.hpp"

/// Relevant data about a case run by a certain number of threads, with a certain number of build
///  sweeps and a certain number of apply sweeps
struct ThreadCaseInfo
{
	int nthreads;
	int nbuildsweeps;
	int napplysweeps;

	int avg_niter;
	sreal avg_residual;

	double avg_precbuildtime;
	double avg_precapplytime;
	double avg_solvetime;

	double dev_precbuildtime;
	double dev_precapplytime;
	double dev_niter;
	sreal dev_residual;
};

/// Run a sequence of OpenMP cases
/** Threads and sweeps are read from the PETSc options database. At most 30 of each are read.
 * * -openmptest_threads_list
 * * -openmptest_build_sweeps
 * * -openmptest_apply_sweeps
 * Note that the number of entries in the build-sweeps and apply-sweeps arrays must be the same.
 * Further, the number of threads and sweeps to use for the reference run are needed.
 * * -openmptest_ref_threads
 * * -openmptest_ref_build_sweeps
 * * -openmptest_ref_apply_sweeps
 * The number of repetitions to perform for each test run and the reference run
 * * -openmptest_num_repeats
 * * -openmptest_ref_num_repeats
 * Also, a file path to write the results to is required.
 * -openmptest_output_file
 */
void runtests_openmp(const PDEBase *const pde, const CartMesh& m);

/// Run a case at one setting
ThreadCaseInfo runcase(const CartMesh& m, const SMat& A, const SVec& b, const int nruns);

#endif
