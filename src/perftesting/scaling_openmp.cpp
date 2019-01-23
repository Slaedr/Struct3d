/** \file
 * \brief Implementation of one case run
 */

#include <cstdio>
#include <omp.h>
#include "linalg/solverfactory.hpp"
#include "common_utils.hpp"
#include "scaling_openmp.hpp"

/// Compute average of an array
template <typename T>
T average(const std::vector<T>& arr);

/// Compute the standard deviation of an array relative to its mean
template <typename T>
double relative_deviation(const std::vector<T> arr, const T avg);

/// Set the number of build- and apply-sweeps into the PETSc options database
static void setSweeps(const int buildsweeps, const int applysweeps);

/// Write info about the reference run and the column headers to the output file
static void writeReportHeader(const ThreadCaseInfo refrun, FILE *const fp);

/// Write performance metrics of runs with a particular setting to the output file
static void writeReportLine(const ThreadCaseInfo run, const ThreadCaseInfo refrun, FILE *const fp);

void runtests_openmp(const PDEBase *const pde, const CartMesh& m)
{
	// Set up
	const SVec b = pde->computeVector(&m, pde->manufactured_solution()[0]);
	const SMat A = pde->computeLHS(&m);

	// Read test params
	const int refthreads = petscoptions_get_int("-openmptest_ref_threads");
	const int refbuildsweeps = petscoptions_get_int("-openmptest_ref_build_sweeps");
	const int refapplysweeps = petscoptions_get_int("-openmptest_ref_apply_sweeps");
	const int nrepeats = petscoptions_get_int("-openmptest_num_repeats");
	const int nrefrepeats = petscoptions_get_int("-openmptest_ref_num_repeats");
	const std::string outfilepath = petscoptions_get_string("-openmptest_output_file",200);
	const std::vector<int> threadlist = petscoptions_get_array_int("-openmptest_threads_list",30);
	const std::vector<int> nbswplist = petscoptions_get_array_int("-openmptest_build_sweeps",30);
	const std::vector<int> naswplist = petscoptions_get_array_int("-openmptest_apply_sweeps",30);

	printf("Ref threads %d\n", refthreads);
	printf("Threads list %d, %d\n", threadlist[0], threadlist[1]);
	printf("Build sweeps list %d %d\n", nbswplist[0], nbswplist[1]);
	printf("Apply sweeps list %d %d\n", naswplist[0], naswplist[1]);

	if(nbswplist.size() != naswplist.size())
		throw std::runtime_error("Sizes of build sweeps list and apply sweeps list must be the same!");

	FILE *const fp = fopen(outfilepath.c_str(), "w");
	if(!fp) {
		printf("!Output file could not be opened!\n");
		exit(-1);
	}

	// Reference run
	omp_set_num_threads(refthreads);
	setSweeps(refbuildsweeps, refapplysweeps);
	const ThreadCaseInfo refrun = runcase(m, A, b, nrefrepeats);
	writeReportHeader(refrun, fp);

	// Scaling test
	for(int iswp = 0; iswp < static_cast<int>(nbswplist.size()); iswp++)
	{
		setSweeps(nbswplist[iswp], naswplist[iswp]);

		for(int thread : threadlist)
		{
			printf(" Sweeps = %d,%d; num threads = %d.\n", nbswplist[iswp], naswplist[iswp], thread);

			omp_set_num_threads(thread);
			const ThreadCaseInfo tci = runcase(m, A, b, nrepeats);
			writeReportLine(tci, refrun, fp);
			printf("\n");
		}
	}

	fclose(fp);
}

ThreadCaseInfo runcase(const CartMesh& m, const SMat& A, const SVec& b, const int nruns)
{
	std::vector<double> wtime(nruns);
	std::vector<double> precbuildtime(nruns), precapplytime(nruns);
	std::vector<int> iters(nruns);
	std::vector<sreal> resnorms(nruns);
	for(int irun = 0; irun < nruns; irun++)
	{
		printf("    Test run %d:\n", irun);

		SolverBase *const solver = createSolver(A);

		const sreal buildstarttime = MPI_Wtime();
		solver->updateOperator();
		const sreal buildtimeinterval = MPI_Wtime() - buildstarttime;
		precbuildtime[irun] = buildtimeinterval;
		wtime[irun] = buildtimeinterval;

		SVec u(&m);

		const sreal starttime = MPI_Wtime();
		const SolveInfo slvinfo = solver->apply(b, u);
		const sreal endtime = MPI_Wtime() - starttime;
		wtime[irun] += endtime;
		precapplytime[irun] = slvinfo.precapplywtime;

		delete solver;

		if(!slvinfo.converged)
			throw std::runtime_error("Solver did not converge!");

		iters[irun] = slvinfo.iters;
		resnorms[irun] = slvinfo.resnorm;
	}

	ThreadCaseInfo tci;

	tci.avg_niter = average(iters);
	tci.avg_precbuildtime = average(precbuildtime);
	tci.avg_precapplytime = average(precapplytime);
	tci.avg_solvetime = average(wtime);
	tci.avg_residual = average(resnorms);
	tci.dev_niter = relative_deviation(iters, tci.avg_niter);
	tci.dev_precbuildtime = relative_deviation(precbuildtime, tci.avg_precbuildtime);
	tci.dev_precapplytime = relative_deviation(precapplytime, tci.avg_precapplytime);
	tci.nthreads =
#ifdef _OPENMP
		omp_get_max_threads()
#else
		1
#endif
		;
	tci.nbuildsweeps = petscoptions_get_int("-s3d_pc_build_sweeps");
	tci.napplysweeps = petscoptions_get_int("-s3d_pc_apply_sweeps");

	return tci;
}

template <typename T>
T average(const std::vector<T>& arr)
{
	double avg = 0;
	for(size_t i = 0; i < arr.size(); i++)
		avg += arr[i];
	avg = avg/arr.size();
	return static_cast<T>(avg);
}

/// Compute the standard deviation of an array relative to its mean
template <typename T>
double relative_deviation(const std::vector<T> arr, const T avg)
{
	double dev = 0;
	for(size_t i = 0; i < arr.size(); i++) {
		dev += (arr[i]-avg)*(arr[i]-avg);
	}
	dev = sqrt(dev/arr.size());
	double reldev = dev/avg;
	return reldev;
}

void setSweeps(const int nbswp, const int naswp)
{
	// add option
	std::string bvalue = std::to_string(nbswp);
	std::string avalue = std::to_string(naswp);
	int ierr = PetscOptionsSetValue(NULL, "-s3d_pc_build_sweeps", bvalue.c_str());
	if(ierr) throw std::runtime_error("Couldn't set PETSc option for build sweeps");
	ierr = PetscOptionsSetValue(NULL, "-s3d_pc_apply_sweeps", avalue.c_str());
	if(ierr) throw std::runtime_error("Couldn't set PETSc option for apply sweeps");

	// Check
	const int chkbswp = petscoptions_get_int("-s3d_pc_build_sweeps");
	const int chkaswp = petscoptions_get_int("-s3d_pc_apply_sweeps");

	if(chkbswp != nbswp)
		throw std::runtime_error("Async build sweeps not set properly!");
	if(chkaswp != naswp)
		throw std::runtime_error("Async apply sweeps not set properly!");
}

void writeReportHeader(const ThreadCaseInfo refrun, FILE *const fp)
{
	fprintf(fp, "Reference run: Threads %d, build sweeps %d, apply sweeps %d\n",
	        refrun.nthreads, refrun.nbuildsweeps, refrun.napplysweeps);
	fprintf(fp, "              iterations %d, iterations dev. %f, residual norm %g, res norm dev. %f\n",
	        refrun.avg_niter, refrun.dev_niter, refrun.avg_residual, refrun.dev_residual);
	fprintf(fp, "              build time %f, build time dev. %f, apply time %f, apply time dev. %f\n",
	        refrun.avg_precbuildtime, refrun.dev_precbuildtime,
	        refrun.avg_precapplytime, refrun.dev_precapplytime);
	fprintf(fp, "\n");
	fprintf(fp, "%8s %8s %8s %6s %10s %8s %8s %8s %12s %12s %12s\n",
	        "b.swps.", "a.swps.", "threads", "iters", "dev.iters", "b.spdp", "a.spdp", "spdp",
	        "dev.b.time", "dev.a.time", "dev.resnorm");
	fprintf(fp, "---------------------------------------------------------------------------------");
	fprintf(fp, "-----------------------------\n");
	writeReportLine(refrun, refrun, fp);
}

void writeReportLine(const ThreadCaseInfo run, const ThreadCaseInfo refrun, FILE *const fp)
{
	const double totalspdp = (refrun.avg_precbuildtime + refrun.avg_precapplytime)
		/ (run.avg_precbuildtime + run.avg_precapplytime);
	fprintf(fp, "%8d %8d %8d %6d %10f %8f %8f %8f %12f %12f %12f\n", run.nbuildsweeps, run.napplysweeps,
	        run.nthreads, run.avg_niter, run.dev_niter, refrun.avg_precbuildtime/run.avg_precbuildtime,
	        refrun.avg_precapplytime/run.avg_precapplytime, totalspdp,
	        run.dev_precbuildtime, run.dev_precapplytime, run.dev_residual);
}
