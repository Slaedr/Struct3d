/** \file cartmesh.cpp
 * \brief Implementation of single-block but distributed, non-uniform Cartesian grid using PETSc
 * \author Aditya Kashi
 */

#include <limits>
#include "cartmesh.hpp"
#include "common_utils.hpp"

void CartMesh::computeMeshSize()
{
	// estimate h
	h = 0.0;
	PetscReal hd[NDIM];
	for(int k = 0; k < npoind[2]-1; k++)
	{
		hd[2] = coords[2][k+1]-coords[2][k];
		for(int j = 0; j < npoind[1]-1; j++)
		{
			hd[1] = coords[1][j+1]-coords[1][j];
			for(int i = 0; i < npoind[0]-1; i++)
			{
				hd[0] = coords[0][i+1]-coords[0][i];
				PetscReal diam = 0;
				for(int idim = 0; idim < NDIM; idim++)
					diam += hd[idim]*hd[idim];
				diam = std::sqrt(diam);
				if(diam > h)
					h = diam;
			}
		}
	}
}

CartMesh::CartMesh()
	: coords{NULL}, nghost{1}
{ }

PetscErrorCode CartMesh::createMeshAndDMDA(const MPI_Comm comm, const PetscInt npdim[NDIM], 
                                           PetscInt ndofpernode, PetscInt stencil_width,
                                           DMBoundaryType bx, DMBoundaryType by, DMBoundaryType bz,
                                           DMDAStencilType stencil_type,
                                           DM *const dap)
{
	PetscErrorCode ierr = 0;
	const int rank = get_mpi_rank(comm);

	for(int i = 0; i < NDIM; i++) {
		npoind[i] = npdim[i];
	}

	if(rank == 0) {
		std::printf("CartMesh: Number of points in each direction: ");
		for(int i = 0; i < NDIM; i++) {
			std::printf("%d ", npoind[i]);
		}
		std::printf("\n");
	}

	npointotal = 1;
	for(int i = 0; i < NDIM; i++)
		npointotal *= npoind[i];

	PetscInt ngpoints = npoind[0]*npoind[1]*2 + (npoind[2]-2)*npoind[0]*2 + 
		(npoind[1]-2)*(npoind[2]-2)*2;
	nDomPoin = npointotal-ngpoints;

	if(rank == 0)
		std::printf("CartMesh: Setting up DMDA\n");
	
	ierr = DMDACreate3d(comm, bx, by, bz, stencil_type, 
			npoind[0]-2, npoind[1]-2, npoind[2]-2, 
			PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, ndofpernode, stencil_width, 
			NULL, NULL, NULL, dap);
	CHKERRQ(ierr);
	ierr = DMSetUp(*dap); CHKERRQ(ierr);

	PetscInt M,N,P;
	ierr = DMDAGetInfo(*dap, NULL, &M, &N, &P, &nprocs[0], &nprocs[1], &nprocs[2], 
			NULL, NULL, NULL, NULL, NULL, NULL);
	CHKERRQ(ierr);

	ntprocs = nprocs[0]*nprocs[1]*nprocs[2];

	// have each process store coords; hardly costs anything
	coords = (PetscReal**)std::malloc(NDIM*sizeof(PetscReal*));
	for(int i = 0; i < NDIM; i++)
		coords[i] = (PetscReal*)std::malloc(npoind[i]*sizeof(PetscReal));

	if(rank == 0) {
		std::printf("CartMesh: Number of points in each direction: %d,%d,%d.\n", 
		            M,N,P);
		std::printf("CartMesh: Number of procs in each direction: %d,%d,%d.\n", 
		            nprocs[0], nprocs[1], nprocs[2]);
		std::printf("CartMesh: Total points = %d, domain points = %d, partitions = %d\n", 
		            npointotal, nDomPoin, ntprocs);
	}

	return ierr;
}

PetscErrorCode CartMesh::createMesh(const PetscInt npdim[NDIM])
{
	int ierr = 0;
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int mpisize = get_mpi_size(MPI_COMM_WORLD);
	if(mpisize > 1) {
		throw std::runtime_error("Our version does not work with Petsc yet!");
	}

	for(int i = 0; i < NDIM; i++) {
		npoind[i] = npdim[i];
	}

	if(rank == 0) {
		std::printf("CartMesh: Number of points in each direction: ");
		for(int i = 0; i < NDIM; i++) {
			std::printf("%d ", npoind[i]);
		}
		std::printf("\n");
	}

	npointotal = 1;
	for(int i = 0; i < NDIM; i++)
		npointotal *= npoind[i];

	sint ngpoints = npoind[0]*npoind[1]*2 + (npoind[2]-2)*npoind[0]*2 + 
		(npoind[1]-2)*(npoind[2]-2)*2;
	nDomPoin = npointotal-ngpoints;
	
	nprocs[0] = nprocs[1] = nprocs[2] = 1;

	ntprocs = nprocs[0]*nprocs[1]*nprocs[2];

	// have each process store coords; hardly costs anything
	coords = (sreal**)std::malloc(NDIM*sizeof(sreal*));
	for(int i = 0; i < NDIM; i++)
		coords[i] = (sreal*)std::malloc(npoind[i]*sizeof(sreal));

	if(rank == 0) {
		std::printf("CartMesh: Number of procs in each direction: %d,%d,%d.\n", 
		            nprocs[0], nprocs[1], nprocs[2]);
		std::printf("CartMesh: Total points = %d, domain points = %d, partitions = %d\n", 
		            npointotal, nDomPoin, ntprocs);
	}

	return ierr;
}

CartMesh::~CartMesh()
{
	for(int i = 0; i < NDIM; i++)
		std::free(coords[i]);
	std::free(coords);
}

void CartMesh::generateMesh_ChebyshevDistribution(const sreal rmin[NDIM], 
                                                  const sreal rmax[NDIM])
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		std::printf("CartMesh: generateMesh_cheb: Generating grid\n");
	for(int idim = 0; idim < NDIM; idim++)
	{
		PetscReal theta = PI/(npoind[idim]-1);
		for(int i = 0; i < npoind[idim]; i++) {
			coords[idim][i] = (rmax[idim]+rmin[idim])*0.5 + 
				(rmax[idim]-rmin[idim])*0.5*std::cos(PI-i*theta);
		}
	}

	// estimate h
	computeMeshSize();
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Cheb: h = %f\n", h);
}

/// Generates grid with uniform spacing
void CartMesh::generateMesh_UniformDistribution(const sreal rmin[NDIM], 
                                                const sreal rmax[NDIM])
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Uniform: Generating grid\n");
	for(int idim = 0; idim < NDIM; idim++)
	{
		for(int i = 0; i < npoind[idim]; i++) {
			coords[idim][i] = rmin[idim] + (rmax[idim]-rmin[idim])*i/(npoind[idim]-1);
		}
	}

	computeMeshSize();
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Uniform: h = %f\n", h);

	sreal dr[NDIM];
	for(int i = 0; i < NDIM; i++)
		dr[i] = gcoords(i,1)-gcoords(i,0);
	for(int idim = 0; idim < NDIM; idim++)
	{
		for(int j = 0; j < gnpoind(idim)-1; j++) {
			if(std::abs(gcoords(idim,j+1)-gcoords(idim,j)-dr[idim])
			   > std::numeric_limits<sreal>::epsilon())
				throw std::runtime_error("Uniform mesh is not actually uniform!");
		}
	}
}

