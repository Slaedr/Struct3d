#include <cmath>
#include "common_utils.hpp"

PetscReal computeNorm(const CartMesh *const m, Vec v, DM da)
{
	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	
	// get local data that can be accessed by global indices
	PetscReal *** vv;
	DMDAVecGetArray(da, v, &vv);

	PetscReal norm = 0, global_norm = 0;

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				// incremented indices for mesh coords access
				const sint I = i+1, J = j+1, K = k+1;
				const sreal vol = 1.0/8.0*(m->gcoords(0,I+1)-m->gcoords(0,I-1))
					*(m->gcoords(1,J+1)-m->gcoords(1,J-1))*(m->gcoords(2,K+1)-m->gcoords(2,K-1));
				norm += vv[k][j][i]*vv[k][j][i]*vol;
			}

	DMDAVecRestoreArray(da, v, &vv);

	// get global norm
	MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

	return sqrt(global_norm);
}

PetscReal compute_error(const CartMesh& m, const DM da, const Vec u, const Vec uexact)
{
	PetscReal errnorm;
	Vec err;
	VecDuplicate(u, &err);
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	errnorm = computeNorm(&m, err, da);
	// VecNorm(err, NORM_2, &errnorm);
	// errnorm = errnorm / sqrt(m.gnpointotal());
	VecDestroy(&err);
	return errnorm;
}

int get_mpi_rank(MPI_Comm comm) {
	int rank;
	MPI_Comm_rank(comm, &rank);
	return rank;
}

int get_mpi_size(MPI_Comm comm) {
	int size;
	MPI_Comm_size(comm, &size);
	return size;
}
