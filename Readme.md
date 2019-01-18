Struct3d
========

Parallel Cartesian-grid 3D finite-difference solver for simple PDEs. Currently has Poisson and (steady) convection-diffusion, and assumes homogeneous Dirichlet boundaries.

The code is MPI-parallel, using PETSc as the parallel computation framework. PETSc's DMDA is used to handle the vectors and matrix. There are also "native" linear algebra and solvers optimized for structured grids. These are not MPI-parallel.
