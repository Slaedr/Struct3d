
add_library(base case.cpp linalg/matvec.cpp cartmesh.cpp common_utils.cpp)
target_link_libraries(base ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_library(linalg linalg/solverfactory.cpp
  linalg/s3d_ilu.cpp linalg/s3d_jacobi.cpp linalg/s3d_sgspreconditioners.cpp linalg/s3d_isai.cpp
  linalg/s3d_gcr.cpp linalg/s3d_mr.cpp linalg/s3d_solverbase.cpp)
target_link_libraries(linalg base)

add_library(cases case.cpp)

add_library(pdes pde/pdefactory.cpp pde/convdiff.cpp pde/convdiff_circular.cpp pde/poisson.cpp
  pde/pdebase.cpp)
target_link_libraries(pdes base)

add_subdirectory(perftesting)

