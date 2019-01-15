
add_library(base case.cpp pde/pdebase.cpp linalg/matvec.cpp cartmesh.cpp common_utils.cpp)
target_link_libraries(base ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_library(linalg linalg/solverfactory.cpp
  linalg/s3d_ilu.cpp linalg/s3d_jacobi.cpp linalg/s3d_sgspreconditioners.cpp
  linalg/s3d_solverbase.cpp)
target_link_libraries(linalg base)

add_library(cases case.cpp)

add_library(poisson pde/poisson.cpp)
target_link_libraries(poisson base)

add_library(convdiff pde/convdiff.cpp)
target_link_libraries(convdiff base)

add_executable(runcase runcase.cpp)
target_link_libraries(runcase linalg poisson convdiff base)
