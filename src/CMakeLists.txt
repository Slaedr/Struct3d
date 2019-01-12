
add_library(base case.cpp pde/pdebase.cpp linalg/matvec.cpp cartmesh.cpp common_utils.cpp)
target_link_libraries(base ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_library(cases case.cpp)

add_library(poisson pde/poisson.cpp)
target_link_libraries(poisson base)

add_library(convdiff pde/convdiff.cpp)
target_link_libraries(convdiff base)

