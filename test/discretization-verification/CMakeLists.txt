
add_executable(gridconv-petsc gridconv-petsc.cpp)
target_link_libraries(gridconv-petsc poisson convdiff base)

add_executable(gridconv gridconv.cpp)
target_link_libraries(gridconv linalg poisson convdiff base)

add_test(NAME Poisson_NativeGridConv_UniformGrid
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/poisson-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  )

add_test(NAME CD_NativeGridConv_UniformGrid
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  )

add_test(NAME CDCirc_NativeGridConv_UniformGrid
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cdcirc-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  )

add_test(NAME Poisson_PetscGridConv_UniformGrid
  COMMAND ${MPIEXEC} -n 1 gridconv-petsc ${CMAKE_CURRENT_SOURCE_DIR}/poisson-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/petscsolver.perc
  )

add_test(NAME Poisson_PetscGridConv_ChebyshevGrid
  COMMAND ${MPIEXEC} -n 4 gridconv-petsc ${CMAKE_CURRENT_SOURCE_DIR}/poisson-cheb-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/gen.perc
  )

add_test(NAME CD_PetscGridConv_UniformGrid
  COMMAND ${MPIEXEC} -n 2 gridconv-petsc ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/gen.perc
  )
