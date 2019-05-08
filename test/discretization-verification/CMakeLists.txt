
add_executable(gridconv-petsc gridconv-petsc.cpp)
target_link_libraries(gridconv-petsc pdes)

add_executable(gridconv gridconv.cpp)
target_link_libraries(gridconv linalg pdes)

add_test(NAME Poisson_NativeGridConv_UniformGrid
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/poisson-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  )

add_test(NAME CD_NativeGridConv_UniformGrid
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc -ksp_max_it 550
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

# The following are kinda supposed to be solver tests

add_test(NAME StructSolver_GridConv_CDCirc_UniformGrid_ILU0
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cdcirc-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  -ksp_max_it 500 -s3d_ksp_type gcr -s3d_pc_apply_sweeps 2
  )

add_test(NAME StructSolver_GridConv_Poisson_UniformGrid_SGS-ISAI
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/poisson-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  -ksp_max_it 1000 -s3d_ksp_type richardson -s3d_pc_type sgs_isai -s3d_pc_apply_sweeps 1
  )

add_test(NAME StructSolver_GridConv_CD_UniformGrid_SGS-ISAI
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  -ksp_max_it 200 -s3d_ksp_type gcr -s3d_pc_type sgs_isai -s3d_pc_apply_sweeps 1
  )

add_test(NAME StructSolver_GridConv_CD_UniformGrid_ILU0-ISAI
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/native.perc
  -ksp_max_it 1000 -s3d_ksp_type richardson -s3d_pc_type ilu_isai
  -s3d_pc_build_sweeps 1 -s3d_pc_apply_sweeps 1
  )
