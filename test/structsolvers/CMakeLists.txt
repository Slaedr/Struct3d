
add_executable(comparewithpetsc compare.cpp)
target_link_libraries(comparewithpetsc linalg pdes)

add_test(NAME StructSolver_VsPetsc_Poisson_Jacobi
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type jacobi -s3d_pc_type jacobi
  )

add_test(NAME StructSolver_VsPetsc_CD_Jacobi
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc -ksp_view
  -pc_type jacobi -s3d_pc_type jacobi
  )

add_test(NAME StructSolver_VsPetsc_CD_GCR_Jacobi
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -ksp_type gcr -ksp_view -s3d_ksp_type gcr
  -pc_type jacobi -s3d_pc_type jacobi
  )

add_test(NAME StructSolver_VsPetsc_CD_SGS
  COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type sor -s3d_pc_type sgs -s3d_pc_use_threaded_apply false -s3d_pc_apply_sweeps 4
  )

add_test(NAME StructSolver_VsPetsc_CD_StrILU
  COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type ilu -s3d_pc_type strilu
  -s3d_pc_use_threaded_build false -s3d_pc_use_threaded_apply false
  -s3d_pc_build_sweeps 1 -s3d_pc_apply_sweeps 5
  )

add_test(NAME StructSolver_VsPetsc_CD_Large_StrILU_1thread
  COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ./comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff-large.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type ilu -s3d_pc_type strilu
  -s3d_pc_use_threaded_build true -s3d_pc_use_threaded_apply true
  -s3d_pc_build_sweeps 1 -s3d_pc_apply_sweeps 1 -relative_iters_deviation 0.02
  )
