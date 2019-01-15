
add_executable(comparewithpetsc compare.cpp)
target_link_libraries(comparewithpetsc linalg base poisson convdiff)

add_test(NAME StructSolver_VsPetsc_Poisson_Jacobi
  COMMAND comparewithpetsc ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type jacobi -s3d_pc_type jacobi
  )

add_test(NAME StructSolver_VsPetsc_CD_Jacobi
  COMMAND comparewithpetsc ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type jacobi -s3d_pc_type jacobi
  )

add_test(NAME StructSolver_VsPetsc_CD_SGS
  COMMAND comparewithpetsc ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type sor -s3d_pc_type sgs -s3d_pc_use_threaded_apply false -s3d_pc_apply_sweeps 1
  )

add_test(NAME StructSolver_VsPetsc_CD_StrILU
  COMMAND comparewithpetsc ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff-large.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type ilu -s3d_pc_type strilu
  -s3d_pc_use_threaded_build false -s3d_pc_use_threaded_apply false
  -s3d_pc_build_sweeps 1 -s3d_pc_apply_sweeps 1
  )

add_test(NAME StructSolver_VsPetsc_CD_StrILU_Threaded
  COMMAND env OMP_NUM_THREADS=4 ${CMAKE_CURRENT_BINARY_DIR}/comparewithpetsc 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff-verylarge.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/struct-petsc.perc
  -pc_type ilu -s3d_pc_type strilu
  -s3d_pc_use_threaded_build true -s3d_pc_use_threaded_apply true
  -s3d_pc_build_sweeps 1 -s3d_pc_apply_sweeps 1
  )
