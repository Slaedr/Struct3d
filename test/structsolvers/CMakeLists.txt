
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
