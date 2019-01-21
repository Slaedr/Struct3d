# test executables

add_executable(testnativeassembly nativevectorassemblytest.cpp)
target_link_libraries(testnativeassembly base poisson)

add_executable(testmatvec matvectest.cpp)
target_link_libraries(testmatvec base poisson convdiff)

add_test(NAME Assembly
  COMMAND ${MPIEXEC} -n 1 ./testnativeassembly
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poissonsmall.control
  )

add_test(NAME MatVec
  COMMAND ${MPIEXEC} -n 1 ./testmatvec
  ${CMAKE_CURRENT_SOURCE_DIR}/discretization-verification/poisson-conv.control
  )

add_subdirectory(discretization-verification)

add_subdirectory(structsolvers)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input/run-defaults.perc ${CMAKE_BINARY_DIR}/run-defaults.perc)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff-testcase.control 
	${CMAKE_BINARY_DIR}/convdiff-testcase.control)

add_test(NAME NativeRunCase
  COMMAND ${SEQEXEC} ${SEQTASKS} ${THREADOTPS} ${CMAKE_BINARY_DIR}/src/runcase 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/convdiff.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/run-defaults.perc
  )

if(WITH_BLASTED)
  add_executable(testblasted_csr compare_blasted.cpp)
  target_link_libraries(testblasted_csr ${BLASTED_LIB} base poisson)

  add_executable(testblasted_csr_threaded compare-threaded_blasted.cpp)
  target_link_libraries(testblasted_csr_threaded ${BLASTED_LIB} base poisson)

  add_test(NAME BlastedMPIPoissonCSRPetscILU0
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testblasted_csr
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc)

  add_test(NAME BlastedThreadedPoissonCSRPetscSAPILU0
	COMMAND ${MPIEXEC} -n 1 ./testblasted_csr_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
	-ref_sub_pc_type ilu -blasted_pc_type sapilu0 -test_type compare_error)

  add_test(NAME BlastedAsyncRelaxationPoissonPetsc-SGS-IterationsLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testblasted_csr_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type sgs -test_type compare_its)
endif(WITH_BLASTED)
