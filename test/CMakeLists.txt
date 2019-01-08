# test executables
add_executable(testfd_csr compare.cpp)
target_link_libraries(testfd_csr ${BLASTED_LIB} base poisson)

add_executable(testfd_csr_threaded compare-threaded.cpp)
target_link_libraries(testfd_csr_threaded ${BLASTED_LIB} base poisson)

# Tests
#add_test(NAME SerialPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testfd_csr 
#${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
#-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_ilu0.perc)

#add_test(NAME SerialPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testfd_csr 
#${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
#-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_sgs.perc)

#add_test(NAME SerialPoissonCSRPetscAMGILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testfd_csr 
#  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-amg.control 
#  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_amg.perc)

add_test(NAME MPIPoissonCSRPetscJacobiRelaxation
  COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testfd_csr
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
  -ref_sub_pc_type jacobi -blasted_pc_type jacobi
  )

add_test(NAME MPIPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testfd_csr 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc)

add_test(NAME ThreadedPoissonCSRPetscILU0 COMMAND ${MPIEXEC} -n 1 ./testfd_csr_threaded 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc
  -ref_sub_pc_type ilu -blasted_pc_type ilu0 -test_type compare_error -error_tolerance_factor 1e7)

add_test(NAME ThreadedPoissonCSRPetscSAPILU0 COMMAND ${MPIEXEC} -n 1 ./testfd_csr_threaded 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
  -ref_sub_pc_type ilu -blasted_pc_type sapilu0 -test_type compare_error)

add_test(NAME AsyncRelaxationPoissonPetsc-SGS-IterationsLowerBound
  COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testfd_csr_threaded
  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc 
  -ref_sub_pc_type jacobi -blasted_pc_type sgs -test_type compare_its)

