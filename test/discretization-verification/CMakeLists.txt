
add_executable(gridconv gridconv.cpp)
target_link_libraries(gridconv poisson convdiff base)

add_test(NAME Poisson_GridConv_UniformGrid
  COMMAND ${MPIEXEC} -n 4 gridconv ${CMAKE_CURRENT_SOURCE_DIR}/poisson-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/spd.perc
  )

add_test(NAME CD_GridConv_UniformGrid
  COMMAND ${MPIEXEC} -n 2 gridconv ${CMAKE_CURRENT_SOURCE_DIR}/cd-conv.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/gen.perc
  )
