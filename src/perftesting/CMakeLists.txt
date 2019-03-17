
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_library(openmptest scaling_openmp.cpp)
target_link_libraries(openmptest linalg pdes)

add_executable(runcase runcase.cpp)
target_link_libraries(runcase openmptest)

add_executable(runtest_openmp runtest_openmp.cpp)
target_link_libraries(runtest_openmp openmptest)
