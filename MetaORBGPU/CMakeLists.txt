cmake_minimum_required(VERSION 3.0)
project(MetaORBGPU)
find_package(OpenCV REQUIRED)
add_executable(MetaORBGPU MetaORB.cpp)
target_link_libraries(MetaORBGPU ${OpenCV_LIBS})