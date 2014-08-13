set(OpenCV_DIR "/home/kreimer/opencv3.0/share/OpenCV/")
set(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories("/home/kreimer/tmp/eigen-eigen-6b38706d90a9/")

ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS log log_setup system filesystem thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_library(viso viso.cpp mvg.cpp estimation.cpp misc.cpp)

add_executable(kitti kitti.cpp)
target_link_libraries(kitti viso ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(calib_sfm calib_sfm.cpp)
target_link_libraries(calib_sfm viso ${OpenCV_LIBS} ${Boost_LIBRARIES})