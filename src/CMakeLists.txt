set(OpenCV_DIR "/home/kreimer/opencv-mod/share/OpenCV/")
set(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories("/home/kreimer/tmp/eigen-eigen-6b38706d90a9/")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS log log_setup system filesystem thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_library(viso viso.cpp mvg.cpp estimation.cpp)

add_executable(kitti kitti.cpp)
target_link_libraries(kitti viso ${OpenCV_LIBS} ${Boost_LIBRARIES}) 

add_executable(registration registration.cpp)
target_link_libraries(registration viso ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(mod3d mod3d.cpp)
target_link_libraries(mod3d viso ${OpenCV_LIBS} ${Boost_LIBRARIES})
