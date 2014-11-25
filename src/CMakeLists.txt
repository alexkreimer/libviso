set(OpenCV_DIR "/home/kreimer/opencv2.4.9/share/OpenCV/")
set(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories("/home/kreimer/tmp/eigen-eigen-1306d75b4a21/")

ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS        OFF)  # I've already tried ON
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS log log_setup system filesystem thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_library(viso viso.cpp mvg.cpp estimation.cpp misc.cpp)

add_executable(kitti kitti.cpp)
target_link_libraries(kitti viso ${OpenCV_LIBS} ${Boost_LIBRARIES})