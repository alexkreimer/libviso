set(OpenCV_DIR "/home/kreimer/opencv2.4.9/share/OpenCV/")
set(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories("/home/kreimer/tmp/eigen-eigen-10219c95fe65/")

add_definitions (-DBOOST_ALL_DYN_LINK)
set(Boost_USE_STATIC_LIBS        OFF)  # I've already tried ON
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS log log_setup system filesystem thread unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories (${TEST_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

add_executable (tester test.cpp)
target_link_libraries (tester viso ${OpenCV_LIBS}  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_LIBRARIES})
