set(OpenCV_DIR "/home/kreimer/opencv-mod/share/OpenCV/")
set(CMAKE_CXX_FLAGS "-std=c++0x")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
include_directories("/home/kreimer/tmp/eigen-eigen-6b38706d90a9/")

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

include_directories (${TEST_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
add_definitions (-DBOOST_TEST_DYN_LINK)
add_executable (Test test.cpp)
target_link_libraries (Test viso  ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
