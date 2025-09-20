cmake_minimum_required(VERSION 2.8)

project( DisplayImage )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV 4.1.2 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( DisplayImage src/corner_refinement.cpp )
add_executable( main src/main.cpp)
add_executable( temp src/temp.cpp)
add_executable( undistortion src/undistortion.cpp)
add_executable( histogram src/histogram.cpp)

target_link_libraries( DisplayImage ${OpenCV_LIBS} )
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( temp ${OpenCV_LIBS} )
target_link_libraries( undistortion ${OpenCV_LIBS} )
target_link_libraries( histogram ${OpenCV_LIBS} )
