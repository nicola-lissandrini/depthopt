cmake_minimum_required(VERSION 3.0.2)
project(depthopt)

add_compile_options(-std=c++14)
add_compile_options(-ffast-math)
add_compile_options (-Wno-pmf-conversions)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	nlib
	message_generation std_msgs geometry_msgs nav_msgs)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (Torch REQUIRED)
find_package (lietorch REQUIRED)

catkin_package (
	INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS}
	CATKIN_DEPENDS nlib
)

include_directories(
	include
	include/depthopt
	${catkin_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}
	${lietorch_INCLUDE_DIRS}
)

add_executable(depthopt
	src/depthopt_node.cpp
	src/ros_conversions.cpp
	src/depthopt_modflow.cpp
	src/synchronization.cpp
	src/landscape.cpp
	src/optimizer.cpp)

target_link_libraries (depthopt dl ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${lietorch_LIBRARIES})
