# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build

# Include any dependencies generated for this target.
include CMakeFiles/test_lietorch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_lietorch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_lietorch.dir/flags.make

CMakeFiles/test_lietorch.dir/test_queue.cpp.o: CMakeFiles/test_lietorch.dir/flags.make
CMakeFiles/test_lietorch.dir/test_queue.cpp.o: ../test_queue.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_lietorch.dir/test_queue.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_lietorch.dir/test_queue.cpp.o -c /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/test_queue.cpp

CMakeFiles/test_lietorch.dir/test_queue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_lietorch.dir/test_queue.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/test_queue.cpp > CMakeFiles/test_lietorch.dir/test_queue.cpp.i

CMakeFiles/test_lietorch.dir/test_queue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_lietorch.dir/test_queue.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/test_queue.cpp -o CMakeFiles/test_lietorch.dir/test_queue.cpp.s

# Object files for target test_lietorch
test_lietorch_OBJECTS = \
"CMakeFiles/test_lietorch.dir/test_queue.cpp.o"

# External object files for target test_lietorch
test_lietorch_EXTERNAL_OBJECTS =

test_lietorch: CMakeFiles/test_lietorch.dir/test_queue.cpp.o
test_lietorch: CMakeFiles/test_lietorch.dir/build.make
test_lietorch: /opt/libtorch/lib/libtorch.so
test_lietorch: /opt/libtorch/lib/libc10.so
test_lietorch: /opt/libtorch/lib/libkineto.a
test_lietorch: /usr/local/cuda-11.5/lib64/stubs/libcuda.so
test_lietorch: /usr/local/cuda-11.5/lib64/libnvrtc.so
test_lietorch: /usr/local/cuda-11.5/lib64/libnvToolsExt.so
test_lietorch: /usr/local/cuda-11.5/lib64/libcudart.so
test_lietorch: /opt/libtorch/lib/libc10_cuda.so
test_lietorch: /opt/libtorch/lib/libc10_cuda.so
test_lietorch: /opt/libtorch/lib/libc10.so
test_lietorch: /usr/local/cuda-11.5/lib64/libcufft.so
test_lietorch: /usr/local/cuda-11.5/lib64/libcurand.so
test_lietorch: /usr/local/cuda-11.5/lib64/libcublas.so
test_lietorch: /usr/lib/x86_64-linux-gnu/libcudnn.so
test_lietorch: /usr/local/cuda-11.5/lib64/libnvToolsExt.so
test_lietorch: /usr/local/cuda-11.5/lib64/libcudart.so
test_lietorch: CMakeFiles/test_lietorch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_lietorch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_lietorch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_lietorch.dir/build: test_lietorch

.PHONY : CMakeFiles/test_lietorch.dir/build

CMakeFiles/test_lietorch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_lietorch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_lietorch.dir/clean

CMakeFiles/test_lietorch.dir/depend:
	cd /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build /home/nicola/Dropbox/ros/tum_ws/src/slife/src/ctests/build/CMakeFiles/test_lietorch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_lietorch.dir/depend

