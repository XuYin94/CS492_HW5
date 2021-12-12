cmake_minimum_required (VERSION 2.8)

get_filename_component(target_name ${CMAKE_CURRENT_LIST_DIR} NAME)
project (${target_name})


# Read configuration file.
include ("${CMAKE_CURRENT_SOURCE_DIR}/../config.cmake.in/")

# Set output directory.
set (OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bin/")

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
          "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
      endif (NOT CMAKE_BUILD_TYPE)
set (CMAKE_DEBUG_POSTFIX "d")

if (WIN32)
  add_definitions(
      -D_USE_MATH_DEFINES -DNOMINMAX
      -D_CRT_SECURE_NO_WARNINGS
  )
endif ()

# Find needed packages for gui applications.
find_package (OpenGL)
find_package (GLUT)

# Set target name.
set (target_name ${target_name})

# Add header and source files.
include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GLUT_INCLUDE_DIR}
  ${OPENMESH_SOURCE_DIR}
  ${EIGEN_SOURCE_DIR}
)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

if (MSVC)
	source_group("Header Files" FILES ${headers})
	source_group("Source Files" FILES ${sources})
endif()

# Add library directories.
link_directories (
  ${OPENMESH_LIBRARY_DIR}
)

# Create executable.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
add_executable(${target_name} ${headers} ${sources})

# Link libraries.
target_link_libraries (${target_name}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  debug OpenMeshCore${CMAKE_DEBUG_POSTFIX}
  debug OpenMeshTools${CMAKE_DEBUG_POSTFIX}
  optimized OpenMeshCore
  optimized OpenMeshTools
)

