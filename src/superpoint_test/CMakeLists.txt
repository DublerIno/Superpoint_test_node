cmake_minimum_required(VERSION 3.8)
project(superpoint_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

set(DEPENDENCIES
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  mrs_lib
)

foreach(DEP IN LISTS DEPENDENCIES)
  find_package(${DEP} REQUIRED)
endforeach()

# --- Torch ---
set(CMAKE_PREFIX_PATH "/opt/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${mrs_lib_INCLUDE_DIRS}
)


# Build as component library 
# not add_executable!
add_library(${PROJECT_NAME}_node SHARED
  src/superpoint_detector_node.cpp
  src/SuperPoint.cpp
)

target_compile_options(${PROJECT_NAME}_node PRIVATE ${TORCH_CXX_FLAGS})

target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

ament_target_dependencies(${PROJECT_NAME}_node
  ${DEPENDENCIES}
)

# Register component + generate runnable executable
rclcpp_components_register_nodes(${PROJECT_NAME}_node
  PLUGIN "superpoint_test::SuperPointDetectorNode"
  EXECUTABLE ${PROJECT_NAME}
)

# Torch runtime lookup (optional but handy)
set_target_properties(${PROJECT_NAME}_node PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Install like MRS packages do
install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_libraries(${PROJECT_NAME}_node)
ament_export_dependencies(${DEPENDENCIES})
ament_package()
