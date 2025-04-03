cmake_minimum_required(VERSION 3.8)
project(smart_walker_ai)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Seccion de importe de librerias
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# Instalacion de source para ejecucion c++ y python
ament_python_install_package(${PROJECT_NAME})

include_directories(include)

install(
  DIRECTORY assets agent_config config retriever_data
  DESTINATION share/${PROJECT_NAME}
)

# Instalacion de ejecutables de python
install(PROGRAMS
  ${PROJECT_NAME}/navigation_ai_node.py
  ${PROJECT_NAME}/path_generator_ia.py  
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
