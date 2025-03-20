cmake_minimum_required(VERSION 3.8)
project(yolo_pose_buff)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 设置默认的 ONNX Runtime 根目录，如果用户没有手动提供
if(NOT ONNXRUNTIME_ROOTDIR)
    if(WIN32)
        # Windows 系统下设置默认的 ONNX Runtime 路径
        set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
    else()
        # 非 Windows 系统下设置默认路径
        set(ONNXRUNTIME_ROOTDIR "/home/qianli/onnxruntime-linux-x64-gpu-1.20.0")
    endif()
endif()

# 显示当前使用的 ONNXRUNTIME_ROOTDIR 路径
message(STATUS "Using ONNX Runtime directory: ${ONNXRUNTIME_ROOTDIR}")

# 设置编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 添加包含路径
include_directories(
  ${ONNXRUNTIME_ROOTDIR}/include
  ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
  ${CMAKE_SOURCE_DIR}/include/yolo_pose_buff
)

# 查找依赖项
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(yolo_pose_buff
  src/inference.cpp
  src/main.cpp
)

# 设置依赖关系
ament_target_dependencies(yolo_pose_buff rclcpp)

# 链接库文件
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

# 链接 OpenCV 和 ONNX Runtime 库
target_link_libraries(yolo_pose_buff ${OpenCV_LIBRARIES} onnxruntime)

# 安装目标文件
install(TARGETS
  yolo_pose_buff
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 启用测试并进行自动化检查
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
