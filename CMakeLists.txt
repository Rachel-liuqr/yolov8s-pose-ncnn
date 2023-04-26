cmake_minimum_required(VERSION 3.11)
project(yolov8s-pose-ncnn)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4 REQUIRED)
#find_package(ncnn REQUIRED)

set(CMAKE_BUILD_TYPE "Debug")

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
# 分别设置ncnn的链接库和头文件
set(NCNN_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libncnn.a)
set(NCNN_MI_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libMachineIndependent.a)
set(NCNN_OSD_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libOSDependent.a)
set(NCNN_GC_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libGenericCodeGen.a)
set(NCNN_SPIRV_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libSPIRV.a)
set(NCNN_GLS_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libglslang.a)
set(NCNN_OGLC_LIBS /home/user/3rdParty/ncnn_gpu/build/install/lib/libOGLCompiler.a)

set(NCNN_INCLUDE_DIRS /home/user/3rdParty/ncnn_gpu/build/install/include/ncnn)

include_directories(
	${NCNN_INCLUDE_DIRS}
	${NCNN_GLSLANG_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/parse-pose.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(${PROJECT_NAME} ncnn)

target_link_libraries(${PROJECT_NAME}
	${NCNN_LIBS}
	${NCNN_MI_LIBS}
	${NCNN_OSD_LIBS}
	${NCNN_GC_LIBS}
	${NCNN_SPIRV_LIBS}
	${NCNN_GLS_LIBS}
	${NCNN_OGLC_LIBS}
	vulkan
)
