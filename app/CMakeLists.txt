find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

set(GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only")
set(GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only")
set(GLAD_GL "")

option(GLFW-CMAKE-STARTER-USE-GLFW-GLAD "Use GLAD from GLFW" ON)

if(GLFW-CMAKE-STARTER-USE-GLFW-GLAD)
    include_directories("../external/glfw/deps")
    set(GLAD_GL "../external/glfw/deps/glad/gl.h")
endif()

if(MSVC)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

set(APP_SRC "app.cpp" "cudaTest.cu" "cudaTest.h" "render.h" "render.cpp")

add_executable (app ${APP_SRC} ${GLAD_GL})
set_target_properties(app PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "50" CXX_STANDARD 17)

target_link_libraries(app core image pattern ${OPENGL_LIBRARIES} glfw)
target_include_directories(app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
