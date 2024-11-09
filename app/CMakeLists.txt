set (GLAD_SRC "../external/glad/src/glad.c")
include_directories("../external/glad/include")

if(MSVC)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

set(APP_SRC 
    "app.h" "app.cpp"
    "frame.h" "frame.cpp" 
    "graphCanvas.h" "graphCanvas.cpp" 
    "graphPanel.h" "graphPanel.cpp"
)

add_executable (app ${APP_SRC} ${GLAD_SRC})
set_target_properties(app PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "50" CXX_STANDARD 17)

target_link_libraries(app core image pattern utils wx::core wx::base wx::gl)
target_include_directories(app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(app PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>)
