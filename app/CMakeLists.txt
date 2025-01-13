set (GLAD_SRC "../external/glad/src/glad.c")
include_directories("../external/glad/include")

if(MSVC)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

set(APP_SRC 
    "src/app.h" "src/app.cpp"
    "src/frame.h" "src/frame.cpp" 
    "src/graphCanvas.h" "src/graphCanvas.cpp" 
    "src/graphPanel.h" "src/graphPanel.cpp"
    "src/graphManager.h" "src/graphManager.cpp"
    "src/configDialog.h" "src/configDialog.cpp" 
    "src/configDefaults.h" "src/graphDrawingSettings.h"
)

add_executable (app ${APP_SRC} ${GLAD_SRC})
set_target_properties(app PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "61" CXX_STANDARD 20)

target_link_libraries(app core image pattern utils wx::core wx::base wx::gl)
target_include_directories(app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(app PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>)

add_custom_target(copy-shaders ALL 
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders" 
	COMMENT "Copying shaders"
)
add_dependencies(app copy-shaders)
