set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest) 

set(wxBUILD_SHARED OFF)
add_subdirectory(wxWidgets)

add_subdirectory(minorminer)

add_subdirectory(vf2lib)