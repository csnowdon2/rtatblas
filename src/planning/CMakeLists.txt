add_library(planning INTERFACE)
target_link_libraries(planning INTERFACE methods)
target_include_directories(planning INTERFACE ${CMAKE_CURRENT_LIST_DIR})
