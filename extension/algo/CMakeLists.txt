include_directories(
        ${PROJECT_SOURCE_DIR}/src/include
        ${CMAKE_BINARY_DIR}/src/include
        src/include)

add_subdirectory(src/main)
add_subdirectory(src/function)
add_subdirectory(src/common)
add_subdirectory(test)

build_extension_lib(${BUILD_STATIC_EXTENSION} "algo")

# The algo extension links the vendored GVE-Leiden backend, which needs the
# OpenMP runtime (libgomp). lbug_algo_function is an OBJECT library, so the link
# dependency must be attached to the final extension target here. Works for both
# the SHARED (dynamic-load) and STATIC (statically-linked) build variants.
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    if (BUILD_STATIC_EXTENSION)
        target_link_libraries(lbug_algo_static_extension PUBLIC OpenMP::OpenMP_CXX)
    else ()
        target_link_libraries(lbug_algo_extension PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endif ()
