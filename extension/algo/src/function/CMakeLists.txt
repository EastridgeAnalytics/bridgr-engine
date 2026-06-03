add_subdirectory(config)

add_library(lbug_algo_function
        OBJECT
        component_ids.cpp
        strongly_connected_components.cpp
        strongly_connected_components_kosaraju.cpp
        weakly_connected_components.cpp
        page_rank.cpp
        k_core_decomposition.cpp
        louvain.cpp
        leiden.cpp
        gve_leiden.cpp
        spanning_forest.cpp
        )

# The GVE-Leiden community-detection backend (vendored under src/include/gve,
# MIT) is OpenMP-parallel and selects its edge-weight type at compile time via
# -DTYPE. These flags must apply ONLY to gve_leiden.cpp: TYPE collides with the
# engine (it is used both as a template parameter name in factorized_table.h and
# as a config field name in the *_config.h headers), so defining it globally
# breaks every other algo TU. set_source_files_properties scopes it to the one
# TU that actually includes the GVE headers.
find_package(OpenMP REQUIRED)
# -w silences the vendored GVE headers' warnings (-Wall -Wextra is on globally);
# they are third-party header-only code and would otherwise add noise and break a
# future -Werror build. Scoped to this TU only.
set_source_files_properties(gve_leiden.cpp PROPERTIES
        COMPILE_DEFINITIONS "TYPE=float"
        COMPILE_OPTIONS "${OpenMP_CXX_FLAGS};-w")

set(ALGO_EXTENSION_OBJECT_FILES
        ${ALGO_EXTENSION_OBJECT_FILES} $<TARGET_OBJECTS:lbug_algo_function>
        PARENT_SCOPE)
