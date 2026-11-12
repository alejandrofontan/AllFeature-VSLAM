# feature_lab: the debug / fine-tune / train tooling for keypoint extraction and feature matching.
#
# Added from the top-level CMakeLists.txt (add_subdirectory), so everything here links the SLAM library
# itself: its headers (include/ is a PUBLIC include directory of ${PROJECT_NAME}), the compiled feature
# extractors and matchers, and its dependencies (OpenCV, ONNX Runtime, PoseLib, Eigen, yaml-cpp) come
# through that one link. Executables land in bin/ like the others. Not installed.

# mimalloc first on the link line, as for every executable (see the mimalloc section of the top-level file).
add_executable(feature_benchmark feature_benchmark.cpp)
# PoseLib and placecell's MegaLoc embedder directly too: the geometric check of the matches and the VPR similarity
# matrix call them (the SLAM library links both as well). MegaLoc is built whenever ALLFEATURE_WITH_MEGALOC is ON,
# its default on every platform; feature_benchmark needs it.
if(NOT ALLFEATURE_WITH_MEGALOC)
  message(FATAL_ERROR "tools/feature_lab needs ALLFEATURE_WITH_MEGALOC=ON (the MegaLoc VPR matrix of feature_benchmark)")
endif()
target_link_libraries(feature_benchmark ${ALLFEATURE_ALLOCATOR} ${PROJECT_NAME} PoseLib::PoseLib placecell::megaloc)
