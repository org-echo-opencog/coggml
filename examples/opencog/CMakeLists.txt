add_executable(opencog-inference opencog-inference.cpp)
target_link_libraries(opencog-inference PRIVATE ggml)
target_compile_features(opencog-inference PRIVATE cxx_std_11)