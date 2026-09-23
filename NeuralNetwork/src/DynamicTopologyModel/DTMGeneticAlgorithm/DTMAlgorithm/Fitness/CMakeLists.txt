target_sources(MainBuild PRIVATE dtmfitnessEvaluation.h)
target_include_directories(MainBuild PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(MeanSquaredEval)
