target_include_directories(MainBuild PRIVATE "Logger")
target_include_directories(MainBuild PRIVATE "TrainingData")
target_include_directories(MainBuild PRIVATE "FixedTopologyModel")
target_include_directories(MainBuild PRIVATE "DynamicTopologyModel")

add_subdirectory(Logger)
add_subdirectory(TrainingData)
add_subdirectory(FixedTopologyModel)
add_subdirectory(DynamicTopologyModel)
