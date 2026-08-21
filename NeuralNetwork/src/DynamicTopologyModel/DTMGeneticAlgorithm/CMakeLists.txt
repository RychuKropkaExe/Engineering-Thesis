target_include_directories(MainBuild PRIVATE "DTMAlgorithm")
target_include_directories(MainBuild PRIVATE "DTIndividual")
target_include_directories(MainBuild PRIVATE "Hyperparameters")

add_subdirectory(DTMAlgorithm)
add_subdirectory(DTIndividual)
add_subdirectory(Hyperparameters)
