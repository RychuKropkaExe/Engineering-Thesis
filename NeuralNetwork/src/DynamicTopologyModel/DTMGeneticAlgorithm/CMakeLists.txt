target_include_directories(MainBuild PRIVATE "DTMAlgorithm")
target_include_directories(MainBuild PRIVATE "DTIndividual")
target_include_directories(MainBuild PRIVATE "Hyperparameters")
target_include_directories(MainBuild PRIVATE "GAParents")

add_subdirectory(DTMAlgorithm)
add_subdirectory(DTIndividual)
add_subdirectory(Hyperparameters)
add_subdirectory(GAParents)
