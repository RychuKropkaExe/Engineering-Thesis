target_include_directories(MainBuild PRIVATE "Individual")
target_include_directories(MainBuild PRIVATE "Algorithm")
target_include_directories(MainBuild PRIVATE "Selection")
target_include_directories(MainBuild PRIVATE "Crossover")

add_subdirectory(Individual)
add_subdirectory(Algorithm)
add_subdirectory(Selection)
add_subdirectory(Crossover)
