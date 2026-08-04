target_include_directories(MainBuild PRIVATE "Individual")
target_include_directories(MainBuild PRIVATE "Algorithm")
target_include_directories(MainBuild PRIVATE "Selection")

add_subdirectory(Individual)
add_subdirectory(Algorithm)
add_subdirectory(Selection)
