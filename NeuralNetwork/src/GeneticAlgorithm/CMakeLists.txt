target_include_directories(MainBuild PRIVATE "Individual")
target_include_directories(MainBuild PRIVATE "Algorithm")

add_subdirectory(Individual)
add_subdirectory(Algorithm)
