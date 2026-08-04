file(GLOB SelectionAlgorithm *.cpp *.h)
target_sources(MainBuild PRIVATE ${SelectionAlgorithm})

target_include_directories(MainBuild PRIVATE "TournamentSelection")
target_include_directories(MainBuild PRIVATE "RouletteSelection")
target_include_directories(MainBuild PRIVATE "Parents")

add_subdirectory(TournamentSelection)
add_subdirectory(RouletteSelection)
add_subdirectory(Parents)
