file(GLOB Crossover *.cpp *.h)
target_sources(MainBuild PRIVATE ${Crossover})

target_include_directories(MainBuild PRIVATE "weightedCrossover")
target_include_directories(MainBuild PRIVATE "noCrossover")

add_subdirectory(weightedCrossover)
add_subdirectory(noCrossover)
