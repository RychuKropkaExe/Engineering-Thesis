file(GLOB Mutation *.cpp *.h)
target_sources(MainBuild PRIVATE ${Mutation})

target_include_directories(MainBuild PRIVATE "FlipWeight")
target_include_directories(MainBuild PRIVATE "FlipBias")
target_include_directories(MainBuild PRIVATE "ChangeActivation")
target_include_directories(MainBuild PRIVATE "AdjustWeight")
target_include_directories(MainBuild PRIVATE "AddNeuron")

add_subdirectory(FlipWeight)
add_subdirectory(FlipBias)
add_subdirectory(ChangeActivation)
add_subdirectory(AdjustWeight)
add_subdirectory(AddNeuron)
