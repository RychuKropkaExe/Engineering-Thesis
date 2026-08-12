file(GLOB Fitness *.cpp *.h)
target_sources(MainBuild PRIVATE ${Fitness})

target_include_directories(MainBuild PRIVATE "MmseEval")

add_subdirectory(MmseEval)
