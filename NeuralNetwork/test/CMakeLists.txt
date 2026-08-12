file(GLOB test *.cpp *.h)
target_sources(MainBuild PRIVATE ${test})

target_include_directories(MainBuild PRIVATE "TestUtils")

add_subdirectory(TestUtils)
