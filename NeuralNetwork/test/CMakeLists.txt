file(GLOB test CONFIGURE_DEPENDS *.cpp *.h)
target_sources(MainBuild PRIVATE ${test})

target_include_directories(MainBuild PRIVATE "TestUtils")

add_subdirectory(TestUtils)
