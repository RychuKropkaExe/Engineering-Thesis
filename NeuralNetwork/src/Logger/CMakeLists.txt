find_package(Threads REQUIRED)
file(GLOB Logger CONFIGURE_DEPENDS *.h *.cpp)
target_sources(MainBuild PRIVATE ${Logger})
target_link_libraries(MainBuild Threads::Threads)
