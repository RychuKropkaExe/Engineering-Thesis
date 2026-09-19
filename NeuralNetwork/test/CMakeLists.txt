file(GLOB test CONFIGURE_DEPENDS *.cpp *.h)
target_sources(MainBuild PRIVATE ${test})

target_include_directories(MainBuild PRIVATE "TestUtils")

# Replay metadata regression checks need the complete vehicle Ident. The
# public validation report exposes only the model enum, not collection/author.
# Keep this internal dependency confined to the test translation unit.
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/runTests.cpp"
  TARGET_DIRECTORY MainBuild APPEND PROPERTY
  INCLUDE_DIRECTORIES "${forevervalidator_SOURCE_DIR}/src")

add_subdirectory(TestUtils)
