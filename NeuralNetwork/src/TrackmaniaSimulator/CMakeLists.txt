target_sources(MainBuild PRIVATE
  trackmaniaSimulator.cpp
  trackmaniaSimulator.h
  replayExport.cpp
  replayExport.h
)
target_include_directories(MainBuild PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# MainBuild already links ForeverValidator::native (which supplies core).
target_link_libraries(MainBuild ZLIB::ZLIB)
