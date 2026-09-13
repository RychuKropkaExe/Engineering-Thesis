file(GLOB DTMAlgorithm CONFIGURE_DEPENDS *.cpp *.h)
target_sources(MainBuild PRIVATE ${DTMAlgorithm})
