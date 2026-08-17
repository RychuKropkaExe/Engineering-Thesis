cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

message("DEBUG_MODE=${DEBUG_MODE}")
message("LOG_PRIO=${LOG_PRIO}")

set(COMPILE_FLAGS "-Wall -Wextra -Werror -DLOGGING_ACTIVATED")

if(DEBUG_MODE)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -O0 -ggdb3 -g -DDEBUG_PRIO=1")
else()
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -O3")
endif()

if(LOG_PRIO GREATER 0)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -DDEBUG_PRIO=${LOG_PRIO}")
endif()

project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${COMPILE_FLAGS}")

add_executable(MainBuild)

target_include_directories(MainBuild PRIVATE "test/testUtils")

add_subdirectory(src)
add_subdirectory(test)

target_link_libraries(
  MainBuild
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(MainBuild)
