cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("DEBUG_MODE=${DEBUG_MODE}")
message("LOG_PRIO=${LOG_PRIO}")

set(COMPILE_FLAGS "-Wall -Wextra -Werror -O3 -DLOGGING_ACTIVATED -DDEBUG_PRIO=1")

if(DEBUG_MODE)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -ggdb3 -g")
endif()

if(LOG_PRIO GREATER 0)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -DDEBUG_PRIO=${LOG_PRIO}")
endif()

project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${COMPILE_FLAGS}")

add_executable(MainBuild)

target_include_directories(MainBuild PRIVATE "src/FastMatrix")
target_include_directories(MainBuild PRIVATE "src/Layer")
target_include_directories(MainBuild PRIVATE "src/Logger")
target_include_directories(MainBuild PRIVATE "src/Model")
target_include_directories(MainBuild PRIVATE "src/TrainingData")

target_include_directories(MainBuild PRIVATE "test/testUtils")

add_subdirectory(src)
add_subdirectory(test)
