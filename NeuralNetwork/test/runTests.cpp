#include "logger.h"
#include "matrixOperationsTest.h"
#include "modelTest.h"
#include "trainingDataTest.h"
#include <gtest/gtest.h>
#include <iostream>
#include <time.h>

std::ofstream Logger::logFile = std::ofstream("logs.log");

#define BATCH_SIZE 64

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
