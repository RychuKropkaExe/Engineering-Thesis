#include "logger.h"
#include "matrixOperationsTest.h"
#include "modelTest.h"
#include "testFramework.h"
#include "trainingDataTest.h"
#include <iostream>
#include <time.h>

std::ofstream Logger::logFile = std::ofstream("logs.log");

#define BATCH_SIZE 64

int main()
{
    uint32_t time_ui = uint32_t(time(NULL));
    srand(time_ui);
    matrixOperationTest();
    trainingDataTest();
    modelTests();
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "Asserts Summary: " << assertPassed << " out of " << (assertPassed + assertFailed) << " asserts passed");
    LOG(ESSENTIAL_LOGS, INFO_TYPE, "Tests Summary: " << passed << " out of " << (passed + failed) << " tests passed");
}
