#include "matrixOperationsTest.h"
#include "modelTest.h"
#include "testFramework.h"
#include "trainingDataTest.h"
#include <iostream>
#include <time.h>

inline uint32_t random_u32(uint32_t prev)
{
    return prev * 1664525U + 1013904223U; // assuming complement-2 integers and non-signaling overflow
}

#define BATCH_SIZE 64

int main()
{
    uint32_t time_ui = uint32_t(time(NULL));
    srand(time_ui);
    matrixOperationTest();
    trainingDataTest();
    modelTests();
    std::cout
        << c_blue << "\nAsserts Summary: " << assertPassed << " out of " << (assertPassed + assertFailed) << " asserts passed" << c_reset << "\n";
    std::cout
        << c_blue << "\nTests Summary: " << passed << " out of " << (passed + failed) << " tests passed" << c_reset << "\n";
}
