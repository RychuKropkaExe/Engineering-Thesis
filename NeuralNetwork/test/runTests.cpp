#include "matrixOperationsTest.h"
#include "modelTest.h"
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
    modelTests();
}
