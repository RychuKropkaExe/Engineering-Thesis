#include "FastMatrix.h"
#include <cassert>
void testMatrixAddition();
void testMatrixMultiplication();

void testMatrixAddition(){
    FastMatrix expectedResult(4,4, 4.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 4, 2.0f);
    FastMatrix result = m1+m2;
    assert(result == expectedResult);
    std::cout << "MATRIX ADDITION TEST PASSED!\n";
}

void testMatrixMultiplication(){
    FastMatrix expectedResult(4,4, 16.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 4, 2.0f);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST PASSED!\n";
}

void matrixOperationTest(){
    testMatrixAddition();
    testMatrixMultiplication();
}