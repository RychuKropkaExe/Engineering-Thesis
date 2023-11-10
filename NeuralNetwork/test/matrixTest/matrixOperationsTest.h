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

void mulTest1(){
    FastMatrix expectedResult(4,4, 16.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 4, 2.0f);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 1 PASSED!\n";
}

void mulTest2(){
    FastMatrix expectedResult(4,2, 16.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 2, 2.0f);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 2 PASSED!\n";
}

void testMatrixMultiplication(){
    mulTest1();
    mulTest2();
}

void matrixOperationTest(){
    testMatrixAddition();
    testMatrixMultiplication();
}