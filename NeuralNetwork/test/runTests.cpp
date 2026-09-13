#include "logger.h"
#include "matrixOperationsTest.h"
// #include "modelTest.h"
#include "trainingDataTest.h"
#include "dtmodelTest.h"
#include "dtmodelTopologyTest.h"
#include "dtmodelFeedForwardTest.h"
#include "dtmgeneticCrossoverTest.h"
#include "dtmgeneticFitnessTest.h"
#include "dtmgeneticMutationTest.h"
#include "dtmgeneticParameterMutationTest.h"
#include "dtmgeneticAlgorithmTest.h"
#include <gtest/gtest.h>
#include <iostream>
#include <time.h>

#define BATCH_SIZE 64

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
