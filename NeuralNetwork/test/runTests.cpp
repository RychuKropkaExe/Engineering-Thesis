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
#include "trackmaniaSimulatorTest.h"
#include <gtest/gtest.h>
#include <iostream>
#include <time.h>

#define BATCH_SIZE 64

int main(int argc, char **argv)
{
    try
    {
        testing::InitGoogleTest(&argc, argv);
#ifdef LOGGING_ACTIVATED
        // Initialize on the producer thread; the worker waits for a full buffer.
        TimeStampLogger &timeStampLogger = TimeStampLogger::getInstance();
        timeStampLogger.start();
#endif
        const int result = RUN_ALL_TESTS();
#ifdef LOGGING_ACTIVATED
        // Publish the partial buffer, drain all timestamps and join before exit.
        timeStampLogger.stop();
#endif
        return result;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Test runner failed: " << error.what() << '\n';
        return 1;
    }
}
