#ifndef MODEL_TEST_H
#define MODEL_TEST_H
#include "model.h"
#include "testFramework.h"
#include "trainingData.h"
#include <string>
#include <vector>
using std::vector;

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to model XOR logic gate
 ******************************************************************************/
void xorModelTest()
{
    TEST_START;
    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/TestData/xorData.txt"));
    vector<size_t> arch = {2, 2, 4, 1};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double eps = 1e-3;
    double learningRate = 1e-1f;

    model.setEps(eps);
    model.setLearningRate(learningRate);
    model.learn(td, 100000, true, 0);

    double cost = model.costMeanSquare();

    MY_TEST_ASSERT(cost < 0.05f, cost);
    TEST_RESULT();
}

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to model parabole on numbers in (-20, 20)
 ******************************************************************************/
void paraboleModelTest()
{
    TEST_START;
    vector<size_t> arch = {1, 10, 10, 1};

    vector<vector<double>> trainingInputs;
    vector<vector<double>> trainingOutputs;

    size_t numberOfSamples = 1000;

    size_t inputSize = 1;
    size_t outputSize = 1;

    trainingInputs.resize(numberOfSamples);
    trainingOutputs.resize(numberOfSamples);

    for (size_t i = 0; i < numberOfSamples; i++)
    {
        trainingInputs[i].resize(inputSize);
        trainingOutputs[i].resize(outputSize);
    }

    for (size_t i = 0; i < numberOfSamples; i++)
    {
        double inputValue = -20.f + ((double)i / 25.f);
        trainingInputs[i][0] = inputValue;
        trainingOutputs[i][0] = inputValue * inputValue;
    }

    TrainingData td = TrainingData(trainingInputs, inputSize, numberOfSamples, trainingOutputs, outputSize, numberOfSamples);

    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {RELU, RELU, RELU};

    Model model(arch, archSize, actFunc, archSize, true);

    model.modelXavierInitialize();

    double eps = 1e-3;
    double learningRate = 1e-2;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 1000000, true, 32);
    double cost = model.costMeanSquare();

    MY_TEST_ASSERT(cost < 100.f, cost);
    TEST_RESULT();
}

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to predict if given 8-bit number is
 *        even or odd
 ******************************************************************************/
void parityModelTest()
{
    TEST_START;
    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/TestData/parityTestData.txt"));
    vector<size_t> arch = {8, 8, 1};
    size_t archSize = 3;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double eps = 1e-1;
    double learningRate = 1e-1;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 10000, false, 0);

    double cost = model.costMeanSquare();

    MY_TEST_ASSERT(cost < 0.05f, cost);
    TEST_RESULT();
}

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to calculate hamming length of 7-bit number
 ******************************************************************************/
void hammingLengthTest()
{
    TEST_START;
    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/TestData/hammingLengthTest.txt"));
    vector<size_t> arch = {7, 10, 10, 3};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double learningRate = 1e-1;

    model.setLearningRate(learningRate);

    model.learn(td, 300000, false, 32);
    double cost = model.costMeanSquare();

    MY_TEST_ASSERT(cost < 0.10f, cost);
    TEST_RESULT();
}

/******************************************************************************
 * @brief Tests if NeuralNetwork is able to recognize digits given their
 *        features
 ******************************************************************************/
void digitRecognitionTest()
{
    TEST_START;
    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/TestData/pendigits.tra"));
    td.normalizeData(MIN_MAX_NORMALIZATION);
    vector<size_t> arch = {16, 10, 10, 1};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {RELU, RELU, RELU};

    Model model(arch, archSize, actFunc, archSize, true);
    model.modelXavierInitialize();

    double learningRate = 1e-3;

    model.setLearningRate(learningRate);

    model.learn(td, 100000, true, 128);

    td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/TestData/pendigits.tes"));
    td.normalizeData(MIN_MAX_NORMALIZATION);
    model.trainingData = td;
    double cost = model.costMeanSquare();
    for (size_t i = 0; i < td.numOfSamples; i++)
    {
        LOG(INFO_LEVEL, "PREDICTION FOR SAMPLE: " << td.inputs[i]);
        FastMatrix prediction = model.run(td.inputs[i]);
        td.denomralizeOutput(MIN_MAX_NORMALIZATION, prediction);
        LOG(INFO_LEVEL, "PREDICTION RESULT: " << prediction);
    }

    LOG(INFO_LEVEL, "CURRENT MODEL: " << model);

    LOG(INFO_LEVEL, "COST VALUE: " << cost);

    MY_TEST_ASSERT(cost < 0.10f, cost);
    TEST_RESULT();
}

void modelTests()
{
    TEST_SET;
    xorModelTest();
    parityModelTest();
    hammingLengthTest();
    paraboleModelTest();
    digitRecognitionTest();
}

#endif
