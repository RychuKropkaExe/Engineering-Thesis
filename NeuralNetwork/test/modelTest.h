#ifndef MODEL_TEST_H
#define MODEL_TEST_H
#include <vector>
#include "trainingData.h"
#include "model.h"
#include <string>
using std::vector;

void xorModelTest(){

    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/xorData.txt"));
    vector<size_t> arch = {2, 2, 4, 1};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double eps = 1e-3;
    double learningRate = 1e-1f;

    model.setEps(eps);
    model.setLearningRate(learningRate);
    model.learn(td, 100000, false);

    assert(model.costMeanSquare() < 0.05f);
}

void paraboleModelTest(){
    vector<size_t> arch = {1, 10, 10, 10, 1};

    vector<vector<double>> trainingInputs;
    vector<vector<double>> trainingOutputs;

    size_t numberOfSamples = 1000;

    size_t inputSize = 1;
    size_t outputSize = 1;

    trainingInputs.resize(numberOfSamples);
    trainingOutputs.resize(numberOfSamples);

    for(size_t i = 0; i < numberOfSamples; i++){
        trainingInputs[i].resize(inputSize);
        trainingOutputs[i].resize(outputSize);
    }

    for(size_t i = 0; i < numberOfSamples; i++){

        double inputValue = -20.f + ((double)i/25.f);

        trainingInputs[i][0] = inputValue;
        trainingOutputs[i][0] = inputValue*inputValue;

    }

    TrainingData td = TrainingData(trainingInputs, inputSize, numberOfSamples, trainingOutputs, outputSize, numberOfSamples);

    size_t archSize = 5;
    vector<ActivationFunctionE> actFunc = {RELU, RELU, RELU, RELU};

    Model model(arch, archSize, actFunc, archSize, true);

    model.modelXavierInitialize();

    double eps = 1e-3;
    double learningRate = 1e-3;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 100000, true);

    for(size_t i = 0; i < td.numOfSamples; ++i){
        std::cout << "FOR INPUT: " << std::endl;
        printFastMatrix(td.inputs[i]);
        std::cout << "OUTPUT IS: " << std::endl;
        FastMatrix result = model.run(td.inputs[i]);
        printFastMatrix(result);
    }

    assert(model.costMeanSquare() < 100.f);
}


void parityModelTest(){

    TrainingData td = TrainingData(std::string("C:/Users/Rychu/Desktop/Projekty/Trackmania/Engineering-Thesis/NeuralNetwork/test/parityTestData.txt"));
    vector<size_t> arch = {8, 8, 1};
    size_t archSize = 3;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double eps = 1e-1;
    double learningRate = 1e-1;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 10000, false);

    assert(model.costMeanSquare() < 0.05f);
}

void hammingLengthTest(){
    vector<vector<double>> trainingInputs {
        { 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 1, 0 },
        { 0, 0, 0, 0, 0, 1, 1 },
        { 0, 0, 0, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 1, 0, 1 },
        { 0, 0, 0, 0, 1, 1, 0 },
        { 0, 0, 0, 0, 1, 1, 1 },
        { 0, 0, 0, 1, 0, 0, 0 },
        { 0, 0, 0, 1, 0, 0, 1 },
        { 0, 0, 0, 1, 0, 1, 0 },
        { 0, 0, 0, 1, 0, 1, 1 },
        { 0, 0, 0, 1, 1, 0, 0 },
        { 0, 0, 0, 1, 1, 0, 1 },
        { 0, 0, 0, 1, 1, 1, 0 },
        { 0, 0, 0, 1, 1, 1, 1 },
        { 0, 0, 1, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 1, 0, 0, 1, 0 },
        { 0, 0, 1, 0, 0, 1, 1 },
        { 0, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 1 },
        { 0, 0, 1, 0, 1, 1, 0 },
        { 0, 0, 1, 0, 1, 1, 1 },
        { 0, 0, 1, 1, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 1 },
        { 0, 0, 1, 1, 0, 1, 0 },
        { 0, 0, 1, 1, 0, 1, 1 },
        { 0, 0, 1, 1, 1, 0, 0 },
        { 0, 0, 1, 1, 1, 0, 1 },
        { 0, 0, 1, 1, 1, 1, 0 },
        { 0, 0, 1, 1, 1, 1, 1 },
        { 0, 1, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 1 },
        { 0, 1, 0, 0, 0, 1, 0 },
        { 0, 1, 0, 0, 0, 1, 1 },
        { 0, 1, 0, 0, 1, 0, 0 },
        { 0, 1, 0, 0, 1, 0, 1 },
        { 0, 1, 0, 0, 1, 1, 0 },
        { 0, 1, 0, 0, 1, 1, 1 },
        { 0, 1, 0, 1, 0, 0, 0 },
        { 0, 1, 0, 1, 0, 0, 1 },
        { 0, 1, 0, 1, 0, 1, 0 },
        { 0, 1, 0, 1, 0, 1, 1 },
        { 0, 1, 0, 1, 1, 0, 0 },
        { 0, 1, 0, 1, 1, 0, 1 },
        { 0, 1, 0, 1, 1, 1, 0 },
        { 0, 1, 0, 1, 1, 1, 1 },
        { 0, 1, 1, 0, 0, 0, 0 },
        { 0, 1, 1, 0, 0, 0, 1 },
        { 0, 1, 1, 0, 0, 1, 0 },
        { 0, 1, 1, 0, 0, 1, 1 },
        { 0, 1, 1, 0, 1, 0, 0 },
        { 0, 1, 1, 0, 1, 0, 1 },
        { 0, 1, 1, 0, 1, 1, 0 },
        { 0, 1, 1, 0, 1, 1, 1 },
        { 0, 1, 1, 1, 0, 0, 0 },
        { 0, 1, 1, 1, 0, 0, 1 },
        { 0, 1, 1, 1, 0, 1, 0 },
        { 0, 1, 1, 1, 0, 1, 1 },
        { 0, 1, 1, 1, 1, 0, 0 },
        { 0, 1, 1, 1, 1, 0, 1 },
        { 0, 1, 1, 1, 1, 1, 0 },
        { 0, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 1, 0 },
        { 1, 0, 0, 0, 0, 1, 1 },
        { 1, 0, 0, 0, 1, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 1 },
        { 1, 0, 0, 0, 1, 1, 0 },
        { 1, 0, 0, 0, 1, 1, 1 },
        { 1, 0, 0, 1, 0, 0, 0 },
        { 1, 0, 0, 1, 0, 0, 1 },
        { 1, 0, 0, 1, 0, 1, 0 },
        { 1, 0, 0, 1, 0, 1, 1 },
        { 1, 0, 0, 1, 1, 0, 0 },
        { 1, 0, 0, 1, 1, 0, 1 },
        { 1, 0, 0, 1, 1, 1, 0 },
        { 1, 0, 0, 1, 1, 1, 1 },
        { 1, 0, 1, 0, 0, 0, 0 },
        { 1, 0, 1, 0, 0, 0, 1 },
        { 1, 0, 1, 0, 0, 1, 0 },
        { 1, 0, 1, 0, 0, 1, 1 },
        { 1, 0, 1, 0, 1, 0, 0 },
        { 1, 0, 1, 0, 1, 0, 1 },
        { 1, 0, 1, 0, 1, 1, 0 },
        { 1, 0, 1, 0, 1, 1, 1 },
        { 1, 0, 1, 1, 0, 0, 0 },
        { 1, 0, 1, 1, 0, 0, 1 },
        { 1, 0, 1, 1, 0, 1, 0 },
        { 1, 0, 1, 1, 0, 1, 1 },
        { 1, 0, 1, 1, 1, 0, 0 },
        { 1, 0, 1, 1, 1, 0, 1 },
        { 1, 0, 1, 1, 1, 1, 0 },
        { 1, 0, 1, 1, 1, 1, 1 },
        { 1, 1, 0, 0, 0, 0, 0 },
        { 1, 1, 0, 0, 0, 0, 1 },
        { 1, 1, 0, 0, 0, 1, 0 },
        { 1, 1, 0, 0, 0, 1, 1 },
        { 1, 1, 0, 0, 1, 0, 0 },
        { 1, 1, 0, 0, 1, 0, 1 },
        { 1, 1, 0, 0, 1, 1, 0 },
        { 1, 1, 0, 0, 1, 1, 1 },
        { 1, 1, 0, 1, 0, 0, 0 },
        { 1, 1, 0, 1, 0, 0, 1 },
        { 1, 1, 0, 1, 0, 1, 0 },
        { 1, 1, 0, 1, 0, 1, 1 },
        { 1, 1, 0, 1, 1, 0, 0 },
        { 1, 1, 0, 1, 1, 0, 1 },
        { 1, 1, 0, 1, 1, 1, 0 },
        { 1, 1, 0, 1, 1, 1, 1 },
        { 1, 1, 1, 0, 0, 0, 0 },
        { 1, 1, 1, 0, 0, 0, 1 },
        { 1, 1, 1, 0, 0, 1, 0 },
        { 1, 1, 1, 0, 0, 1, 1 },
        { 1, 1, 1, 0, 1, 0, 0 },
        { 1, 1, 1, 0, 1, 0, 1 },
        { 1, 1, 1, 0, 1, 1, 0 },
        { 1, 1, 1, 0, 1, 1, 1 },
        { 1, 1, 1, 1, 0, 0, 0 },
        { 1, 1, 1, 1, 0, 0, 1 },
        { 1, 1, 1, 1, 0, 1, 0 },
        { 1, 1, 1, 1, 0, 1, 1 },
        { 1, 1, 1, 1, 1, 0, 0 },
        { 1, 1, 1, 1, 1, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 0 },
        { 1, 1, 1, 1, 1, 1, 1 }
    };

    vector<vector<double>> trainingOutputs {
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 1},
        {1, 1, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 1},
        {1, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 1},
        {1, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 1},
        {1, 1, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 0},
        {1, 1, 1}
    };

    TrainingData td = TrainingData(trainingInputs, (size_t)7, (size_t)128, trainingOutputs, (size_t)3, (size_t)128);
    vector<size_t> arch = {7, 10, 10, 5, 3};
    size_t archSize = 5;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    double learningRate = 1e-1;

    model.setLearningRate(learningRate);

    model.learn(td, 200000, false);

    assert(model.costMeanSquare() < 0.10f);
}

void parsingTest(){
    Model model = parseModelFromFile("/home/rychu/Engineering-Thesis/NeuralNetwork/printedModel.log");
    model.printModel();
    vector<double> v {0.f,1.f};
    FastMatrix input(v, (size_t)2, ROW_VECTOR);
    FastMatrix res = model.run(input);
    std::cout << "FOR INPUT:" << std::endl;
    printFastMatrix(input);
    std::cout << "RESULT IS: " << std::endl;
    printFastMatrix(res);
}

void modelTests(){

    xorModelTest();
    parityModelTest();
    hammingLengthTest();
    paraboleModelTest();
    //parsingTest();
}

#endif