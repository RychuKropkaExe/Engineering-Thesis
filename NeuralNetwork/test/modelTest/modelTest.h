#ifndef MODEL_TEST_H
#define MODEL_TEST_H
#include <vector>
#include "trainingData.h"
#include "model.h"
#include <string>
using std::vector;

void xorModelTest(){

    TrainingData td = TrainingData(std::string("/home/rychu/Engineering-Thesis/NeuralNetwork/test/modelTest/xorData.txt"));
    td.printTrainingData();
    vector<size_t> arch = {2, 2, 4, 1};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    // model.printModel();

    float eps = 1e-3;
    float learningRate = 1e-1f;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 100000);

    //model.printModelToFile("/home/rychu/Engineering-Thesis/NeuralNetwork/printedModel.log");

    assert(model.cost() < 0.05f);
}

void parityModelTest(){

    vector<vector<float>> trainingInputs {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 0, 1, 0 },
        { 0, 0, 0, 0, 0, 0, 1, 1 },
        { 0, 0, 0, 0, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0, 1, 0, 1 },
        { 0, 0, 0, 0, 0, 1, 1, 0 },
        { 0, 0, 0, 0, 0, 1, 1, 1 },
        { 0, 0, 0, 0, 1, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 0, 0, 1 },
        { 0, 0, 0, 0, 1, 0, 1, 0 },
        { 0, 0, 0, 0, 1, 0, 1, 1 },
        { 0, 0, 0, 0, 1, 1, 0, 0 },
        { 0, 0, 0, 0, 1, 1, 0, 1 },
        { 0, 0, 0, 0, 1, 1, 1, 0 },
        { 0, 0, 0, 0, 1, 1, 1, 1 },
        { 0, 0, 0, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 0, 1, 0, 0, 1, 0 },
        { 0, 0, 0, 1, 0, 0, 1, 1 },
        { 0, 0, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 0, 1, 0, 1, 0, 1 },
        { 0, 0, 0, 1, 0, 1, 1, 0 },
        { 0, 0, 0, 1, 0, 1, 1, 1 },
        { 0, 0, 0, 1, 1, 0, 0, 0 },
        { 0, 0, 0, 1, 1, 0, 0, 1 },
        { 0, 0, 0, 1, 1, 0, 1, 0 },
        { 0, 0, 0, 1, 1, 0, 1, 1 },
        { 0, 0, 0, 1, 1, 1, 0, 0 },
        { 0, 0, 0, 1, 1, 1, 0, 1 },
        { 0, 0, 0, 1, 1, 1, 1, 0 },
        { 0, 0, 0, 1, 1, 1, 1, 1 },
        { 0, 0, 1, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 0, 0, 0, 1 },
        { 0, 0, 1, 0, 0, 0, 1, 0 },
        { 0, 0, 1, 0, 0, 0, 1, 1 },
        { 0, 0, 1, 0, 0, 1, 0, 0 },
        { 0, 0, 1, 0, 0, 1, 0, 1 },
        { 0, 0, 1, 0, 0, 1, 1, 0 },
        { 0, 0, 1, 0, 0, 1, 1, 1 },
        { 0, 0, 1, 0, 1, 0, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 0, 1 },
        { 0, 0, 1, 0, 1, 0, 1, 0 },
        { 0, 0, 1, 0, 1, 0, 1, 1 },
        { 0, 0, 1, 0, 1, 1, 0, 0 },
        { 0, 0, 1, 0, 1, 1, 0, 1 },
        { 0, 0, 1, 0, 1, 1, 1, 0 },
        { 0, 0, 1, 0, 1, 1, 1, 1 },
        { 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 0, 1 },
        { 0, 0, 1, 1, 0, 0, 1, 0 },
        { 0, 0, 1, 1, 0, 0, 1, 1 },
        { 0, 0, 1, 1, 0, 1, 0, 0 },
        { 0, 0, 1, 1, 0, 1, 0, 1 },
        { 0, 0, 1, 1, 0, 1, 1, 0 },
        { 0, 0, 1, 1, 0, 1, 1, 1 },
        { 0, 0, 1, 1, 1, 0, 0, 0 },
        { 0, 0, 1, 1, 1, 0, 0, 1 },
        { 0, 0, 1, 1, 1, 0, 1, 0 },
        { 0, 0, 1, 1, 1, 0, 1, 1 },
        { 0, 0, 1, 1, 1, 1, 0, 0 },
        { 0, 0, 1, 1, 1, 1, 0, 1 },
        { 0, 0, 1, 1, 1, 1, 1, 0 },
        { 0, 0, 1, 1, 1, 1, 1, 1 },
        { 0, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 1 },
        { 0, 1, 0, 0, 0, 0, 1, 0 },
        { 0, 1, 0, 0, 0, 0, 1, 1 },
        { 0, 1, 0, 0, 0, 1, 0, 0 },
        { 0, 1, 0, 0, 0, 1, 0, 1 },
        { 0, 1, 0, 0, 0, 1, 1, 0 },
        { 0, 1, 0, 0, 0, 1, 1, 1 },
        { 0, 1, 0, 0, 1, 0, 0, 0 },
        { 0, 1, 0, 0, 1, 0, 0, 1 },
        { 0, 1, 0, 0, 1, 0, 1, 0 },
        { 0, 1, 0, 0, 1, 0, 1, 1 },
        { 0, 1, 0, 0, 1, 1, 0, 0 },
        { 0, 1, 0, 0, 1, 1, 0, 1 },
        { 0, 1, 0, 0, 1, 1, 1, 0 },
        { 0, 1, 0, 0, 1, 1, 1, 1 },
        { 0, 1, 0, 1, 0, 0, 0, 0 },
        { 0, 1, 0, 1, 0, 0, 0, 1 },
        { 0, 1, 0, 1, 0, 0, 1, 0 },
        { 0, 1, 0, 1, 0, 0, 1, 1 },
        { 0, 1, 0, 1, 0, 1, 0, 0 },
        { 0, 1, 0, 1, 0, 1, 0, 1 },
        { 0, 1, 0, 1, 0, 1, 1, 0 },
        { 0, 1, 0, 1, 0, 1, 1, 1 },
        { 0, 1, 0, 1, 1, 0, 0, 0 },
        { 0, 1, 0, 1, 1, 0, 0, 1 },
        { 0, 1, 0, 1, 1, 0, 1, 0 },
        { 0, 1, 0, 1, 1, 0, 1, 1 },
        { 0, 1, 0, 1, 1, 1, 0, 0 },
        { 0, 1, 0, 1, 1, 1, 0, 1 },
        { 0, 1, 0, 1, 1, 1, 1, 0 },
        { 0, 1, 0, 1, 1, 1, 1, 1 },
        { 0, 1, 1, 0, 0, 0, 0, 0 },
        { 0, 1, 1, 0, 0, 0, 0, 1 },
        { 0, 1, 1, 0, 0, 0, 1, 0 },
        { 0, 1, 1, 0, 0, 0, 1, 1 },
        { 0, 1, 1, 0, 0, 1, 0, 0 },
        { 0, 1, 1, 0, 0, 1, 0, 1 },
        { 0, 1, 1, 0, 0, 1, 1, 0 },
        { 0, 1, 1, 0, 0, 1, 1, 1 },
        { 0, 1, 1, 0, 1, 0, 0, 0 },
        { 0, 1, 1, 0, 1, 0, 0, 1 },
        { 0, 1, 1, 0, 1, 0, 1, 0 },
        { 0, 1, 1, 0, 1, 0, 1, 1 },
        { 0, 1, 1, 0, 1, 1, 0, 0 },
        { 0, 1, 1, 0, 1, 1, 0, 1 },
        { 0, 1, 1, 0, 1, 1, 1, 0 },
        { 0, 1, 1, 0, 1, 1, 1, 1 },
        { 0, 1, 1, 1, 0, 0, 0, 0 },
        { 0, 1, 1, 1, 0, 0, 0, 1 },
        { 0, 1, 1, 1, 0, 0, 1, 0 },
        { 0, 1, 1, 1, 0, 0, 1, 1 },
        { 0, 1, 1, 1, 0, 1, 0, 0 },
        { 0, 1, 1, 1, 0, 1, 0, 1 },
        { 0, 1, 1, 1, 0, 1, 1, 0 },
        { 0, 1, 1, 1, 0, 1, 1, 1 },
        { 0, 1, 1, 1, 1, 0, 0, 0 },
        { 0, 1, 1, 1, 1, 0, 0, 1 },
        { 0, 1, 1, 1, 1, 0, 1, 0 },
        { 0, 1, 1, 1, 1, 0, 1, 1 },
        { 0, 1, 1, 1, 1, 1, 0, 0 },
        { 0, 1, 1, 1, 1, 1, 0, 1 },
        { 0, 1, 1, 1, 1, 1, 1, 0 },
        { 0, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 1, 0 },
        { 1, 0, 0, 0, 0, 0, 1, 1 },
        { 1, 0, 0, 0, 0, 1, 0, 0 },
        { 1, 0, 0, 0, 0, 1, 0, 1 },
        { 1, 0, 0, 0, 0, 1, 1, 0 },
        { 1, 0, 0, 0, 0, 1, 1, 1 },
        { 1, 0, 0, 0, 1, 0, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 0, 1 },
        { 1, 0, 0, 0, 1, 0, 1, 0 },
        { 1, 0, 0, 0, 1, 0, 1, 1 },
        { 1, 0, 0, 0, 1, 1, 0, 0 },
        { 1, 0, 0, 0, 1, 1, 0, 1 },
        { 1, 0, 0, 0, 1, 1, 1, 0 },
        { 1, 0, 0, 0, 1, 1, 1, 1 },
        { 1, 0, 0, 1, 0, 0, 0, 0 },
        { 1, 0, 0, 1, 0, 0, 0, 1 },
        { 1, 0, 0, 1, 0, 0, 1, 0 },
        { 1, 0, 0, 1, 0, 0, 1, 1 },
        { 1, 0, 0, 1, 0, 1, 0, 0 },
        { 1, 0, 0, 1, 0, 1, 0, 1 },
        { 1, 0, 0, 1, 0, 1, 1, 0 },
        { 1, 0, 0, 1, 0, 1, 1, 1 },
        { 1, 0, 0, 1, 1, 0, 0, 0 },
        { 1, 0, 0, 1, 1, 0, 0, 1 },
        { 1, 0, 0, 1, 1, 0, 1, 0 },
        { 1, 0, 0, 1, 1, 0, 1, 1 },
        { 1, 0, 0, 1, 1, 1, 0, 0 },
        { 1, 0, 0, 1, 1, 1, 0, 1 },
        { 1, 0, 0, 1, 1, 1, 1, 0 },
        { 1, 0, 0, 1, 1, 1, 1, 1 },
        { 1, 0, 1, 0, 0, 0, 0, 0 },
        { 1, 0, 1, 0, 0, 0, 0, 1 },
        { 1, 0, 1, 0, 0, 0, 1, 0 },
        { 1, 0, 1, 0, 0, 0, 1, 1 },
        { 1, 0, 1, 0, 0, 1, 0, 0 },
        { 1, 0, 1, 0, 0, 1, 0, 1 },
        { 1, 0, 1, 0, 0, 1, 1, 0 },
        { 1, 0, 1, 0, 0, 1, 1, 1 },
        { 1, 0, 1, 0, 1, 0, 0, 0 },
        { 1, 0, 1, 0, 1, 0, 0, 1 },
        { 1, 0, 1, 0, 1, 0, 1, 0 },
        { 1, 0, 1, 0, 1, 0, 1, 1 },
        { 1, 0, 1, 0, 1, 1, 0, 0 },
        { 1, 0, 1, 0, 1, 1, 0, 1 },
        { 1, 0, 1, 0, 1, 1, 1, 0 },
        { 1, 0, 1, 0, 1, 1, 1, 1 },
        { 1, 0, 1, 1, 0, 0, 0, 0 },
        { 1, 0, 1, 1, 0, 0, 0, 1 },
        { 1, 0, 1, 1, 0, 0, 1, 0 },
        { 1, 0, 1, 1, 0, 0, 1, 1 },
        { 1, 0, 1, 1, 0, 1, 0, 0 },
        { 1, 0, 1, 1, 0, 1, 0, 1 },
        { 1, 0, 1, 1, 0, 1, 1, 0 },
        { 1, 0, 1, 1, 0, 1, 1, 1 },
        { 1, 0, 1, 1, 1, 0, 0, 0 },
        { 1, 0, 1, 1, 1, 0, 0, 1 },
        { 1, 0, 1, 1, 1, 0, 1, 0 },
        { 1, 0, 1, 1, 1, 0, 1, 1 },
        { 1, 0, 1, 1, 1, 1, 0, 0 },
        { 1, 0, 1, 1, 1, 1, 0, 1 },
        { 1, 0, 1, 1, 1, 1, 1, 0 },
        { 1, 0, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 0, 0, 0, 0, 0, 0 },
        { 1, 1, 0, 0, 0, 0, 0, 1 },
        { 1, 1, 0, 0, 0, 0, 1, 0 },
        { 1, 1, 0, 0, 0, 0, 1, 1 },
        { 1, 1, 0, 0, 0, 1, 0, 0 },
        { 1, 1, 0, 0, 0, 1, 0, 1 },
        { 1, 1, 0, 0, 0, 1, 1, 0 },
        { 1, 1, 0, 0, 0, 1, 1, 1 },
        { 1, 1, 0, 0, 1, 0, 0, 0 },
        { 1, 1, 0, 0, 1, 0, 0, 1 },
        { 1, 1, 0, 0, 1, 0, 1, 0 },
        { 1, 1, 0, 0, 1, 0, 1, 1 },
        { 1, 1, 0, 0, 1, 1, 0, 0 },
        { 1, 1, 0, 0, 1, 1, 0, 1 },
        { 1, 1, 0, 0, 1, 1, 1, 0 },
        { 1, 1, 0, 0, 1, 1, 1, 1 },
        { 1, 1, 0, 1, 0, 0, 0, 0 },
        { 1, 1, 0, 1, 0, 0, 0, 1 },
        { 1, 1, 0, 1, 0, 0, 1, 0 },
        { 1, 1, 0, 1, 0, 0, 1, 1 },
        { 1, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 1, 0, 1, 0, 1, 0, 1 },
        { 1, 1, 0, 1, 0, 1, 1, 0 },
        { 1, 1, 0, 1, 0, 1, 1, 1 },
        { 1, 1, 0, 1, 1, 0, 0, 0 },
        { 1, 1, 0, 1, 1, 0, 0, 1 },
        { 1, 1, 0, 1, 1, 0, 1, 0 },
        { 1, 1, 0, 1, 1, 0, 1, 1 },
        { 1, 1, 0, 1, 1, 1, 0, 0 },
        { 1, 1, 0, 1, 1, 1, 0, 1 },
        { 1, 1, 0, 1, 1, 1, 1, 0 },
        { 1, 1, 0, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 0, 0, 0, 0, 0 },
        { 1, 1, 1, 0, 0, 0, 0, 1 },
        { 1, 1, 1, 0, 0, 0, 1, 0 },
        { 1, 1, 1, 0, 0, 0, 1, 1 },
        { 1, 1, 1, 0, 0, 1, 0, 0 },
        { 1, 1, 1, 0, 0, 1, 0, 1 },
        { 1, 1, 1, 0, 0, 1, 1, 0 },
        { 1, 1, 1, 0, 0, 1, 1, 1 },
        { 1, 1, 1, 0, 1, 0, 0, 0 },
        { 1, 1, 1, 0, 1, 0, 0, 1 },
        { 1, 1, 1, 0, 1, 0, 1, 0 },
        { 1, 1, 1, 0, 1, 0, 1, 1 },
        { 1, 1, 1, 0, 1, 1, 0, 0 },
        { 1, 1, 1, 0, 1, 1, 0, 1 },
        { 1, 1, 1, 0, 1, 1, 1, 0 },
        { 1, 1, 1, 0, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 0, 0, 0, 0 },
        { 1, 1, 1, 1, 0, 0, 0, 1 },
        { 1, 1, 1, 1, 0, 0, 1, 0 },
        { 1, 1, 1, 1, 0, 0, 1, 1 },
        { 1, 1, 1, 1, 0, 1, 0, 0 },
        { 1, 1, 1, 1, 0, 1, 0, 1 },
        { 1, 1, 1, 1, 0, 1, 1, 0 },
        { 1, 1, 1, 1, 0, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 0, 0, 0 },
        { 1, 1, 1, 1, 1, 0, 0, 1 },
        { 1, 1, 1, 1, 1, 0, 1, 0 },
        { 1, 1, 1, 1, 1, 0, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 0, 0 },
        { 1, 1, 1, 1, 1, 1, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 0 },
        { 1, 1, 1, 1, 1, 1, 1, 1 }
    };

    vector<vector<float>> trainingOutputs {
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0}
    };

    TrainingData td = TrainingData(trainingInputs, 8, 256, trainingOutputs, 1, 256);
    vector<size_t> arch = {8, 8, 1};
    size_t archSize = 3;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    //model.printModel();

    float eps = 1e-1;
    float learningRate = 1e-1;

    model.setEps(eps);
    model.setLearningRate(learningRate);

    model.learn(td, 10000);

    //model.printModel();

    // for(size_t i = 0; i < td.numOfSamples; ++i){
    //     std::cout << "FOR INPUT: " << "\n";
    //     printFastMatrix(td.inputs[i]);
    //     std::cout << "OUTPUT IS: " << "\n";
    //     FastMatrix result = model.run(td.inputs[i]);
    //     printFastMatrix(result);
    // }

    assert(model.cost() < 0.05f);
}

void hammingLengthTest(){
    vector<vector<float>> trainingInputs {
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

    vector<vector<float>> trainingOutputs {
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

    TrainingData td = TrainingData(trainingInputs, 7, 128, trainingOutputs, 3, 128);
    vector<size_t> arch = {7, 7, 10, 3};
    size_t archSize = 4;
    vector<ActivationFunctionE> actFunc = {SIGMOID, SIGMOID, SIGMOID};

    Model model(arch, archSize, actFunc, archSize, true);

    //model.printModel();

    float learningRate = 1e-1;

    model.setLearningRate(learningRate);

    model.learn(td, 100000);
    
    // for(size_t i = 0; i < td.numOfSamples; ++i){
    //     std::cout << "FOR INPUT: " << "\n";
    //     printFastMatrix(td.inputs[i]);
    //     std::cout << "OUTPUT IS: " << "\n";
    //     FastMatrix result = model.run(td.inputs[i]);
    //     printFastMatrix(result);
    // }

    assert(model.cost() < 0.05f);
}

void parsingTest(){
    Model model = parseModelFromFile("/home/rychu/Engineering-Thesis/NeuralNetwork/printedModel.log");
    model.printModel();
    vector<float> v {0.f,1.f};
    FastMatrix input(v, 2, ROW_VECTOR);
    FastMatrix res = model.run(input);
    std::cout << "FOR INPUT:" << "\n";
    printFastMatrix(input);
    std::cout << "RESULT IS: " << "\n";
    printFastMatrix(res);
}

void modelTests(){
    xorModelTest();
    //parityModelTest();
    //hammingLengthTest();
    //parsingTest();
}

#endif